open Torch

type t = {
  num_factors: int;
  hidden_size: int;
  dropout: float;
  device: Device.t;
  stock_embedder: Module.t;
  encoder: Module.t;
  prior_nu: Tensor.t;
  prior_sigma: Tensor.t;
}

let create ?(num_factors=64) ?(hidden_size=256) ?(dropout=0.25) ?(device=Device.Cpu) () =
  let stock_embedder = 
    Module.sequential
      [
        Module.lstm ~input_dim:(1 + 256) ~hidden_size ~num_layers:1 ~batch_first:true ();
        Module.linear ~input_dim:hidden_size ~output_dim:hidden_size ();
        Module.relu ();
        Module.dropout ~p:dropout ();
        Module.linear ~input_dim:hidden_size ~output_dim:hidden_size ();
        Module.relu ();
        Module.dropout ~p:dropout ();
      ]
  in
  let encoder =
    Module.sequential
      [
        Module.linear ~input_dim:hidden_size ~output_dim:hidden_size ();
        Module.relu ();
        Module.dropout ~p:dropout ();
        Module.linear ~input_dim:hidden_size ~output_dim:(4 * num_factors) ();
      ]
  in
  let prior_nu = Tensor.full [num_factors] 5. ~device in
  let prior_sigma = Tensor.ones [num_factors] ~device in
  { num_factors; hidden_size; dropout; device; stock_embedder; encoder; prior_nu; prior_sigma }

let softplus x =
  Tensor.log1p (Tensor.exp x)

let student_t_log_prob mu sigma nu x =
  let z = (x - mu) / sigma in
  let log_const = Tensor.log (Tensor.tgamma ((nu +. 1.) /. 2.)) -. 
                  Tensor.log (Tensor.tgamma (nu /. 2.)) -. 
                  0.5 *. Tensor.log (Tensor.mul_scalar nu Float.pi) -. 
                  Tensor.log sigma in
  let log_kernel = -. ((nu +. 1.) /. 2.) *. Tensor.log1p (Tensor.pow z 2. /. nu) in
  log_const +. log_kernel

let stock_embed model x_ts x_static =
  let batch_size, seq_len, _ = Tensor.shape x_ts in
  let h = Module.forward model.stock_embedder x_ts in
  let h = Tensor.narrow h ~dim:1 ~start:(seq_len - 1) ~length:1 in
  let h = Tensor.squeeze h ~dim:1 in
  Tensor.cat [h; x_static] ~dim:1

let encode model h =
  let output = Module.forward model.encoder h in
  let alpha = Tensor.narrow output ~dim:1 ~start:0 ~length:model.num_factors in
  let beta = Tensor.narrow output ~dim:1 ~start:model.num_factors ~length:model.num_factors in
  let log_sigma = Tensor.narrow output ~dim:1 ~start:(2 * model.num_factors) ~length:model.num_factors in
  let log_nu = Tensor.narrow output ~dim:1 ~start:(3 * model.num_factors) ~length:model.num_factors in
  let sigma = softplus log_sigma in
  let nu = Tensor.add (softplus log_nu) (Tensor.full_like log_nu 4.) in
  (alpha, beta, sigma, nu)

let decode model alpha beta sigma nu z =
  let mu = Tensor.add alpha (Tensor.matmul beta z) in
  (mu, sigma, nu)

let sample_latent model batch_size =
  let z = Tensor.randn [batch_size; model.num_factors] ~device:model.device in
  let z = Tensor.mul z model.prior_sigma in
  z

let kl_divergence mu_q sigma_q nu_q mu_p sigma_p nu_p =
  let log_det_ratio = Tensor.log (Tensor.div sigma_p sigma_q) in
  let trace_term = Tensor.div (Tensor.pow sigma_q 2.) (Tensor.pow sigma_p 2.) in
  let mu_diff_sq = Tensor.pow (Tensor.sub mu_q mu_p) 2. in
  let mu_term = Tensor.div mu_diff_sq (Tensor.pow sigma_p 2.) in
  let nu_term = Tensor.sub (Tensor.digamma (Tensor.div nu_q 2.)) (Tensor.digamma (Tensor.div nu_p 2.)) in
  let nu_log_term = Tensor.sub (Tensor.log (Tensor.div nu_q 2.)) (Tensor.log (Tensor.div nu_p 2.)) in
  let kl = Tensor.add (Tensor.add log_det_ratio trace_term) (Tensor.add mu_term (Tensor.add nu_term nu_log_term)) in
  Tensor.div kl 2.

let loss model x_ts x_static y =
  let batch_size = Tensor.shape x_ts |> List.hd in
  let h = stock_embed model x_ts x_static in
  let (alpha, beta, sigma, nu) = encode model h in
  let z = sample_latent model batch_size in
  let (mu, sigma, nu) = decode model alpha beta sigma nu z in
  let log_prob = student_t_log_prob mu sigma nu y in
  let kl_div = kl_divergence (Tensor.zeros_like mu) sigma nu Tensor.zeros model.prior_sigma model.prior_nu in
  Tensor.mean (Tensor.neg log_prob) +. Tensor.mean kl_div

let train model optimizer x_ts x_static y =
  Optimizer.zero_grad optimizer;
  let loss = loss model x_ts x_static y in
  Tensor.backward loss;
  Optimizer.step optimizer;
  Tensor.float_value loss

let train_epoch model optimizer data =
  let total_loss = ref 0. in
  let batches = Data.batch_data data 32 in
  List.iter (fun batch ->
    try
      let loss = train model optimizer batch.Data.x_ts batch.Data.x_static batch.Data.y in
      total_loss := !total_loss +. loss
    with
    | _ -> Printf.eprintf "Error processing batch, skipping\n"
  ) batches;
  !total_loss /. float_of_int (List.length batches)

let evaluate model data =
  let batches = Data.batch_data data 32 in
  let total_loss = ref 0. in
  List.iter (fun batch ->
    try
      let loss = loss model batch.Data.x_ts batch.Data.x_static batch.Data.y in
      total_loss := !total_loss +. Tensor.float_value loss
    with
    | _ -> Printf.eprintf "Error evaluating batch, skipping\n"
  ) batches;
  !total_loss /. float_of_int (List.length batches)

let predict model x_ts x_static =
  try
    let h = stock_embed model x_ts x_static in
    let (alpha, beta, sigma, nu) = encode model h in
    let z = sample_latent model (Tensor.shape x_ts |> List.hd) in
    let (mu, _, _) = decode model alpha beta sigma nu z in
    mu
  with
  | _ -> failwith "Error making predictions"

let train_model model optimizer num_epochs data_train data_val =
  for epoch = 1 to num_epochs do
    try
      let train_loss = train_epoch model optimizer data_train in
      let val_loss = evaluate model data_val in
      Printf.printf "Epoch %d: Train Loss = %f, Val Loss = %f\n" epoch train_loss val_loss;
    with
    | _ -> Printf.eprintf "Error in epoch %d, skipping\n" epoch
  done

let parameters model =
  Module.parameters model.stock_embedder @ Module.parameters model.encoder

let named_parameters model =
  Module.named_parameters model.stock_embedder @ Module.named_parameters model.encoder
open Torch

type optimizer_config = {
  learning_rate: float;
  momentum: float option;
  beta1: float option;
  beta2: float option;
  weight_decay: float;
  grad_clip: float option;
}

type optimizer_state = {
  momentum_buffer: (string, Tensor.t) Hashtbl.t;
  exp_avg: (string, Tensor.t) Hashtbl.t;
  exp_avg_sq: (string, Tensor.t) Hashtbl.t;
  step: int;
}

type t = {
  parameters: Tensor.t list;
  config: optimizer_config;
  mutable state: optimizer_state;
}

let create config network =
  let state = {
    momentum_buffer = Hashtbl.create 16;
    exp_avg = Hashtbl.create 16;
    exp_avg_sq = Hashtbl.create 16;
    step = 0;
  } in
  { parameters = WavKANNetwork.parameters network;
    config = config;
    state = state }

let clip_gradients params max_norm =
  let total_norm = List.fold_left (fun acc param ->
    match Tensor.grad param with
    | Some grad -> 
        acc +. (Tensor.float_value (Tensor.norm grad ~p:(Tensor.of_float 2.0)))
    | None -> acc
  ) 0. params in
  if total_norm > max_norm then
    let scale = max_norm /. total_norm in
    List.iter (fun param ->
      match Tensor.grad param with
      | Some grad -> Tensor.mul_scalar_ grad scale
      | None -> ()
    ) params

let step t =
  t.state <- { t.state with step = t.state.step + 1 };
  let step = float t.state.step in
  
  (* Apply gradient clipping if configured *)
  (match t.config.grad_clip with
   | Some clip_value -> clip_gradients t.parameters clip_value
   | None -> ());

  List.iter (fun param ->
    match Tensor.grad param with
    | Some grad ->
        let param_id = Tensor.name param in
        
        (* Momentum update *)
        (match t.config.momentum with
         | Some beta ->
             let v = try Hashtbl.find t.state.momentum_buffer param_id
                    with Not_found -> Tensor.zeros_like grad in
             let v_new = Tensor.(v * (f beta) + grad * (f (1. -. beta))) in
             Hashtbl.replace t.state.momentum_buffer param_id v_new;
             Tensor.copy_ v_new ~src:grad
         | None -> ());

        (* Adam update *)
        (match (t.config.beta1, t.config.beta2) with
         | Some beta1, Some beta2 ->
             let m = try Hashtbl.find t.state.exp_avg param_id
                    with Not_found -> Tensor.zeros_like grad in
             let v = try Hashtbl.find t.state.exp_avg_sq param_id
                    with Not_found -> Tensor.zeros_like grad in
             
             (* Update biased first moment estimate *)
             let m' = Tensor.(m * (f beta1) + grad * (f (1. -. beta1))) in
             Hashtbl.replace t.state.exp_avg param_id m';
             
             (* Update biased second moment estimate *)
             let v' = Tensor.(v * (f beta2) + 
                             (grad * grad) * (f (1. -. beta2))) in
             Hashtbl.replace t.state.exp_avg_sq param_id v';
             
             (* Bias correction *)
             let m_hat = Tensor.div m' (Tensor.f (1. -. (beta1 ** step))) in
             let v_hat = Tensor.div v' (Tensor.f (1. -. (beta2 ** step))) in
             
             (* Parameter update *)
             let update = Tensor.div m_hat 
               Tensor.((sqrt v_hat) + (f 1e-8)) in
             param -= Tensor.mul_scalar update t.config.learning_rate
         | _ -> 
             (* Simple SGD update *)
             param -= Tensor.mul_scalar grad t.config.learning_rate);

        (* Weight decay *)
        if t.config.weight_decay > 0. then
          param *= Tensor.f (1. -. t.config.weight_decay *. 
                           t.config.learning_rate)
    | None -> ()
  ) t.parameters

let zero_grad t =
  List.iter (fun param -> Tensor.zero_grad param) t.parameters

module Regularization = struct
  open Torch

  type regularizer_type =
    | Smoothness of float
    | Sparsity of float
    | Energy of float
    | Orthogonality of float

  let compute_smoothness_penalty params coeff =
    List.fold_left (fun acc param ->
      let grad = Tensor.diff param ~dim:0 in
      acc + Tensor.mean (Tensor.pow grad (Tensor.of_float 2.0))
    ) (Tensor.zeros []) params
    |> fun x -> Tensor.mul_scalar x coeff

  let compute_sparsity_penalty params coeff =
    List.fold_left (fun acc param ->
      acc + Tensor.mean (Tensor.abs param)
    ) (Tensor.zeros []) params
    |> fun x -> Tensor.mul_scalar x coeff

  let compute_energy_penalty params coeff =
    List.fold_left (fun acc param ->
      let energy = Tensor.sum (Tensor.pow param (Tensor.of_float 2.0)) in
      acc + Tensor.abs (energy - Tensor.ones [])
    ) (Tensor.zeros []) params
    |> fun x -> Tensor.mul_scalar x coeff

  let compute_orthogonality_penalty params coeff =
    List.fold_left (fun acc param ->
      let gram = Tensor.mm param (Tensor.transpose param ~dim0:0 ~dim1:1) in
      let identity = Tensor.eye (Tensor.shape param |> List.hd) in
      acc + Tensor.mse_loss gram identity
    ) (Tensor.zeros []) params
    |> fun x -> Tensor.mul_scalar x coeff

  let compute_penalty params regularizers =
    List.fold_left (fun acc regularizer ->
      let penalty = match regularizer with
        | Smoothness coeff -> compute_smoothness_penalty params coeff
        | Sparsity coeff -> compute_sparsity_penalty params coeff
        | Energy coeff -> compute_energy_penalty params coeff
        | Orthogonality coeff -> compute_orthogonality_penalty params coeff
      in
      Tensor.(acc + penalty)
    ) (Tensor.zeros []) regularizers
end
open Torch

let create_feed_forward ~input_dim ~hidden_dim ~output_dim =
  let module N = Torch.Nn in
  N.Sequential.of_list [
    N.Linear.create ~in_dim:input_dim ~out_dim:hidden_dim;
    N.Relu.create;
    N.Linear.create ~in_dim:hidden_dim ~out_dim:hidden_dim;
    N.Relu.create;
    N.Linear.create ~in_dim:hidden_dim ~out_dim:output_dim;
  ]

let tensor_to_float t = Tensor.to_float0_exn t

let glorot_normal_init vs ~shape =
  let fan_in, fan_out =
    match shape with
    | [in_dim; out_dim] -> in_dim, out_dim
    | _ -> failwith "Invalid shape for Glorot normal initialization"
  in
  let std = sqrt (2.0 /. float_of_int (fan_in + fan_out)) in
  Var_store.new_var vs ~shape ~init:(Torch.Init.Normal { mean = 0.0; stdev = std })
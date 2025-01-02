open Torch

type t = {
  n1 : int;
  n2 : int;
  r1 : int;
  r2 : int;
  g : Tensor.t;
  c : Tensor.t;
}

let create n1 n2 r1 r2 =
  { n1; n2; r1; r2;
    g = Tensor.randn [n1; r1];
    c = Tensor.randn [n2; r2] }

let fit ?(max_iter=100) ?(tol=1e-4) ?(verbose=false) data model =
  let open Matrix_factor_models_utils in
  match check_input_shape data [model.n1; model.n2] with
  | Error e -> Error e
  | Ok () ->
      if verbose then Logger.log Logger.Info "Starting Tucker model fitting";
      let x1 = Tensor.reshape data ~shape:[model.n1; model.n2] in
      let u1, _, _ = Tensor.svd x1 ~some:true in
      let g = Tensor.narrow u1 ~dim:1 ~start:0 ~length:model.r1 in
      
      let x2 = Tensor.transpose data ~dim0:0 ~dim1:1 in
      let x2 = Tensor.reshape x2 ~shape:[model.n2; model.n1] in
      let u2, _, _ = Tensor.svd x2 ~some:true in
      let c = Tensor.narrow u2 ~dim:1 ~start:0 ~length:model.r2 in
      
      if verbose then Logger.log Logger.Info "Tucker model fitting completed";
      Ok { model with g; c }

let transform data model =
  let open Matrix_factor_models_utils in
  match check_input_shape data [model.n1; model.n2] with
  | Error e -> Error e
  | Ok () ->
      let g_t = Tensor.transpose model.g ~dim0:0 ~dim1:1 in
      let c_t = Tensor.transpose model.c ~dim0:0 ~dim1:1 in
      Ok (parallel_matmul (parallel_matmul data g_t) c_t)

let reconstruction_error data model =
  match transform data model with
  | Error e -> Error e
  | Ok reconstructed ->
      let error = Tensor.mse_loss data reconstructed in
      Ok (Tensor.to_float0_exn error)

let save model filename =
  try
    let state = [
      ("n1", Tensor.of_int0 model.n1);
      ("n2", Tensor.of_int0 model.n2);
      ("r1", Tensor.of_int0 model.r1);
      ("r2", Tensor.of_int0 model.r2);
      ("g", model.g);
      ("c", model.c);
    ] in
    Serialize.save ~filename state;
    Ok ()
  with
  | e -> Error (Printf.sprintf "Failed to save Tucker model: %s" (Printexc.to_string e))

let load filename =
  try
    let state = Serialize.load ~filename in
    let n1 = Tensor.to_int0_exn (List.assoc "n1" state) in
    let n2 = Tensor.to_int0_exn (List.assoc "n2" state) in
    let r1 = Tensor.to_int0_exn (List.assoc "r1" state) in
    let r2 = Tensor.to_int0_exn (List.assoc "r2" state) in
    let g = List.assoc "g" state in
    let c = List.assoc "c" state in
    Ok { n1; n2; r1; r2; g; c }
  with
  | e -> Error (Printf.sprintf "Failed to load Tucker model: %s" (Printexc.to_string e))
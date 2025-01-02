open Torch
open Lwt
open Lwt.Infix

type t = {
  n1 : int;
  n2 : int;
  r : int;
  a : Tensor.t;
  b : Tensor.t;
}

let create n1 n2 r =
  { n1; n2; r;
    a = Tensor.randn [n1; r];
    b = Tensor.randn [n2; r] }

let fit ?(max_iter=100) ?(tol=1e-4) ?(n_threads=4) ?(verbose=false) data model =
  let open Matrix_factor_models_utils in
  match check_input_shape data [model.n1; model.n2] with
  | Error e -> Error e
  | Ok () ->
      if verbose then Logger.log Logger.Info "Starting CP model fitting";
      let rec als_iteration a b iter prev_error =
        if iter >= max_iter then Lwt.return (a, b)
        else
          let b_t = Tensor.transpose b ~dim0:0 ~dim1:1 in
          let%lwt a_new = Lwt_list.map_p
            (fun i ->
               let chunk = Tensor.narrow data ~dim:0 ~start:(i * model.n1 / n_threads) ~length:(model.n1 / n_threads) in
               Lwt.return (parallel_matmul chunk b))
            (List.init n_threads (fun i -> i))
          >|= Tensor.cat ~dim:0 in
          let a_new = Tensor.div a_new (parallel_matmul b_t b) in
          let a_new = normalize_columns a_new in
          
          let a_t = Tensor.transpose a_new ~dim0:0 ~dim1:1 in
          let%lwt b_new = Lwt_list.map_p
            (fun i ->
               let chunk = Tensor.narrow (Tensor.transpose data ~dim0:0 ~dim1:1) ~dim:0 ~start:(i * model.n2 / n_threads) ~length:(model.n2 / n_threads) in
               Lwt.return (parallel_matmul chunk a_new))
            (List.init n_threads (fun i -> i))
          >|= Tensor.cat ~dim:0 in
          let b_new = Tensor.div b_new (parallel_matmul a_t a_new) in
          let b_new = normalize_columns b_new in
          
          let error = Tensor.frobenius_norm (Tensor.sub data (parallel_matmul a_new (Tensor.transpose b_new ~dim0:0 ~dim1:1))) in
          
          if verbose && iter mod 10 = 0 then
            Logger.log Logger.Info (Printf.sprintf "CP iteration %d, error: %f" iter (Tensor.to_float0_exn error));
          
          if Tensor.abs (Tensor.sub error prev_error) |> Tensor.to_float0_exn < tol then
            Lwt.return (a_new, b_new)
          else
            als_iteration a_new b_new (iter + 1) error
      in
      
      let%lwt a_init, b_init = als_iteration model.a model.b 0 (Tensor.of_float 1e10) in
      if verbose then Logger.log Logger.Info "CP model fitting completed";
      Lwt.return (Ok { model with a = a_init; b = b_init })

let transform data model =
  let open Matrix_factor_models_utils in
  match check_input_shape data [model.n1; model.n2] with
  | Error e -> Error e
  | Ok () ->
      let b_t = Tensor.transpose model.b ~dim0:0 ~dim1:1 in
      Ok (parallel_matmul (parallel_matmul data model.b) b_t)

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
      ("r", Tensor.of_int0 model.r);
      ("a", model.a);
      ("b", model.b);
    ] in
    Serialize.save ~filename state;
    Ok ()
  with
  | e -> Error (Printf.sprintf "Failed to save CP model: %s" (Printexc.to_string e))

let load filename =
  try
    let state = Serialize.load ~filename in
    let n1 = Tensor.to_int0_exn (List.assoc "n1" state) in
    let n2 = Tensor.to_int0_exn (List.assoc "n2" state) in
    let r = Tensor.to_int0_exn (List.assoc "r" state) in
    let a = List.assoc "a" state in
    let b = List.assoc "b" state in
    Ok { n1; n2; r; a; b }
  with
  | e -> Error (Printf.sprintf "Failed to load CP model: %s" (Printexc.to_string e))
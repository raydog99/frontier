open Torch
open Lwt
open Lwt.Infix

module Logger = Logger
module Tucker = Tucker
module CP = CP

let check_input_shape tensor expected_shape =
  let actual_shape = Tensor.shape tensor in
  if actual_shape <> expected_shape then
    Error (Printf.sprintf "Expected shape %s, but got %s"
             (String.concat "x" (List.map string_of_int expected_shape))
             (String.concat "x" (List.map string_of_int actual_shape)))
  else
    Ok ()

let normalize_columns tensor =
  let norms = Tensor.norm tensor ~dim:0 ~p:2 ~keepdim:true in
  Tensor.div tensor norms

let parallel_matmul a b =
  let n, m = Tensor.shape2_exn a in
  let _, p = Tensor.shape2_exn b in
  let chunk_size = max 1 (n / 4) in
  let rec process_chunks start acc =
    if start >= n then
      Tensor.cat acc ~dim:0
    else
      let end_idx = min (start + chunk_size) n in
      let chunk = Tensor.narrow a ~dim:0 ~start ~length:(end_idx - start) in
      let result = Tensor.matmul chunk b in
      process_chunks end_idx (result :: acc)
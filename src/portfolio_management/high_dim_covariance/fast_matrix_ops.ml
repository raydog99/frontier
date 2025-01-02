open Torch

let optimal_block_size d memory_limit =
  let element_size = 4 in (* bytes *)
  Int.of_float (sqrt (Float.of_int (memory_limit / element_size)))

let rect_multiply a b =
  let d = Tensor.size a 1 in
  match d with
  | d when d <= 1000 -> Tensor.mm a b
  | _ ->
      let block_size = optimal_block_size d (1024 * 1024 * 1024) in
      block_multiply a b block_size

let block_multiply a b block_size =
  let m, n = Tensor.size a 0, Tensor.size a 1 in
  let _, p = Tensor.size b 0, Tensor.size b 1 in
  let result = Tensor.zeros [|m; p|] in
  
  for i = 0 to (m + block_size - 1) / block_size - 1 do
    for j = 0 to (p + block_size - 1) / block_size - 1 do
      let start_m = i * block_size in
      let start_p = j * block_size in
      let size_m = min block_size (m - start_m) in
      let size_p = min block_size (p - start_p) in
      
      let block_result = Tensor.mm
        (Tensor.narrow a ~dim:0 ~start:start_m ~length:size_m)
        (Tensor.narrow b ~dim:1 ~start:start_p ~length:size_p) in
      
      Tensor.copy_ 
        (Tensor.narrow result ~dim:0 ~start:start_m ~length:size_m
           |> Tensor.narrow ~dim:1 ~start:start_p ~length:size_p)
        block_result
    done
  done;
  result

let transpose_multiply a b =
  let at = Tensor.transpose a ~dim0:0 ~dim1:1 in
  rect_multiply at b
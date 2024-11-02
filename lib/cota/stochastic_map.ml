open Torch

let create_from_plan plan =
  let row_sums = Tensor.(sum plan ~dim:[1] |> reshape ~shape:[-1; 1]) in
  let normalized_plan = Tensor.(div plan (row_sums + Tensor.full [1] 1e-10)) in
  fun x ->
    let x_hot = if Tensor.dim x = 1 then
      Tensor.one_hot x (Tensor.shape normalized_plan |> List.hd)
    else x in
    Tensor.matmul x_hot normalized_plan

let average_maps maps =
  fun x -> 
    let results = List.map (fun f -> f x) maps in
    let sum = List.fold_left Tensor.add (List.hd results) (List.tl results) in
    Tensor.(div sum (float_of_int (List.length maps)))
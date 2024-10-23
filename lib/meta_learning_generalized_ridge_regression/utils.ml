open Torch

let split_list lst n =
  let rec aux acc rest n =
    if n = 0 then (List.rev acc, rest)
    else match rest with
      | [] -> (List.rev acc, [])
      | x :: xs -> aux (x :: acc) xs (n-1)
  in
  aux [] lst n

let mean_std lst =
  let n = float_of_int (List.length lst) in
  let mean = List.fold_left (+.) 0. lst /. n in
  let var = List.fold_left (fun acc x -> 
    acc +. (x -. mean) ** 2.
  ) 0. lst /. n in
  (mean, sqrt var)

let list_split4 lst =
  let (l1, l2, l3, l4) = 
    List.fold_right (fun (a, b, c, d) (acc1, acc2, acc3, acc4) ->
      (a :: acc1, b :: acc2, c :: acc3, d :: acc4)
    ) lst ([], [], [], [])
  in
  (l1, l2, l3, l4)

let tensor_to_float_list tensor =
  let size = Tensor.size tensor |> List.hd in
  List.init size (fun i ->
    Tensor.get tensor [|i|] |> Tensor.to_float0_exn
  )

let generate_grid min_val max_val num_points =
  let step = (max_val -. min_val) /. float_of_int (num_points - 1) in
  List.init num_points (fun i ->
    min_val +. step *. float_of_int i
  )
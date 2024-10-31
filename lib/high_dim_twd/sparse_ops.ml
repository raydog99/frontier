open Torch

type sparse_tensor = {
  indices: Tensor.t;
  values: Tensor.t;
  size: int array;
}

let create_sparse indices values size = {indices; values; size}

let to_dense {indices; values; size} =
  let dense = Tensor.zeros size in
  let n = Tensor.size values 0 in
  for i = 0 to n - 1 do
    let row = Tensor.get indices [|0; i|] |> Int.of_float in
    let col = Tensor.get indices [|1; i|] |> Int.of_float in
    let value = Tensor.get values [|i|] in
    Tensor.set dense [|row; col|] value
  done;
  dense

let from_dense tensor sparsity_threshold =
  let rows, cols = Tensor.shape2_exn tensor in
  let indices = ref [] in
  let values = ref [] in
  
  for i = 0 to rows - 1 do
    for j = 0 to cols - 1 do
      let value = Tensor.get tensor [|i; j|] in
      if Float.abs value > sparsity_threshold then begin
        indices := (float_of_int i, float_of_int j) :: !indices;
        values := value :: !values
      end
    done
  done;
  
  let indices_tensor = Tensor.of_float2_list [
    List.map fst !indices;
    List.map snd !indices
  ] in
  let values_tensor = Tensor.of_float_list !values in
  
  create_sparse indices_tensor values_tensor [|rows; cols|]

let sparse_mm sp1 sp2 =
  let m, _ = sp1.size in
  let _, n = sp2.size in
  
  let result_indices = ref [] in
  let result_values = ref [] in
  
  (* Convert indices to hashable format for faster lookup *)
  let sp2_dict = Hashtbl.create 1024 in
  let n_values2 = Tensor.size sp2.values 0 in
  for i = 0 to n_values2 - 1 do
    let row = Tensor.get sp2.indices [|0; i|] |> Int.of_float in
    let col = Tensor.get sp2.indices [|1; i|] |> Int.of_float in
    let value = Tensor.get sp2.values [|i|] in
    Hashtbl.add sp2_dict (row, col) value
  done;
  
  let n_values1 = Tensor.size sp1.values 0 in
  for i = 0 to n_values1 - 1 do
    let row1 = Tensor.get sp1.indices [|0; i|] |> Int.of_float in
    let col1 = Tensor.get sp1.indices [|1; i|] |> Int.of_float in
    let val1 = Tensor.get sp1.values [|i|] in
    
    (* Multiply with matching elements in sp2 *)
    Hashtbl.iter (fun (row2, col2) val2 ->
      if col1 = row2 then begin
        result_indices := (float_of_int row1, float_of_int col2) :: !result_indices;
        result_values := (val1 *. val2) :: !result_values
      end
    ) sp2_dict
  done;
  
  let indices_tensor = Tensor.of_float2_list [
    List.map fst !result_indices;
    List.map snd !result_indices
  ] in
  let values_tensor = Tensor.of_float_list !result_values in
  
  create_sparse indices_tensor values_tensor [|m; n|]
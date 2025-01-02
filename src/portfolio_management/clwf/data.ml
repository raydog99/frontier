open Torch

type mask = Tensor.t

module TimeSeries = struct
  type t = {
    data: Tensor.t;
    mask: mask;
    dimensions: int;
    sequence_length: int;
  }

  let create ~data ~mask =
    let dims = Tensor.shape data in
    match dims with 
    | [_; d; l] -> { 
        data; 
        mask; 
        dimensions = d; 
        sequence_length = l 
      }
    | _ -> failwith "Invalid data dimensions"

  let split_condition_target ts ~target_mask =
    let open Tensor in
    let condition_mask = sub (ones_like target_mask) target_mask in
    let conditional = mul ts.data condition_mask in
    let target = mul ts.data target_mask in
    conditional, target, condition_mask, target_mask
    
  let generate_random_mask ts ~missing_ratio =
    let open Tensor in
    let mask_shape = shape ts.data in
    let mask = rand mask_shape in
    let threshold = scalar (1. -. missing_ratio) in
    ge mask threshold |> to_type ~type_:Float
end

module DataLoader = struct
  type t = {
    data: TimeSeries.t;
    batch_size: int;
    shuffle: bool;
    mutable current_idx: int;
  }

  let create ~data ~batch_size ~shuffle = {
    data;
    batch_size;
    shuffle;
    current_idx = 0;
  }

  let shuffle_indices n =
    let indices = Array.init n (fun i -> i) in
    for i = n - 1 downto 1 do
      let j = Random.int (i + 1) in
      let temp = indices.(i) in
      indices.(i) <- indices.(j);
      indices.(j) <- temp;
    done;
    Array.to_list indices

  let next_batch t =
    let open Tensor in
    let n = shape t.data.data |> List.hd in
    if t.current_idx >= n then None
    else
      let end_idx = min (t.current_idx + t.batch_size) n in
      let indices = 
        List.init (end_idx - t.current_idx) (fun i -> t.current_idx + i)
        |> List.map float_of_int in
      let batch_idx = of_float1 indices |> to_type ~type_:Int64 in
      let batch_data = index_select t.data.data ~dim:0 ~index:batch_idx in
      let batch_mask = index_select t.data.mask ~dim:0 ~index:batch_idx in
      t.current_idx <- end_idx;
      Some (TimeSeries.create ~data:batch_data ~mask:batch_mask)

  let reset t =
    t.current_idx <- 0
end
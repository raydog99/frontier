
open Torch
open Stdio
open Base
open Lwt

type dataset = {
x_ts: Tensor.t;
x_static: Tensor.t;
y: Tensor.t;
timestamps: float array;
}

exception Data_error of string

let load_data_parallel filename num_threads =
let load_chunk lines =
  let data = List.map lines ~f:(String.split ~on:',') in
  let n = List.length data in
  let x_ts_len = 256 in
  let x_static_len = 100 in
  
  let x_ts = Tensor.zeros [n; x_ts_len; 257] in
  let x_static = Tensor.zeros [n; x_static_len] in
  let y = Tensor.zeros [n; 1] in
  let timestamps = Array.create ~len:n 0. in
  
  List.iteri data ~f:(fun i row ->
    if List.length row <> (x_ts_len * 257 + x_static_len + 2) then
      raise (Data_error (Printf.sprintf "Invalid data format in row %d" (i + 1)));
    
    let x_ts_data = List.take row (x_ts_len * 257) in
    let x_static_data = List.slice row (x_ts_len * 257) (x_ts_len * 257 + x_static_len) in
    let y_data = List.nth_exn row (List.length row - 2) in
    let timestamp = Float.of_string (List.last_exn row) in
    
    for j = 0 to x_ts_len - 1 do
      for k = 0 to 256 do
        Tensor.set x_ts [|i; j; k|] (Float.of_string (List.nth_exn x_ts_data (j * 257 + k)))
      done;
    done;
    
    for j = 0 to x_static_len - 1 do
      Tensor.set x_static [|i; j|] (Float.of_string (List.nth_exn x_static_data j))
    done;
    
    Tensor.set y [|i; 0|] (Float.of_string y_data);
    timestamps.(i) <- timestamp
  );
  
  { x_ts; x_static; y; timestamps }
in

try
  let lines = In_channel.read_lines filename in
  let header = List.hd_exn lines in
  let data_lines = List.tl_exn lines in
  let chunk_size = (List.length data_lines + num_threads - 1) / num_threads in
  let chunks = List.chunks_of ~length:chunk_size data_lines in
  
  let loaded_chunks = 
    Lwt_list.map_p (fun chunk -> Lwt_preemptive.detach load_chunk chunk) chunks
    |> Lwt_main.run
  in
  
  let combine_datasets datasets =
    let total_size = List.sum (module Int) datasets ~f:(fun d -> List.hd_exn (Tensor.shape d.x_ts)) in
    let combined_x_ts = Tensor.zeros [total_size; 256; 257] in
    let combined_x_static = Tensor.zeros [total_size; 100] in
    let combined_y = Tensor.zeros [total_size; 1] in
    let combined_timestamps = Array.create ~len:total_size 0. in
    
    let rec combine offset = function
      | [] -> ()
      | d :: rest ->
          let size = List.hd_exn (Tensor.shape d.x_ts) in
          Tensor.narrow combined_x_ts ~dim:0 ~start:offset ~length:size |> Tensor.copy_ ~src:d.x_ts;
          Tensor.narrow combined_x_static ~dim:0 ~start:offset ~length:size |> Tensor.copy_ ~src:d.x_static;
          Tensor.narrow combined_y ~dim:0 ~start:offset ~length:size |> Tensor.copy_ ~src:d.y;
          Array.blit ~src:d.timestamps ~src_pos:0 ~dst:combined_timestamps ~dst_pos:offset ~len:size;
          combine (offset + size) rest
    in
    combine 0 datasets;
    { x_ts = combined_x_ts; x_static = combined_x_static; y = combined_y; timestamps = combined_timestamps }
  in
  
  combine_datasets loaded_chunks
with
| Sys_error msg -> raise (Data_error (Printf.sprintf "Error reading file: %s" msg))
| Failure msg -> raise (Data_error (Printf.sprintf "Error parsing data: %s" msg))

let preprocess data =
let remove_outliers tensor =
  let mean = Tensor.mean tensor ~dim:[0; 1] ~keepdim:true in
  let std = Tensor.std tensor ~dim:[0; 1] ~keepdim:true in
  let z_scores = Tensor.div (Tensor.sub tensor mean) std in
  let mask = Tensor.lt (Tensor.abs z_scores) (Tensor.of_float 3.0) in
  Tensor.mul tensor (Tensor.to_type mask ~type_:(Tensor.kind tensor))
in
{ 
  x_ts = remove_outliers data.x_ts;
  x_static = remove_outliers data.x_static;
  y = data.y;
  timestamps = data.timestamps;
}

let split_data_by_time data split_time =
let split_index = Array.binary_search data.timestamps `First_greater_than_or_equal_to split_time
                  |> Option.value_exn in
let train = {
  x_ts = Tensor.narrow data.x_ts ~dim:0 ~start:0 ~length:split_index;
  x_static = Tensor.narrow data.x_static ~dim:0 ~start:0 ~length:split_index;
  y = Tensor.narrow data.y ~dim:0 ~start:0 ~length:split_index;
  timestamps = Array.sub data.timestamps ~pos:0 ~len:split_index;
} in
let test = {
  x_ts = Tensor.narrow data.x_ts ~dim:0 ~start:split_index ~length:(Array.length data.timestamps - split_index);
  x_static = Tensor.narrow data.x_static ~dim:0 ~start:split_index ~length:(Array.length data.timestamps - split_index);
  y = Tensor.narrow data.y ~dim:0 ~start:split_index ~length:(Array.length data.timestamps - split_index);
  timestamps = Array.sub data.timestamps ~pos:split_index ~len:(Array.length data.timestamps - split_index);
} in
(train, test)

let batch_data data batch_size =
let n = Tensor.shape data.x_ts |> List.hd in
let num_batches = n / batch_size in
List.init num_batches (fun i ->
  let start = i * batch_size in
  {
    x_ts = Tensor.narrow data.x_ts ~dim:0 ~start ~length:batch_size;
    x_static = Tensor.narrow data.x_static ~dim:0 ~start ~length:batch_size;
    y = Tensor.narrow data.y ~dim:0 ~start ~length:batch_size;
    timestamps = Array.sub data.timestamps ~pos:start ~len:batch_size;
  }
)

let normalize data =
let normalize_tensor tensor =
  let mean = Tensor.mean tensor ~dim:0 ~keepdim:true in
  let std = Tensor.std tensor ~dim:0 ~keepdim:true in
  let normalized = Tensor.div (Tensor.sub tensor mean) std in
  (normalized, (mean, std))
in
let (norm_x_ts, (mean_x_ts, std_x_ts)) = normalize_tensor data.x_ts in
let (norm_x_static, (mean_x_static, std_x_static)) = normalize_tensor data.x_static in
let (norm_y, (mean_y, std_y)) = normalize_tensor data.y in
let normalized_data = {
  x_ts = norm_x_ts;
  x_static = norm_x_static;
  y = norm_y;
  timestamps = data.timestamps;
} in
let stats = (
  Tensor.cat [mean_x_ts; mean_x_static; mean_y] ~dim:0,
  Tensor.cat [std_x_ts; std_x_static; std_y] ~dim:0
) in
(normalized_data, stats)

let denormalize data (mean, std) =
let denormalize_tensor tensor mean std =
  Tensor.add (Tensor.mul tensor std) mean
in
let x_ts_shape = Tensor.shape data.x_ts in
let x_static_shape = Tensor.shape data.x_static in
let y_shape = Tensor.shape data.y in

let mean_x_ts = Tensor.narrow mean ~dim:0 ~start:0 ~length:(List.nth x_ts_shape 2) in
let mean_x_static = Tensor.narrow mean ~dim:0 ~start:(List.nth x_ts_shape 2) ~length:(List.hd x_static_shape) in
let mean_y = Tensor.narrow mean ~dim:0 ~start:((List.nth x_ts_shape 2) + (List.hd x_static_shape)) ~length:(List.hd y_shape) in

let std_x_ts = Tensor.narrow std ~dim:0 ~start:0 ~length:(List.nth x_ts_shape 2) in
let std_x_static = Tensor.narrow std ~dim:0 ~start:(List.nth x_ts_shape 2) ~length:(List.hd x_static_shape) in
let std_y = Tensor.narrow std ~dim:0 ~start:((List.nth x_ts_shape 2) + (List.hd x_static_shape)) ~length:(List.hd y_shape) in

{
  x_ts = denormalize_tensor data.x_ts mean_x_ts std_x_ts;
  x_static = denormalize_tensor data.x_static mean_x_static std_x_static;
  y = denormalize_tensor data.y mean_y std_y;
  timestamps = data.timestamps;
}

let get_batch data batch_index batch_size =
let start = batch_index * batch_size in
{
  x_ts = Tensor.narrow data.x_ts ~dim:0 ~start ~length:batch_size;
  x_static = Tensor.narrow data.x_static ~dim:0 ~start ~length:batch_size;
  y = Tensor.narrow data.y ~dim:0 ~start ~length:batch_size;
  timestamps = Array.sub data.timestamps ~pos:start ~len:batch_size;
}

let shuffle data =
let n = Tensor.shape data.x_ts |> List.hd in
let indices = Tensor.randperm n ~dtype:(T Int64) in
{
  x_ts = Tensor.index_select data.x_ts ~dim:0 ~index:indices;
  x_static = Tensor.index_select data.x_static ~dim:0 ~index:indices;
  y = Tensor.index_select data.y ~dim:0 ~index:indices;
  timestamps = Array.of_list (List.permute (Array.to_list data.timestamps) ~random_state:(Random.State.make [|0|]));
}

let describe data =
let describe_tensor name tensor =
  let mean = Tensor.mean tensor ~dim:[0] in
  let std = Tensor.std tensor ~dim:[0] in
  let min = Tensor.min tensor ~dim:0 |> fst in
  let max = Tensor.max tensor ~dim:0 |> fst in
  Printf.printf "%s:\n" name;
  Printf.printf "  Shape: %s\n" (Tensor.shape_str tensor);
  Printf.printf "  Mean: %s\n" (Tensor.to_string mean ~line_size:1000);
  Printf.printf "  Std: %s\n" (Tensor.to_string std ~line_size:1000);
  Printf.printf "  Min: %s\n" (Tensor.to_string min ~line_size:1000);
  Printf.printf "  Max: %s\n" (Tensor.to_string max ~line_size:1000);
  Printf.printf "\n"
in
describe_tensor "x_ts" data.x_ts;
describe_tensor "x_static" data.x_static;
describe_tensor "y" data.y;
Printf.printf "Timestamps:\n";
Printf.printf "  Min: %f\n" (Array.min_elt data.timestamps ~compare:Float.compare |> Option.value_exn);
Printf.printf "  Max: %f\n" (Array.max_elt data.timestamps ~compare:Float.compare |> Option.value_exn)

let validate_data data =
let validate_tensor name tensor =
  if Tensor.any (Tensor.isnan tensor) then
    raise (Data_error (Printf.sprintf "NaN values found in %s" name));
  if Tensor.any (Tensor.isinf tensor) then
    raise (Data_error (Printf.sprintf "Inf values found in %s" name))
in
validate_tensor "x_ts" data.x_ts;
validate_tensor "x_static" data.x_static;
validate_tensor "y" data.y
open Torch
open Types
open Logger
open Lwt.Infix
open Mst

exception InvalidData of string

let log_return prices =
  let open Tensor in
  let log_prices = log prices in
  sub (slice_along_axis 1 None None (-1) log_prices) (slice_along_axis 0 None None (-1) log_prices)

let standardize prices =
  let open Tensor in
  let mean = mean prices in
  let std = std prices ~dim:[0] ~unbiased:true ~keepdim:true in
  div (sub prices mean) std

let prepare_data market_index =
  if Array.length market_index.prices < 2 then
    raise (InvalidData "Market index must have at least 2 price points");
  let prices = Tensor.of_float1 market_index.prices in
  let standardized_prices = standardize prices in
  let returns = log_return standardized_prices in
  returns

let calculate_dcca_matrix data timescale =
  let n = Array.length data in
  let matrix = Array.make_matrix n n 0.0 in
  Array.iteri (fun i x ->
    Array.iteri (fun j y ->
      if i < j then
        let coeff = Dcca.dcca_coefficient x y timescale in
        let distance = Dcca.dcca_distance coeff in
        matrix.(i).(j) <- distance;
        matrix.(j).(i) <- distance;
    ) data
  ) data;
  matrix

let calculate_dccc short_data long_data short_scale long_scale timestamp =
  let short_mst = Mst.prim_mst (calculate_dcca_matrix short_data short_scale) in
  let long_mst = Mst.prim_mst (calculate_dcca_matrix long_data long_scale) in
  
  let short_length = Mst.normalized_tree_length short_mst in
  let long_length = Mst.normalized_tree_length long_mst in
  
  let dccc = short_length /. long_length in
  { short_scale; long_scale; dccc; timestamp }

let analyze_window market_indices start_idx window_size config =
  let window_data = Array.map (fun mi ->
    let window_prices = Array.sub mi.prices start_idx window_size in
    prepare_data { name = mi.name; prices = window_prices }
  ) market_indices in
  
  let short_dcca_matrix = calculate_dcca_matrix window_data config.short_timescale in
  let long_dcca_matrix = calculate_dcca_matrix window_data config.long_timescale in
  
  let short_mst = Mst.prim_mst short_dcca_matrix in
  let long_mst = Mst.prim_mst long_dcca_matrix in
  
  let timestamp = float_of_int (start_idx + window_size / 2) in
  let dccc_result = calculate_dccc window_data window_data config.short_timescale config.long_timescale timestamp in
  
  (dccc_result, short_mst, long_mst)

let rolling_window_analysis market_indices config =
  let n = Array.length market_indices.(0).prices in
  let num_windows = (n - config.window_size) / config.step + 1 in
  
  info (Printf.sprintf "Starting rolling window analysis with %d windows" num_windows);
  
  let process_window i =
    let start_idx = i * config.step in
    debug (Printf.sprintf "Processing window %d/%d" (i + 1) num_windows);
    Lwt_preemptive.detach (fun () -> analyze_window market_indices start_idx config.window_size config) ()
  in
  
  Lwt_main.run (
    Lwt_list.map_p process_window (List.init num_windows (fun i -> i))
    >|= Array.of_list
    >|= Array.split3
    >|= fun (dccc_results, short_msts, long_msts) ->
      let timestamps = Array.map (fun r -> r.timestamp) dccc_results in
      info "Rolling window analysis completed";
      { dccc_results; short_msts; long_msts; timestamps }
  )
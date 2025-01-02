open Torch

let compute_optimal_scales features max_scale =
  let n = Tensor.size features 0 in
  let affinity = Diffusion_geometry.fast_affinity_matrix features 1.0 in
  let normalized = Diffusion_geometry.stable_normalize_affinity affinity in
  let operator = Diffusion_geometry.build_diffusion_operator normalized in
  
  let eigenvals, _ = Tensor.symeig operator ~eigenvectors:false in
  let sorted_vals = Tensor.sort eigenvals ~descending:true |> fst in
  
  (* Compute decay rates between consecutive eigenvalues *)
  let decay_rates = Array.init (n-1) (fun i ->
    let val1 = Tensor.get sorted_vals [|i|] in
    let val2 = Tensor.get sorted_vals [|i+1|] in
    Float.abs (val1 -. val2) /. Float.abs val1
  ) in
  
  (* Find significant jumps in decay rates *)
  let mean_rate = Array.fold_left (+.) 0. decay_rates /. float_of_int (n-1) in
  let std_rate = 
    let sq_diff_sum = Array.fold_left (fun acc r -> 
      acc +. (r -. mean_rate) *. (r -. mean_rate)) 0. decay_rates in
    Float.sqrt (sq_diff_sum /. float_of_int (n-1))
  in
  
  (* Select scales where decay rate exceeds mean + std *)
  let threshold = mean_rate +. std_rate in
  let scales = Array.to_list (Array.init (n-1) (fun i -> i)) 
    |> List.filter (fun i -> decay_rates.(i) > threshold)
    |> List.map float_of_int in
  
  (* Ensure we don't exceed max_scale *)
  List.filter (fun s -> s <= float_of_int max_scale) scales
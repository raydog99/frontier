open Types
open Torch

type grid_point = {
  lambda: float;
  gamma: float;
  score: float;
  gradient: float * float;
  region_score: float;
}

let compute_theoretical_bounds s =
  let p = (Tensor.shape s).(0) in
  let s_diag = Tensor.diagonal s ~dim1:0 ~dim2:1 in
  
  (* Compute γ_MAX(0) *)
  let max_gamma = ref 0.0 in
  for i = 0 to p-2 do
    for j = i+1 to p-1 do
      let s_ij = Tensor.get s i j |> Tensor.item in
      let s_ii = Tensor.get s_diag i |> Tensor.item in
      let s_jj = Tensor.get s_diag j |> Tensor.item in
      let curr_val = abs_float s_ij /. (s_ii *. s_jj) in
      max_gamma := max !max_gamma curr_val
    done
  done;
  
  (* Compute λ_MAX(γ) for different γ values *)
  let lambda_bounds = Array.init 50 (fun idx ->
    let gamma = !max_gamma *. float_of_int idx /. 49.0 in
    let max_lambda = ref 0.0 in
    for i = 0 to p-2 do
      for j = i+1 to p-1 do
        let s_ij = Tensor.get s i j |> Tensor.item in
        let s_ii = Tensor.get s_diag i |> Tensor.item in
        let s_jj = Tensor.get s_diag j |> Tensor.item in
        let g_ij = abs_float s_ij /. (gamma *. sqrt (s_ii *. s_jj)) in
        if g_ij > 0.0 then begin
          let term = sqrt (0.25 *. (s_ii +. s_jj) +. g_ij) -. 
                    0.5 *. (s_ii +. s_jj) in
          max_lambda := max !max_lambda term
        end
      done
    done;
    (gamma, !max_lambda)
  ) in
  
  (!max_gamma, lambda_bounds)

let create_adaptive_grid s initial_points =
  let (gamma_max, lambda_bounds) = compute_theoretical_bounds s in
  
  (* Create initial grid points *)
  let points = ref [] in
  Array.iteri (fun i (gamma, lambda_max) ->
    let n_lambda = max 1 (initial_points * (50 - i) / 50) in
    for j = 0 to n_lambda - 1 do
      let lambda = lambda_max *. float_of_int j /. float_of_int (n_lambda - 1) in
      points := {
        lambda = lambda;
        gamma = gamma;
        score = neg_infinity;
        gradient = (0.0, 0.0);
        region_score = 0.0;
      } :: !points
    done
  ) lambda_bounds;
  !points

let cross_validate y params k =
  let n = (Tensor.shape y).(0) in
  let fold_size = n / k in
  let scores = ref [] in
  
  for i = 0 to k-1 do
    let test_start = i * fold_size in
    let test_end = if i = k-1 then n else (i+1) * fold_size in
    
    let test_indices = List.init (test_end - test_start) 
                        (fun j -> test_start + j) in
    let train_indices = List.filter (fun x -> 
      not (List.mem x test_indices)) (List.init n (fun x -> x)) in
    
    let train = Tensor.index_select y ~dim:0 
                 ~index:(Tensor.of_int1 train_indices) in
    let test = Tensor.index_select y ~dim:0 
                 ~index:(Tensor.of_int1 test_indices) in
    
    match Estimation.fit params train with
    | Error _ -> ()
    | Ok (sigma, _) ->
        let score = Estimation.compute_objective sigma test params in
        scores := score :: !scores
  done;
  
  match !scores with
  | [] -> None
  | scores ->
      let mean_score = Statistics.mean scores in
      let std_score = Statistics.std scores in
      Some (mean_score, std_score)

let select_parameters y k =
  let s = Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
          Tensor.div_scalar (float_of_int (Tensor.shape y).(0)) in
  
  let grid = create_adaptive_grid s 10 in
  
  (* Evaluate grid points *)
  let evaluated_points = List.filter_map (fun point ->
    let params = {
      lambda = point.lambda;
      gamma = point.gamma;
      graph = None;
    } in
    match cross_validate y params k with
    | None -> None
    | Some (score, _) ->
        Some { point with score = score }
  ) grid in
  
  (* Select best parameters *)
  match evaluated_points with
  | [] -> { lambda = 0.0; gamma = 0.0; graph = None }
  | points ->
      let best = List.fold_left (fun acc p ->
        if p.score > acc.score then p else acc
      ) (List.hd points) points in
      { lambda = best.lambda; gamma = best.gamma; graph = None }
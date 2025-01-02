open Torch

let safe_div a b =
  let epsilon = 1e-10 in
  Tensor.div a (Tensor.add b (Tensor.full_like b epsilon))

let clip_tensor t min_val max_val =
  Tensor.clamp t ~min:(Tensor.full_like t min_val) ~max:(Tensor.full_like t max_val)

let sample_with_replacement lst n =
  List.init n (fun _ -> List.nth lst (Random.int (List.length lst)))

let transpose lst =
  match lst with
  | [] -> []
  | hd::_ -> List.mapi (fun i _ -> List.map (fun l -> List.nth l i) lst) hd

let linear_regression x y =
  let n = Tensor.shape x |> Tensor.to_int1_exn |> Array.to_list |> List.hd in
  let sum_x = Tensor.sum x in
  let sum_y = Tensor.sum y in
  let sum_xy = Tensor.sum (Tensor.mul x y) in
  let sum_xx = Tensor.sum (Tensor.mul x x) in
  
  let slope = (Tensor.of_float0 (float_of_int n) *. sum_xy -. sum_x *. sum_y) /. 
              (Tensor.of_float0 (float_of_int n) *. sum_xx -. sum_x *. sum_x) in
  let intercept = (sum_y -. slope *. sum_x) /. Tensor.of_float0 (float_of_int n) in
  
  (slope, intercept)

let bayesian_optimization objective hyperparameters =
  let n_iterations = 100 in
  let n_initial = 5 in
  
  let sample_hyperparameters () =
    List.map (fun (name, min_val, max_val) ->
      (name, min_val +. Random.float (max_val -. min_val))
    ) hyperparameters
  in
  
  let initial_points = List.init n_initial (fun _ -> sample_hyperparameters ()) in
  let initial_values = List.map objective initial_points in
  
  let rec optimize points values iteration =
    if iteration >= n_iterations then
      let best_idx = List.mapi (fun i v -> (i, v)) values |> List.fold_left (fun (bi, bv) (i, v) -> if v < bv then (i, v) else (bi, bv)) (-1, Float.infinity) |> fst in
      (List.nth points best_idx, List.nth values best_idx)
    else
      let gp = GaussianProcess.create points values in
      let next_point = GaussianProcess.next_sample gp hyperparameters in
      let next_value = objective next_point in
      optimize (next_point :: points) (next_value :: values) (iteration + 1)
  in
  
  optimize initial_points initial_values 0

let train_process process data =
  let params = Process.estimate_parameters process data in
  Process.update_process_parameters process params

let compute_validation_error process data =
  let predictions = List.map (fun (t, _) ->
    let intensity = Process.compute_intensity process t in
    Tensor.to_float0_exn intensity
  ) data in
  let actual = List.map snd data in
  let errors = List.map2 (fun p a -> (p -. a) ** 2.) predictions actual in
  List.fold_left (+.) 0. errors /. float_of_int (List.length errors)

let update_process_parameters process params =
  { process with
    intensity = (fun v t aux ->
      let w1 = Tensor.get params 0 in
      let w2 = Tensor.get params 1 in
      let b = Tensor.get params 2 in
      Tensor.add (Tensor.add (Tensor.mul w1 v) (Tensor.mul w2 (Tensor.of_float0 t))) b
    )
  }

let compute_log_likelihood process data =
  List.fold_left (fun acc (t, v) ->
    let intensity = Process.compute_intensity process t in
    acc +. log (Tensor.to_float0_exn intensity)
  ) 0. data

let bayesian_model_selection processes data =
  let compute_marginal_likelihood process =
    let params = Process.estimate_parameters process data in
    let updated_process = update_process_parameters process params in
    let log_likelihood = compute_log_likelihood updated_process data in
    let num_params = Tensor.shape params |> Tensor.to_int1_exn |> Array.to_list |> List.hd in
    let n = List.length data in
    log_likelihood -. 0.5 *. float_of_int num_params *. log (float_of_int n)  (* BIC approximation *)
  in
  
  let marginal_likelihoods = List.map compute_marginal_likelihood processes in
  let total_likelihood = List.fold_left (+.) 0. marginal_likelihoods in
  
  List.map2 (fun process ml ->
    let posterior = exp (ml -. total_likelihood) in
    (process, posterior)
  ) processes marginal_likelihoods

module GaussianProcess = struct
  type t = {
    points : float list list;
    values : float list;
    kernel : float list -> float list -> float;
  }

  let create points values =
    let kernel x y =
      let squared_distance = List.map2 (fun xi yi -> (xi -. yi) ** 2.) x y |> List.fold_left (+.) 0. in
      exp (-0.5 *. squared_distance)
    in
    { points; values; kernel }

  let predict gp x =
    let k = List.map (fun xi -> gp.kernel x xi) gp.points in
    let K = List.map (fun xi -> List.map (fun xj -> gp.kernel xi xj) gp.points) gp.points in
    let K_inv = Matrix.inverse K in
    let mean = List.fold_left2 (fun acc ki yi -> acc +. ki *. yi) 0. k gp.values in
    let variance = 1. -. List.fold_left2 (fun acc ki kj -> acc +. ki *. kj) 0. k (Matrix.multiply K_inv k) in
    (mean, variance)

  let next_sample gp hyperparameters =
    let rec find_max_ei current_max current_point = function
      | [] -> current_point
      | point :: rest ->
          let (mean, variance) = predict gp point in
          let std_dev = sqrt variance in
          let z = (mean -. List.hd gp.values) /. std_dev in
          let ei = (mean -. List.hd gp.values) *. (Normal.cdf z) +. std_dev *. (Normal.pdf z) in
          if ei > current_max then
            find_max_ei ei point rest
          else
            find_max_ei current_max current_point rest
    in
    let candidate_points = List.init 1000 (fun _ ->
      List.map (fun (_, min_val, max_val) -> min_val +. Random.float (max_val -. min_val)) hyperparameters
    ) in
    find_max_ei Float.neg_infinity [] candidate_points
end

module Normal = struct
  let pdf x =
    1. /. sqrt (2. *. Float.pi) *. exp (-0.5 *. x *. x)

  let cdf x =
    (1. +. erf (x /. sqrt 2.)) /. 2.
end
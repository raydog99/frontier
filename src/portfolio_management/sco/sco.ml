open Torch

type sample_function = {
  f: Tensor.t -> float * Tensor.t;
  is_clean: bool;
}

type optimization_params = {
  epsilon: float;
  sigma: float;
  beta_bar: float option;
  dimension: int;
  diameter: float;
  tau: float;
}

type error_bounds = {
  corruption_error: float;
  statistical_error: float;
}

type optimization_result = {
  solution: Tensor.t;
  value: float;
  bounds: error_bounds;
  iterations: int;
}

let project_onto_domain w diameter =
  let norm = Tensor.norm w ~p:2 ~dim:[0] |> Tensor.to_float0_exn in
  if norm <= diameter then w
  else Tensor.mul_scalar w (diameter /. norm)

let l2_distance x y =
  Tensor.norm (Tensor.sub x y) ~p:2 ~dim:[0] |> Tensor.to_float0_exn

let compute_covariance grads mean params =
  let centered = List.map (fun g -> Tensor.sub g mean) grads in
  let cov = List.fold_left (fun acc g ->
    let g_expanded = Tensor.unsqueeze g ~dim:1 in
    let g_t = Tensor.transpose g_expanded ~dim0:0 ~dim1:1 in
    Tensor.add acc (Tensor.matmul g_expanded g_t)
  ) (Tensor.zeros [params.dimension; params.dimension]) centered in
  Tensor.div_scalar cov (float_of_int (List.length grads))

let verify_covariance_bound cov sigma =
  let eigenvals = Tensor.linalg_eigvals cov in
  Tensor.to_float1 eigenvals
  |> Array.for_all (fun x -> x <= sigma *. sigma)

let estimate_gradient gradients params =
  let n = List.length gradients in
  let initial_mean = List.fold_left Tensor.add 
    (Tensor.zeros_like (List.hd gradients)) gradients
    |> fun x -> Tensor.div_scalar x (float_of_int n) in

  let initial_cov = compute_covariance gradients initial_mean params in

  let rec filter_iterations mean cov iterations =
    if iterations = 0 then mean
    else
      let weights = List.map (fun g ->
        let diff = Tensor.sub g mean in
        let w = Tensor.(matmul (matmul 
          (unsqueeze diff ~dim:0)
          (inverse cov))
          (unsqueeze diff ~dim:1))
          |> Tensor.to_float0_exn in
        if w > params.sigma *. params.sigma then 0.
        else 1.
      ) gradients in

      let new_mean = List.fold_left2 (fun acc g w ->
        Tensor.add acc (Tensor.mul_scalar g w)
      ) (Tensor.zeros_like mean) gradients weights
      |> fun x -> Tensor.div_scalar x 
          (List.fold_left (+.) 0. weights) in

      let new_cov = compute_covariance 
        (List.filter_map (fun (g, w) -> 
          if w > 0. then Some g else None)
        (List.combine gradients weights))
        new_mean params in

      filter_iterations new_mean new_cov (iterations - 1) in

  filter_iterations initial_mean initial_cov 10

let generate params =
  let spacing = match params.beta_bar with
    | Some beta -> params.sigma *. sqrt params.epsilon /. beta
    | None -> params.sigma *. sqrt params.epsilon /. params.diameter in

  let rec generate_points dims point acc =
    if dims = 0 then point :: acc
    else
      let steps = int_of_float (params.diameter /. spacing) in
      List.fold_left (fun acc' i ->
        let pos = spacing *. float_of_int i -. params.diameter /. 2. in
        let new_point = 
          Tensor.cat [point; Tensor.of_float1 [|pos|]] ~dim:0 in
        generate_points (dims - 1) new_point acc'
      ) acc (List.init steps succ) in

  generate_points params.dimension (Tensor.zeros [0]) []

let optimize ~params ~functions =
  let n = List.length functions in
  let cover_points = generate params in
  
  let w = ref (Tensor.zeros [params.dimension]) in
  let avg_w = ref (Tensor.zeros [params.dimension]) in
  let iterations = ref 0 in
  
  let rec optimize_loop iteration best_value =
    if iteration >= 1000 then
      (best_value, iteration)
    else begin
      let nearest = List.fold_left (fun (best_p, best_d) p ->
        let d = l2_distance !w p in
        if d < best_d then (p, d) else (best_p, best_d)
      ) (List.hd cover_points, Float.infinity) cover_points |> fst in

      let gradients = List.map (fun f -> snd (f.f nearest)) functions in
      let g_t = estimate_gradient gradients params in
      
      let step = 1. /. (1. +. float_of_int iteration) in
      w := project_onto_domain 
        (Tensor.sub !w (Tensor.mul_scalar g_t step))
        params.diameter;

      avg_w := Tensor.add !avg_w !w;

      let current_value = 
        List.fold_left (fun acc f -> acc +. fst (f.f !w)) 
          0. functions /. float_of_int n in
      
      optimize_loop (iteration + 1) (min best_value current_value)
    end in
  
  let final_value, total_iterations = optimize_loop 0 Float.infinity in
  iterations := total_iterations;

  { solution = Tensor.div !avg_w (float_of_int !iterations);
    value = final_value;
    bounds = {
      corruption_error = params.sigma *. params.diameter *. sqrt params.epsilon;
      statistical_error = params.sigma *. params.diameter *. 
        sqrt (float_of_int params.dimension /. float_of_int n)
    };
    iterations = !iterations }
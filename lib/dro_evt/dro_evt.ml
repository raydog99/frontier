open Torch
open Gnuplot

let unit_frechet_cdf x =
	Tensor.exp (Tensor.neg (Tensor.reciprocal x))

let sample_unit_frechet ~n =
	let u = Tensor.rand [n] in
	Tensor.neg (Tensor.log (Tensor.neg (Tensor.log u)))

let wasserstein_distance x y =
    let diff = Tensor.sub x y in
    Tensor.mean (Tensor.abs diff)

let compute_spectral_measure samples =
    let norms = Tensor.sum samples ~dim:[1] ~keepdim:true in
    let normalized = Tensor.div samples norms in
    normalized

let simulate_poisson_process ~intensity ~n_points =
    let times = Tensor.cumsum (Tensor.rand [n_points] ~dtype:(T Float)) ~dim:0 in
    let scaled_times = Tensor.div times intensity in
    scaled_times

let compute_v_x x y =
    let max_ratio = Tensor.maximum (Tensor.div y x) in
    Tensor.reciprocal max_ratio

let optimize_scalar ~f ~lower ~upper ~n_steps =
    let step_size = (upper -. lower) /. float_of_int n_steps in
    let rec loop best_x best_value x =
      if x > upper then (best_x, best_value)
      else
        let value = f x in
        if value < best_value then
          loop x value (x +. step_size)
        else
          loop best_x best_value (x +. step_size)
    in
    loop lower Float.infinity lower

let gradient_descent ~f ~grad ~init ~lr ~n_steps =
    let rec loop x step =
      if step >= n_steps then x
      else
        let g = grad x in
        let x' = Tensor.(sub x (mul_scalar g lr)) in
        loop x' (step + 1)
    in
    loop init 0

let tensor_to_float_list tensor =
    Tensor.to_float1_exn tensor |> Array.to_list

let float_list_to_tensor list =
    Tensor.of_float1 (Array.of_list list)

let robustify_cdf ~baseline_dist ~x ~epsilon =
    let n, _ = Tensor.shape2_exn baseline_dist in
    
    let optimize_lambda lambda =
      let v_x = compute_v_x x baseline_dist in
      let a = simulate_poisson_process ~intensity:1.0 ~n_points:n in
      let indicator = Tensor.lt a v_x in
      let expectation = Tensor.mean indicator in
      let obj = lambda +. (Tensor.to_float0_exn expectation -. 1.0 +. epsilon) *. lambda in
      obj
    in

    let optimal_lambda, _ = optimize_scalar ~f:optimize_lambda ~lower:0. ~upper:10. ~n_steps:1000 in
    1.0 -. optimal_lambda

let robustify_rare_set_prob ~baseline_dist ~set_a ~epsilon =
    let n, _ = Tensor.shape2_exn baseline_dist in
    
    let optimize_lambda lambda =
      let distances = Tensor.cdist baseline_dist set_a in
      let min_distances = Tensor.min distances ~dim:[1] |> fst in
      let indicators = Tensor.lt min_distances (Tensor.full [n] lambda) in
      let expectation = Tensor.mean indicators in
      let obj = lambda +. (Tensor.to_float0_exn expectation -. epsilon) *. lambda in
      obj
    in

    let optimal_lambda, _ = optimize_scalar ~f:optimize_lambda ~lower:0. ~upper:10. ~n_steps:1000 in
    Tensor.to_float0_exn (optimize_lambda optimal_lambda)

let robustify_cvar ~baseline_dist ~alpha ~epsilon =
    let n, _ = Tensor.shape2_exn baseline_dist in
    
    let compute_q_alpha alpha =
      let sorted_dist = Tensor.sort baseline_dist ~dim:0 ~descending:true in
      let index = int_of_float (float_of_int n *. (1. -. alpha)) in
      Tensor.select sorted_dist ~dim:0 ~index
    in

    let q_alpha = compute_q_alpha alpha in
    
    let optimize_lambda lambda =
      let norms = Tensor.norm baseline_dist ~dim:[1] ~p:1 in
      let indicators = Tensor.gt norms q_alpha in
      let excess = Tensor.masked_select norms indicators in
      let expectation = Tensor.mean excess in
      lambda /. (1. -. alpha) +. Tensor.to_float0_exn expectation *. 
        (Tensor.to_float0_exn (Tensor.mean indicators) -. (1. -. alpha) +. epsilon)
    in

    let optimal_lambda, optimal_value = optimize_scalar ~f:optimize_lambda ~lower:0. ~upper:10. ~n_steps:1000 in
    optimal_value

let primal_problem ~loss_fn ~baseline_dist ~epsilon =
    let objective p =
      let loss = loss_fn p in
      let dist = wasserstein_distance p baseline_dist in
      Tensor.(add loss (mul_scalar dist epsilon))
    in

    let grad_objective p =
      Tensor.grad objective p
    in

    let init = baseline_dist in
    gradient_descent ~f:objective ~grad:grad_objective ~init ~lr:0.01 ~n_steps:1000

let dual_problem ~loss_fn ~baseline_dist ~epsilon =
    let objective lambda =
      let inner_max x =
        let loss = loss_fn x in
        let dist = wasserstein_distance x baseline_dist in
        Tensor.(sub loss (mul_scalar dist lambda))
      in
      let max_value = Tensor.maximum (inner_max baseline_dist) in
      Tensor.(add (mul_scalar lambda epsilon) max_value)
    in

    let grad_objective lambda =
      Tensor.grad objective lambda
    in

    let init = Tensor.zeros [] in
    gradient_descent ~f:objective ~grad:grad_objective ~init ~lr:0.01 ~n_steps:1000

let generate_synthetic_data ~n ~d =
    let spectral = compute_spectral_measure (Tensor.rand [n; d]) in
    let radial = sample_unit_frechet ~n in
    Tensor.(mul spectral (expand_dims radial ~dim:1))
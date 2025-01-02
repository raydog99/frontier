open Torch

module Distributions = struct
  let normal mean std =
    let z = Tensor.randn [1] in
    Tensor.(mean + (std * z))
end

module Payoffs = struct
  let forward_start_call x1 x2 =
    Tensor.max (Tensor.sub x2 x1) (Tensor.float 0.)
end

module Wasserstein = struct
  let distance p x y =
    Tensor.(pow_scalar (abs (sub x y)) p |> mean |> pow_scalar (float (1. /. float_of_int p)))

  let adapted_distance p x y =
    let d1 = distance p (Tensor.select x 0) (Tensor.select y 0) in
    let d2 = distance p (Tensor.select x 1) (Tensor.select y 1) in
    Tensor.(pow (d1 ** float p + d2 ** float p) (float (1. /. float_of_int p)))

  let ball mu r =
    fun mu' -> Tensor.(distance 2 mu mu' <= r)

  let adapted_ball mu r =
    fun mu' -> Tensor.(adapted_distance 2 mu mu' <= r)
end

module Martingale = struct
  let constraint x1 x2 =
    Tensor.(mean x1 = mean x2)

  let project x1 x2 =
    let diff = Tensor.(mean x2 - mean x1) in
    (x1, Tensor.(x2 - diff))
end

module Marginal = struct
  let constraint mu1 x1 =
    Tensor.(mean (abs (sub mu1 x1)) <= float 1e-6)

  let project mu1 x1 x2 =
    let diff = Tensor.(mean x1 - mean mu1) in
    (Tensor.(x1 - diff), Tensor.(x2 - diff))
end

module Hedging = struct
  let dynamic h x1 x2 =
    Tensor.(h * (x2 - x1))

  let static f x1 =
    f x1

  let semi_static h f x1 x2 =
    Tensor.((dynamic h x1 x2) + (static f x1))
end

let print_tensor t =
  Tensor.print t;
  Printf.printf "\n"

let run_experiment model_type sigma t =
  let open Models in
  let open Sensitivities in
  let s0 = Tensor.float 1. in
  let mu = match model_type with
    | `BlackScholes -> black_scholes s0 sigma t
    | `Bachelier -> bachelier s0 sigma t
  in
  let sensitivity = sensitivity mu `Martingale in
  let adapted_sensitivity = adapted_sensitivity mu `Martingale in
  let forward_start_sensitivity = forward_start_sensitivity mu in
  let forward_start_adapted_sensitivity = forward_start_adapted_sensitivity mu in
  let martingale_sensitivity_formula = martingale_sensitivity_formula mu in
  let adapted_martingale_sensitivity_formula = adapted_martingale_sensitivity_formula mu in
  
  Printf.printf "Model: %s\n" (match model_type with `BlackScholes -> "Black-Scholes" | `Bachelier -> "Bachelier");
  Printf.printf "Sigma: %f, T: %f\n" (Tensor.to_float0_exn sigma) (Tensor.to_float0_exn t);
  Printf.printf "Sensitivity: "; print_tensor sensitivity;
  Printf.printf "Adapted Sensitivity: "; print_tensor adapted_sensitivity;
  Printf.printf "Forward Start Sensitivity: "; print_tensor forward_start_sensitivity;
  Printf.printf "Forward Start Adapted Sensitivity: "; print_tensor forward_start_adapted_sensitivity;
  Printf.printf "Martingale Sensitivity Formula: "; print_tensor martingale_sensitivity_formula;
  Printf.printf "Adapted Martingale Sensitivity Formula: "; print_tensor adapted_martingale_sensitivity_formula;
  Printf.printf "\n"

let compare_sensitivities model_type sigma_range t_range =
  List.iter (fun sigma ->
    List.iter (fun t ->
      run_experiment model_type (Tensor.float sigma) (Tensor.float t)
    ) t_range
  ) sigma_range

let error_analysis model_type sigma t num_trials =
  let open Models in
  let open Sensitivities in
  let s0 = Tensor.float 1. in
  let mu = match model_type with
    | `BlackScholes -> black_scholes s0 (Tensor.float sigma) (Tensor.float t)
    | `Bachelier -> bachelier s0 (Tensor.float sigma) (Tensor.float t)
  in
  let true_sensitivity = martingale_sensitivity_formula mu in
  let estimated_sensitivities = List.init num_trials (fun _ ->
    sensitivity mu `Martingale
  ) in
  let mean_estimate = Tensor.(mean (of_float1 estimated_sensitivities)) in
  let std_estimate = Tensor.(std (of_float1 estimated_sensitivities) ~dim:[0] ~unbiased:true ~keepdim:false) in
  Printf.printf "Model: %s, Sigma: %f, T: %f\n" 
    (match model_type with `BlackScholes -> "Black-Scholes" | `Bachelier -> "Bachelier")
    sigma t;
  Printf.printf "True Sensitivity: "; print_tensor true_sensitivity;
  Printf.printf "Mean Estimated Sensitivity: "; print_tensor mean_estimate;
  Printf.printf "Std of Estimated Sensitivity: "; print_tensor std_estimate;
  Printf.printf "\n"
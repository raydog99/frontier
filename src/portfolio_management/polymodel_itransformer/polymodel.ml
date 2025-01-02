open Torch
open Error_handling

let calculate_r_squared y_true y_pred =
  let sse = Tensor.(sum (pow (sub y_true y_pred) (Scalar 2.))) in
  let sst = Tensor.(sum (pow (sub y_true (mean y_true)) (Scalar 2.))) in
  Tensor.((sub (Scalar 1.) (div sse sst)))

let calculate_adjusted_r_squared r_squared n p =
  1. -. ((1. -. r_squared) *. (float (n - 1) /. float (n - p - 1)))

let calculate_sharpe_ratio returns risk_free_rate =
  let excess_returns = Tensor.(sub returns risk_free_rate) in
  let mean_excess_return = Tensor.mean excess_returns in
  let std_excess_return = Tensor.std excess_returns ~dim:[0] ~unbiased:true in
  Tensor.(div mean_excess_return std_excess_return)

let calculate_morningstar_risk_adjusted_return returns risk_free_rate gamma n =
  let geometric_excess_return = Tensor.(
    pow (div (add returns (Scalar 1.)) (add risk_free_rate (Scalar 1.))) (Scalar gamma)
  ) in
  let mean_geometric_excess_return = Tensor.(mean geometric_excess_return) in
  Tensor.(pow mean_geometric_excess_return (div (Scalar 1.) (Scalar (float n))) |> sub (Scalar 1.))

let calculate_stress_var returns risk_factors alpha =
  let max_potential_loss = Tensor.(quantile returns (Scalar alpha)) in
  let r_squared = calculate_r_squared returns risk_factors in
  let unexplained_variance = Tensor.((sub (Scalar 1.) r_squared) * (var returns ~unbiased:true)) in
  let xi = 2.33 in (* Corresponds to 99% confidence level *)
  Tensor.(sqrt (add (pow max_potential_loss (Scalar 2.)) 
                    (mul unexplained_variance (Scalar (xi *. xi)))))

let calculate_long_term_alpha returns risk_factors =
  let quantiles = [0.01; 0.16; 0.50; 0.84; 0.99] in
  let factor_quantiles = List.map (fun q -> Tensor.(quantile risk_factors (Scalar q))) quantiles in
  let weights = [0.05; 0.25; 0.40; 0.25; 0.05] in
  let weighted_sum = List.fold_left2 (fun acc w q -> 
    Tensor.(add acc (mul (Scalar w) q))
  ) (Tensor.zeros [1]) weights factor_quantiles in
  weighted_sum

let calculate_long_term_ratio lta svar =
  Tensor.(div lta svar)

let calculate_long_term_stability lta svar lambda =
  Tensor.(sub lta (mul svar (Scalar lambda)))

let polynomial_regression x y degree =
  let x_poly = Tensor.(cat (List.init (degree + 1) (fun i -> pow x (Scalar (float i)))) ~dim:1) in
  let xt_x = Tensor.(matmul (transpose x_poly ~dim0:0 ~dim1:1) x_poly) in
  let xt_y = Tensor.(matmul (transpose x_poly ~dim0:0 ~dim1:1) y) in
  Tensor.(matmul (inverse xt_x) xt_y)

let target_shuffling y x num_shuffles =
  let original_r_squared = calculate_r_squared y (polynomial_regression x y 4) in
  let shuffled_r_squared = List.init num_shuffles (fun _ ->
    let shuffled_y = Tensor.randperm (Tensor.shape y).(0) |> Tensor.index_select y ~dim:0 in
    calculate_r_squared shuffled_y (polynomial_regression x shuffled_y 4)
  ) in
  let count_greater = List.fold_left (fun acc r_sq ->
    if Tensor.(to_float0_exn r_sq > to_float0_exn original_r_squared) then acc + 1 else acc
  ) 0 shuffled_r_squared in
  float count_greater /. float num_shuffles

let extract_features returns risk_factors =
  try
    let sharpe_ratio = calculate_sharpe_ratio returns (Tensor.zeros [1]) in
    let mrar = calculate_morningstar_risk_adjusted_return returns (Tensor.zeros [1]) 2. 36 in
    let svar = calculate_stress_var returns risk_factors 0.98 in
    let lta = calculate_long_term_alpha returns risk_factors in
    let ltr = calculate_long_term_ratio lta svar in
    let lts = calculate_long_term_stability lta svar 0.05 in
    info "Features extracted successfully";
    Tensor.(stack [sharpe_ratio; mrar; svar; lta; ltr; lts] ~dim:1)
  with
  | _ -> raise_error "Failed to extract features"

let extract_features_sequence returns risk_factors sequence_length =
  let num_sequences = (Tensor.shape returns).(0) - sequence_length + 1 in
  let features = Tensor.zeros [num_sequences; sequence_length; 6] in
  for i = 0 to num_sequences - 1 do
    let seq_returns = Tensor.narrow returns ~dim:0 ~start:i ~length:sequence_length in
    let seq_risk_factors = Tensor.narrow risk_factors ~dim:0 ~start:i ~length:sequence_length in
    let seq_features = extract_features seq_returns seq_risk_factors in
    Tensor.copy_ (Tensor.select features ~dim:0 ~index:i) seq_features
  done;
  features
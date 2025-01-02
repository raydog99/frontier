open Torch
open Types
open Kan

type t = {
  kan : Kan.t;
  option_params : option_params;
  option_type : option_type;
  option_style : option_style;
  num_paths : int;
  num_steps : int;
}

let create option_params kan_params option_type option_style num_paths num_steps =
  if option_params.s0 <= 0. || option_params.strike <= 0. || option_params.ttm <= 0. ||
     option_params.volatility <= 0. || option_params.risk_free_rate < 0. || option_params.dividend_rate < 0. then
    raise (Invalid_parameter "Invalid option parameters");
  if num_paths <= 0 || num_steps <= 0 then
    raise (Invalid_parameter "Number of paths and steps must be positive");
  { 
    kan = Kan.create kan_params;
    option_params;
    option_type;
    option_style;
    num_paths;
    num_steps;
  }

let generate_paths t =
  let open Tensor in
  let dt = t.option_params.ttm /. (float t.num_steps) in
  let drift = (t.option_params.risk_free_rate -. t.option_params.dividend_rate -. 0.5 *. t.option_params.volatility ** 2.) *. dt in
  let diffusion = t.option_params.volatility *. sqrt dt in
  let random_walks = randn [t.num_paths; t.num_steps] in
  let log_returns = (random_walks * scalar diffusion) + scalar drift in
  let s0 = full [t.num_paths; 1] t.option_params.s0 in
  exp (cumsum (cat [s0; log_returns] ~dim:1) ~dim:1)

let intrinsic_value t paths step =
  let open Tensor in
  let spot = paths.[[All; step]] in
  match t.option_style, t.option_type with
  | American, Call -> max (spot - scalar t.option_params.strike) (scalar 0.)
  | American, Put -> max (scalar t.option_params.strike - spot) (scalar 0.)
  | AsianAmerican, Call ->
      let avg_price = mean paths.[[All; 0 -- step]] ~dim:1 in
      max (avg_price - scalar t.option_params.strike) (scalar 0.)
  | AsianAmerican, Put ->
      let avg_price = mean paths.[[All; 0 -- step]] ~dim:1 in
      max (scalar t.option_params.strike - avg_price) (scalar 0.)

let discount_factor t dt =
  Tensor.(exp (scalar (-. t.option_params.risk_free_rate *. dt)))

let lsmc t paths =
  let open Tensor in
  let dt = t.option_params.ttm /. (float t.num_steps) in
  let discount = discount_factor t dt in
  let cashflows = ref (intrinsic_value t paths (t.num_steps - 1)) in
  
  for step = t.num_steps - 2 downto 0 do
    let spot = paths.[[All; step]] in
    let input_tensor = match t.option_style with
      | American -> spot.unsqueeze 1
      | AsianAmerican -> 
          let avg_price = mean paths.[[All; 0 -- step]] ~dim:1 in
          stack [spot; avg_price] ~dim:1
    in
    let continuation_value = Kan.forward t.kan input_tensor in
    let exercise_value = intrinsic_value t paths step in
    cashflows := where (exercise_value > continuation_value.squeeze 1)
                   ~then_:exercise_value
                   ~else_:((!cashflows) * discount);
  done;
  
  mean !cashflows

let train t num_epochs learning_rate =
  if num_epochs <= 0 then raise (Invalid_parameter "Number of epochs must be positive");
  if learning_rate <= 0. then raise (Invalid_parameter "Learning rate must be positive");
  let optimizer = Optimizer.adam (Kan.parameters t.kan) ~learning_rate in
  let paths = generate_paths t in

  for _ = 1 to num_epochs do
    Optimizer.zero_grad optimizer;
    let loss = Tensor.neg (lsmc t paths) in
    Tensor.backward loss;
    Optimizer.step optimizer;
  done

let price t =
  let paths = generate_paths t in
  Tensor.to_float0 (lsmc t paths)

let calculate_greek t greek_type =
  let eps = 1e-6 in
  let original_s0 = t.option_params.s0 in
  let up_price = 
    let up_params = { t.option_params with s0 = original_s0 *. (1. +. eps) } in
    let up_t = { t with option_params = up_params } in
    price up_t
  in
  let down_price = 
    let down_params = { t.option_params with s0 = original_s0 *. (1. -. eps) } in
    let down_t = { t with option_params = down_params } in
    price down_t
  in
  match greek_type with
  | `Delta -> (up_price -. down_price) /. (2. *. eps *. original_s0)
  | `Gamma -> 
      let central_price = price t in
      (up_price -. 2. *. central_price +. down_price) /. (eps *. original_s0) ** 2.
  | `Theta ->
      let original_ttm = t.option_params.ttm in
      let down_ttm_params = { t.option_params with ttm = original_ttm -. eps } in
      let down_ttm_t = { t with option_params = down_ttm_params } in
      let down_ttm_price = price down_ttm_t in
      -. (down_ttm_price -. price t) /. eps

let delta t = calculate_greek t `Delta
let gamma t = calculate_greek t `Gamma
let theta t = calculate_greek t `Theta
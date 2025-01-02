open Base
open Torch
open Path_integral_utils
open Advanced_numerical_methods
open Parallel_computation
open Heston_model
open Local_volatility_model
open American_option

type option_type = Call | Put
type barrier_type = UpAndOut | DownAndOut | UpAndIn | DownAndIn

type model =
  | BlackScholes of { r: float; sigma: float }
  | Heston of { r: float; kappa: float; theta: float; xi: float; rho: float }
  | LocalVol of { r: float; sigma: Tensor.t -> float -> float }

type t = {
  model: model;
  num_time_steps: int;
  num_paths: int;
  dt: float;
  initial_state: Tensor.t;
}

let create ~model ~num_time_steps ~num_paths ~maturity ~initial_state =
  let dt = maturity /. Float.of_int num_time_steps in
  { model; num_time_steps; num_paths; dt; initial_state }

let generate_paths t =
  match t.model with
  | BlackScholes { r; sigma } ->
      generate_black_scholes_paths t.initial_state t.num_paths t.num_time_steps t.dt r sigma
  | Heston { r; kappa; theta; xi; rho } ->
      generate_heston_paths t.initial_state t.num_paths t.num_time_steps t.dt r kappa theta xi rho
  | LocalVol { r; sigma } ->
      generate_local_vol_paths t.initial_state t.num_paths t.num_time_steps t.dt r sigma

let price_european_option t ~strike ~option_type =
  let paths = generate_paths t in
  let final_prices = Tensor.select paths (-1) (-1) in
  let payoff = match option_type with
    | Call -> european_call_payoff strike
    | Put -> european_put_payoff strike
  in
  let option_values = payoff final_prices in
  let r = match t.model with
    | BlackScholes { r; _ } | Heston { r; _ } | LocalVol { r; _ } -> r
  in
  let discount_factor = Tensor.exp (Tensor.mul_scalar (Tensor.scalar_float (-r *. t.dt *. Float.of_int t.num_time_steps)) 1.) in
  Tensor.mul discount_factor (Tensor.mean option_values)

let price_american_option t ~strike ~option_type =
  let paths = generate_paths t in
  let payoff = match option_type with
    | Call -> american_call_payoff strike
    | Put -> american_put_payoff strike
  in
  let r = match t.model with
    | BlackScholes { r; _ } | Heston { r; _ } | LocalVol { r; _ } -> r
  in
  let discount_factors = Tensor.exp (Tensor.mul_scalar (Tensor.scalar_float (-r *. t.dt)) (Tensor.arange ~start:1 ~end_:(t.num_time_steps + 1) ~options:(Kind Float, Device Cpu) ())) in
  least_squares_monte_carlo paths payoff discount_factors

let price_barrier_option t ~strike ~option_type ~barrier_type ~barrier_level =
  let paths = generate_paths t in
  let final_prices = Tensor.select paths (-1) (-1) in
  let payoff = match option_type with
    | Call -> european_call_payoff strike
    | Put -> european_put_payoff strike
  in
  let barrier_condition = match barrier_type with
    | UpAndOut -> Tensor.gt paths (Tensor.full_like paths barrier_level)
    | DownAndOut -> Tensor.lt paths (Tensor.full_like paths barrier_level)
    | UpAndIn -> Tensor.le paths (Tensor.full_like paths barrier_level)
    | DownAndIn -> Tensor.ge paths (Tensor.full_like paths barrier_level)
  in
  let barrier_payoff = match barrier_type with
    | UpAndOut | DownAndOut ->
        Tensor.where (Tensor.any barrier_condition ~dim:[-2]) (Tensor.zeros_like final_prices) final_prices
    | UpAndIn | DownAndIn ->
        Tensor.where (Tensor.any barrier_condition ~dim:[-2]) final_prices (Tensor.zeros_like final_prices)
  in
  let option_values = payoff barrier_payoff in
  let r = match t.model with
    | BlackScholes { r; _ } | Heston { r; _ } | LocalVol { r; _ } -> r
  in
  let discount_factor = Tensor.exp (Tensor.mul_scalar (Tensor.scalar_float (-r *. t.dt *. Float.of_int t.num_time_steps)) 1.) in
  Tensor.mul discount_factor (Tensor.mean option_values)

let price_asian_option t ~strike ~option_type ~averaging_points =
  let paths = generate_paths t in
  let average_prices = Tensor.mean (Tensor.narrow paths (-2) (t.num_time_steps - averaging_points) averaging_points) ~dim:[-2] in
  let payoff = match option_type with
    | Call -> asian_call_payoff strike
    | Put -> asian_put_payoff strike
  in
  let option_values = payoff average_prices in
  let r = match t.model with
    | BlackScholes { r; _ } | Heston { r; _ } | LocalVol { r; _ } -> r
  in
  let discount_factor = Tensor.exp (Tensor.mul_scalar (Tensor.scalar_float (-r *. t.dt *. Float.of_int t.num_time_steps)) 1.) in
  Tensor.mul discount_factor (Tensor.mean option_values)

let calculate_greeks t ~strike ~option_type =
  let spot = Tensor.to_float0_exn t.initial_state in
  let maturity = Float.of_int t.num_time_steps *. t.dt in
  let (r, sigma) = match t.model with
    | BlackScholes { r; sigma } -> (r, sigma)
    | Heston { r; kappa; theta; xi; rho } ->
        let v0 = theta in  (* Assuming initial variance is the long-term variance *)
        (r, Float.sqrt v0)  (* Using initial volatility for Black-Scholes approximation *)
    | LocalVol { r; sigma } ->
        (r, sigma t.initial_state 0.)  (* Using initial local volatility *)
  in
  finite_difference_greeks
    ~option_pricing_fn:(fun ~spot ->
      let new_initial_state = Tensor.full_like t.initial_state spot in
      let new_t = { t with initial_state = new_initial_state } in
      price_european_option new_t ~strike ~option_type |> Tensor.to_float0_exn)
    ~spot ~strike ~risk_free_rate:r ~volatility:sigma ~maturity ~option_type

let parallel_price_option t price_fn =
  let chunk_size = t.num_paths / num_workers in
  let partial_t = { t with num_paths = chunk_size } in
  let results = parallel_map (fun _ -> price_fn partial_t) (List.init num_workers ~f:Fn.id) in
  Tensor.mean (Tensor.stack results 0)

let parallel_price_european_option t ~strike ~option_type =
  parallel_price_option t (fun t -> price_european_option t ~strike ~option_type)

let parallel_price_american_option t ~strike ~option_type =
  parallel_price_option t (fun t -> price_american_option t ~strike ~option_type)
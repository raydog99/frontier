open Torch
open Printf

type reference_point = Initial | Strike

type option_type = Call | Put

type t = {
  s0: float;         (** Initial stock price *)
  k: float;          (** Strike price *)
  mu0: float;        (** Initial drift *)
  sigma: float;      (** Volatility *)
  lambda0: float;    (** Minimum frequency of fluctuation *)
  eta: float;        (** Sensitivity of fluctuation *)
  alpha: float;      (** Scaling factor *)
  b: float;          (** Scale parameter *)
  t: float;          (** Time to maturity *)
  r: float;          (** Risk-free rate *)
  num_paths: int;    (** Number of simulation paths *)
  num_steps: int;    (** Number of time steps *)
  reference: reference_point; (** Reference point for calculations *)
}

exception InvalidParameterError of string

let validate_parameters u_t =
  if u_t.s0 <= 0.0 then raise (InvalidParameterError "Initial stock price must be positive");
  if u_t.k <= 0.0 then raise (InvalidParameterError "Strike price must be positive");
  if u_t.sigma <= 0.0 then raise (InvalidParameterError "Volatility must be positive");
  if u_t.lambda0 < 0.0 then raise (InvalidParameterError "Minimum jump frequency must be non-negative");
  if u_t.t <= 0.0 then raise (InvalidParameterError "Time to maturity must be positive");
  if u_t.r < 0.0 then raise (InvalidParameterError "Risk-free rate must be non-negative");
  if u_t.num_paths <= 0 then raise (InvalidParameterError "Number of paths must be positive");
  if u_t.num_steps <= 0 then raise (InvalidParameterError "Number of time steps must be positive")

let create s0 k mu0 sigma lambda0 eta alpha b t r num_paths num_steps reference =
  let u_t = { s0; k; mu0; sigma; lambda0; eta; alpha; b; t; r; num_paths; num_steps; reference } in
  validate_parameters u_t;
  u_t

let jump_rate u_t s_t =
  let reference = match u_t.reference with
    | Initial -> u_t.s0
    | Strike -> u_t.k
  in
  u_t.lambda0 +. u_t.eta *. Float.abs (s_t -. reference)

let location_parameter u_t s_t =
  let reference = match u_t.reference with
    | Initial -> u_t.s0
    | Strike -> u_t.k
  in
  u_t.mu0 +. u_t.alpha *. (s_t -. reference)

let simulate_paths_parallel u_t =
  let dt = u_t.t /. float_of_int u_t.num_steps in
  let paths = Array.make_matrix u_t.num_paths (u_t.num_steps + 1) u_t.s0 in
  for i = 0 to u_t.num_paths - 1 do
    for j = 1 to u_t.num_steps do
      let prev_price = paths.(i).(j-1) in
      let rate = jump_rate u_t prev_price in
      let jump_time = min (float_of_int j *. dt) (float_of_int (j-1) *. dt +. Random.float (1.0 /. rate)) in
      let mu = location_parameter u_t prev_price in
      paths.(i).(j) <- 
        prev_price *. exp((mu -. 0.5 *. u_t.sigma ** 2.0) *. (jump_time -. float_of_int (j-1) *. dt) 
                          +. u_t.sigma *. sqrt(jump_time -. float_of_int (j-1) *. dt) *. Random.float_range (-1.0) 1.0)
    done
  done;
  paths

let payoff option_type s k =
  match option_type with
  | Call -> max (s -. k) 0.0
  | Put -> max (k -. s) 0.0

let longstaff_schwartz_optimized u_t option_type =
  let paths = simulate_paths_parallel u_t in
  let num_paths = Array.length paths in
  let num_steps = Array.length paths.(0) - 1 in
  let payoff = payoff option_type in
  let cashflows = Tensor.zeros [num_paths] in
  
  for i = num_steps - 1 downto 0 do
    let x = Tensor.of_float1 (Array.init num_paths (fun j -> paths.(j).(i))) in
    let y = Tensor.mul_scalar (cashflows) (exp (-. u_t.r *. u_t.t /. float_of_int num_steps)) in
    let x_squared = Tensor.mul x x in
    let x_combined = Tensor.cat [Tensor.ones [num_paths; 1]; Tensor.unsqueeze x (-1); Tensor.unsqueeze x_squared (-1)] 1 in
    let beta = Tensor.matmul (Tensor.pinverse x_combined) (Tensor.unsqueeze y (-1)) in
    
    for j = 0 to num_paths - 1 do
      let continuation_value = 
        Tensor.get beta [0] +. 
        Tensor.get beta [1] *. paths.(j).(i) +. 
        Tensor.get beta [2] *. (paths.(j).(i) ** 2.) in
      let immediate_exercise = payoff paths.(j).(i) u_t.k in
      if immediate_exercise > continuation_value then
        Tensor.set cashflows [j] immediate_exercise
    done
  done;
  
  Tensor.sum cashflows |> Tensor.to_float0 /. float_of_int num_paths *. exp(-. u_t.r *. u_t.t)

let pdifmp_price u_t option_type =
  let paths = simulate_paths_parallel u_t in
  let prices = Array.map (fun path -> 
    Array.fold_left (fun max_payoff price -> 
      max max_payoff (payoff option_type price u_t.k *. exp(-. u_t.r *. u_t.t))
    ) 0.0 path
  ) paths in
  Array.fold_left (+.) 0.0 prices /. float_of_int u_t.num_paths

let ls_pdifmp_price u_t option_type =
  longstaff_schwartz_optimized u_t option_type

let compare_methods u_t option_type =
  let ls_price = longstaff_schwartz_optimized u_t option_type in
  let pdifmp_price = pdifmp_price u_t option_type in
  let ls_pdifmp_price = ls_pdifmp_price u_t option_type in
  (ls_price, pdifmp_price, ls_pdifmp_price)

let run_all_experiments () =
  let base_params = create 40.0 40.0 0.06 0.2 0.5 0.1 0.01 0.01 1.0 0.06 10000 50 Initial in
  let (ls_price, pdifmp_price, ls_pdifmp_price) = compare_methods base_params Call in
  printf "Experiment Results:\n";
  printf "Longstaff-Schwartz price: %.4f\n" ls_price;
  printf "PDifMP price: %.4f\n" pdifmp_price;
  printf "LS+PDifMP price: %.4f\n" ls_pdifmp_price

let run_unit_tests () =
  let test_params = create 100.0 100.0 0.05 0.2 0.5 0.1 0.01 0.01 1.0 0.05 1000 50 Initial in
  let (ls_price, pdifmp_price, ls_pdifmp_price) = compare_methods test_params Call in
  printf "Unit Test Results:\n";
  printf "Longstaff-Schwartz price: %.4f\n" ls_price;
  printf "PDifMP price: %.4f\n" pdifmp_price;
  printf "LS+PDifMP price: %.4f\n" ls_pdifmp_price;
  assert (abs_float (ls_price -. pdifmp_price) < 1.0);
  assert (abs_float (ls_price -. ls_pdifmp_price) < 1.0);
  printf "All tests passed!\n"
open Torch

type option_type = Call | Put

type option = {
  underlying: string;
  option_type: option_type;
  strike: float;
  expiry: float;
  price: float;
}

let black_scholes s k r t sigma option_type =
  let d1 = (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
  let d2 = d1 -. sigma *. sqrt t in
  match option_type with
  | Call -> s *. Torch_normal.cdf d1 -. k *. exp (~-.r *. t) *. Torch_normal.cdf d2
  | Put -> k *. exp (~-.r *. t) *. Torch_normal.cdf (~-.d2) -. s *. Torch_normal.cdf (~-.d1)

let calculate_delta s k r t sigma option_type =
  let d1 = (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
  match option_type with
  | Call -> Torch_normal.cdf d1
  | Put -> Torch_normal.cdf d1 -. 1.

let calculate_gamma s k r t sigma =
  let d1 = (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
  Torch_normal.pdf d1 /. (s *. sigma *. sqrt t)

let calculate_theta s k r t sigma option_type =
  let d1 = (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
  let d2 = d1 -. sigma *. sqrt t in
  match option_type with
  | Call ->
    ~-.(s *. Torch_normal.pdf d1 *. sigma) /. (2. *. sqrt t) -. r *. k *. exp (~-.r *. t) *. Torch_normal.cdf d2
  | Put ->
    ~-.(s *. Torch_normal.pdf d1 *. sigma) /. (2. *. sqrt t) +. r *. k *. exp (~-.r *. t) *. Torch_normal.cdf (~-.d2)

let delta_hedge_strategy stock_price option delta_threshold =
  let delta = calculate_delta stock_price option.strike 0.05 option.expiry 0.2 option.option_type in
  if abs_float delta > delta_threshold then
    int_of_float (delta *. 100.)  (* Hedge with 100 shares per option contract *)
  else
    0

let monte_carlo_option_pricing s k r t sigma option_type num_simulations =
  let dt = t /. 252. in
  let simulations = Tensor.randn [num_simulations; 252] in
  let paths = Tensor.cumprod (Tensor.add (Tensor.mul (Tensor.scalar_tensor (r -. 0.5 *. sigma ** 2.) *. dt) (Tensor.ones_like simulations))
                                         (Tensor.mul (Tensor.scalar_tensor (sigma *. sqrt dt)) simulations)) ~dim:1 in
  let final_prices = Tensor.mul (Tensor.scalar_tensor s) (Tensor.select paths ~dim:1 ~index:251) in
  let payoffs = match option_type with
    | Call -> Tensor.relu (Tensor.sub final_prices (Tensor.scalar_tensor k))
    | Put -> Tensor.relu (Tensor.sub (Tensor.scalar_tensor k) final_prices)
  in
  let option_price = Tensor.mean payoffs |> Tensor.to_float0_exn in
  option_price *. exp (~-.r *. t)
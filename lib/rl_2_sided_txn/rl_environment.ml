open Torch

exception Invalid_action of string
exception Environment_error of string

type t = {
  mutable portfolio_value: float;
  mutable portfolio: Tensor.t;
  mutable current_step: int;
  data: Tensor.t;
  transaction_fee: float;
  interest_rate: float;
}

let create data transaction_fee interest_rate =
  if transaction_fee < 0.0 || transaction_fee > 1.0 then
    raise (Invalid_argument "Transaction fee must be between 0 and 1");
  if interest_rate < 0.0 then
    raise (Invalid_argument "Interest rate must be non-negative");
  {
    portfolio_value = 1.0;
    portfolio = Tensor.zeros [Tensor.size data |> List.hd];
    current_step = 0;
    data;
    transaction_fee;
    interest_rate;
  }

let reset env =
  env.portfolio_value <- 1.0;
  env.portfolio <- Tensor.zeros [Tensor.size env.data |> List.hd];
  env.current_step <- 0

let step env action =
  let open Tensor in
  let m = size env.data |> List.hd in
  
  if size action |> List.hd <> m then
    raise (Invalid_action "Action dimension does not match the number of assets");
  
  let c_t = env.portfolio_value *. (1.0 -. Tensor.get action (m - 1)) in
  let interest = env.portfolio_value *. Tensor.get action (m - 1) *. env.interest_rate in
  let new_portfolio = mul (float_scalar c_t) (slice action ~dim:0 ~start:0 ~end_:(m - 1)) in
  let costs = env.transaction_fee *. (sum (abs (sub new_portfolio env.portfolio)) |> to_float0_exn) in
  
  env.portfolio <- new_portfolio;
  
  let returns = slice env.data ~dim:0 ~start:env.current_step ~end_:(env.current_step + 1) in
  let portfolio_return = sum (mul env.portfolio returns) |> to_float0_exn in
  
  env.portfolio_value <- env.portfolio_value +. portfolio_return +. interest -. costs;
  
  if env.portfolio_value <= 0.0 then
    raise (Environment_error "Portfolio value dropped to zero or below");
  
  let reward = (portfolio_return -. costs) /. c_t in
  
  env.current_step <- env.current_step + 1;
  
  (reward, env.portfolio_value, env.current_step >= (size env.data |> List.nth 2) - 1)

let get_state env =
  Tensor.slice env.data ~dim:2 ~start:env.current_step ~end_:(env.current_step + 1)
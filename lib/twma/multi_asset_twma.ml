open Torch
open Portfolio_constructor
open Twma
open risk_manager
open Ml_model

type t = {
  twmas: Twma.t array;
  portfolio_constructor: Portfolio_constructor.t;
  risk_manager: Risk_manager.t;
  ml_model: Ml_model.t option;
  mutable weights: Tensor.t;
  mutable portfolio_value: float;
  mutable returns: float array array;
}

let create config assets fundamental_data_provider =
  let num_assets = Array.length assets in
  {
    twmas = Array.map (fun _ -> Twma.create config.Config.window_size config.alpha) assets;
    portfolio_constructor = Portfolio_constructor.create config.portfolio_method num_assets;
    risk_manager = Risk_manager.create 
      ~max_drawdown:config.max_drawdown 
      ~var_threshold:config.var_threshold 
      ~volatility_target:config.volatility_target 
      ~max_leverage:config.max_leverage ();
    ml_model = (if config.use_ml then Some (Ml_model.create (num_assets * 2) 50 num_assets 0.001) else None);
    weights = Tensor.ones [num_assets] |> Tensor.div_scalar (float_of_int num_assets);
    portfolio_value = config.initial_cash;
    returns = Array.make num_assets [||];
  }

let update t prices =
  Array.iteri (fun i twma ->
    Twma.update twma prices.(i);
    t.returns.(i) <- Array.append t.returns.(i) [|prices.(i)|]
  ) t.twmas;

  let returns = Array.map Twma.get_ma t.twmas in
  let covariance = Tensor.cov (Tensor.of_float1 returns) in
  
  let raw_weights = Portfolio_constructor.construct_portfolio t.portfolio_constructor (Tensor.of_float1 returns) covariance in
  
  let ml_adjusted_weights = 
    match t.ml_model with
    | Some model ->
        let inputs = Tensor.cat [Tensor.of_float1 returns; raw_weights] ~dim:0 in
        Ml_model.predict model inputs
    | None -> raw_weights
  in

  t.weights <- Tensor.of_float1 (Risk_manager.apply_risk_management t.risk_manager (Tensor.to_float1_exn ml_adjusted_weights) t.returns)

let get_portfolio_value t prices =
  Array.fold_left2 (fun acc w p -> acc +. w *. p) 0. 
    (Tensor.to_float1_exn t.weights) prices

let trade t prices =
  update t prices;
  let new_value = get_portfolio_value t prices in
  let daily_return = (new_value -. t.portfolio_value) /. t.portfolio_value in
  t.portfolio_value <- new_value;
  daily_return

let train_ml_model t =
  match t.ml_model with
  | Some model ->
      let inputs = Tensor.cat [Tensor.of_float2 t.returns; t.weights] ~dim:1 in
      let targets = Tensor.of_float2 (Array.map (fun returns -> [|Array.fold_left ( *. ) 1. returns|]) t.returns) in
      Ml_model.train model inputs targets 100
  | None -> ()
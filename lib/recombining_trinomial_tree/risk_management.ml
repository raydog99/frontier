type market = {
  stock: Tree.t;
  derivative: Perpetual_Derivative.t;
  option: Option.t;
  risk_free_rate: float;
}

let create_market stock derivative option risk_free_rate =
  { stock; derivative; option; risk_free_rate }

let is_complete market =
  abs_float (market.stock.r -. market.risk_free_rate) > 1e-6

let price_option market =
  Option.price market.option

let replicating_portfolio market s t =
  let option_delta = Option.delta market.option s t in
  let derivative_delta = Perpetual_Derivative.delta market.derivative s in
  let option_price = price_option market in
  
  let stock_quantity = (option_delta *. market.derivative.gamma -. derivative_delta) /. 
                       (market.derivative.gamma -. 1.) in
  let derivative_quantity = (option_delta -. stock_quantity) /. derivative_delta in
  let cash = option_price -. stock_quantity *. s -. derivative_quantity *. (Perpetual_Derivative.price market.derivative s) in
  
  (stock_quantity, derivative_quantity, cash)

let hedge_error market s t dt =
  let stock_quantity, derivative_quantity, cash = replicating_portfolio market s t in
  let future_s = s *. exp ((market.stock.r -. 0.5 *. market.stock.sigma ** 2.) *. dt +. 
                           market.stock.sigma *. sqrt dt *. Random.float 1.) in
  let future_derivative_price = Perpetual_Derivative.price market.derivative future_s in
  let future_option_price = Option.price { market.option with tree = { market.option.tree with s0 = future_s; t = t -. dt } } in
  
  future_option_price -. (stock_quantity *. future_s +. 
                          derivative_quantity *. future_derivative_price +. 
                          cash *. exp (market.risk_free_rate *. dt))

let simulate_hedge_performance market s t num_steps num_simulations =
  let dt = t /. float_of_int num_steps in
  List.init num_simulations (fun _ ->
    let rec simulate current_s current_t acc =
      if current_t <= 0. then acc
      else
        let error = hedge_error market current_s current_t dt in
        let new_s = current_s *. exp ((market.stock.r -. 0.5 *. market.stock.sigma ** 2.) *. dt +. 
                                      market.stock.sigma *. sqrt dt *. Random.float 1.) in
        simulate new_s (current_t -. dt) (error :: acc)
    in
    simulate s t []
  )

let value_at_risk market confidence_level num_simulations horizon =
  let simulations = simulate_hedge_performance market market.stock.s0 horizon 1 num_simulations in
  let losses = List.map (fun sim -> -. (List.hd sim)) simulations in
  let sorted_losses = List.sort compare losses in
  List.nth sorted_losses (int_of_float (float_of_int num_simulations *. confidence_level))

let expected_shortfall market confidence_level num_simulations horizon =
  let var = value_at_risk market confidence_level num_simulations horizon in
  let simulations = simulate_hedge_performance market market.stock.s0 horizon 1 num_simulations in
  let losses = List.map (fun sim -> -. (List.hd sim)) simulations in
  let tail_losses = List.filter (fun loss -> loss >= var) losses in
  List.fold_left (+.) 0. tail_losses /. float_of_int (List.length tail_losses)

let incremental_var market positions confidence_level num_simulations horizon =
  let base_var = value_at_risk market confidence_level num_simulations horizon in
  List.map (fun (asset, quantity) ->
    let new_market = { market with stock = { market.stock with s0 = market.stock.s0 +. quantity *. asset } } in
    let new_var = value_at_risk new_market confidence_level num_simulations horizon in
    (asset, quantity, new_var -. base_var)
  ) positions

let component_var market positions confidence_level num_simulations horizon =
  let total_var = value_at_risk market confidence_level num_simulations horizon in
  let incremental_vars = incremental_var market positions confidence_level num_simulations horizon in
  List.map (fun (asset, quantity, inc_var) ->
    (asset, quantity, inc_var *. quantity /. total_var)
  ) incremental_vars

let marginal_var market positions confidence_level num_simulations horizon =
  let incremental_vars = incremental_var market positions confidence_level num_simulations horizon in
  List.map (fun (asset, quantity, inc_var) ->
    (asset, inc_var /. quantity)
  ) incremental_vars

let calc_greeks market =
  let s = market.stock.s0 in
  let t = market.option.tree.t in
  {
    delta = Option.delta market.option s t;
    gamma = Option.gamma market.option s t;
    theta = Option.theta market.option s t;
    vega = Option.vega market.option s t;
    rho = Option.rho market.option s t;
  }

type scenario = {
  delta_s: float;
  delta_sigma: float;
  delta_r: float;
  delta_q: float;
  jump_intensity: float;
  jump_size: float;
}

let apply_scenario market scenario =
  let new_s0 = market.stock.s0 *. (1. +. scenario.delta_s) in
  let new_sigma = market.stock.sigma +. scenario.delta_sigma in
  let new_r = market.stock.r +. scenario.delta_r in
  let new_tree = Tree.create new_s0 new_r new_sigma market.stock.t market.stock.n in
  let new_derivative = Perpetual_Derivative.create new_tree in
  let new_option = { market.option with tree = new_tree } in
  { market with stock = new_tree; derivative = new_derivative; option = new_option }

let stress_test market scenarios =
  List.map (fun scenario ->
    let stressed_market = apply_scenario market scenario in
    let option_price = price_option stressed_market in
    let greeks = calc_greeks stressed_market in
    (scenario, option_price, greeks)
  ) scenarios
open Types
open Torch
open Utils
open Numerical_methods
open Lwt

module MicroModel : MODEL = struct
  let simulate_parallel params market_data num_paths num_steps =
    let num_futures = Array.length market_data.futures_prices in
    let corr_matrix = generate_correlation_matrix params.beta num_futures in
    let chol_corr = Tensor.cholesky corr_matrix in

    let simulate_batch batch_size =
      let paths = Tensor.zeros [batch_size; num_steps; num_futures] in
      let dt = market_data.futures_maturities.(1) -. market_data.futures_maturities.(0) in
      let sqrt_dt = sqrt dt in

      for t = 1 to num_steps - 1 do
        let dW = Tensor.(randn [batch_size; num_futures] * Scalar.float sqrt_dt) in
        let correlated_dW = Tensor.matmul dW chol_corr in

        for i = 0 to num_futures - 1 do
          let prev_prices = Tensor.select paths 1 (t-1) in
          let vol = params.leverage_function t prev_prices.(i) market_data.futures_prices.(i) *. sqrt params.v0 in
          let new_prices = Tensor.(
            prev_prices * exp ((Scalar.float market_data.risk_free_rate - Scalar.float (0.5 *. vol *. vol)) * Scalar.float dt + 
                               Scalar.float vol * select correlated_dW 1 i)
          ) in
          Tensor.select_inplace paths 1 t new_prices
        done
      done;
      paths
    in

    let num_cores = 4 in
    let batch_size = num_paths / num_cores in
    
    let batches = List.init num_cores (fun _ -> Lwt.return (simulate_batch batch_size)) in
    let%lwt results = Lwt.all batches in
    
    Lwt.return (Tensor.cat results ~dim:0)

  let price_vanilla_option params market_data strike maturity option_type =
    let%lwt paths = simulate_parallel params market_data 100000 200 in
    let final_prices = Tensor.select paths 1 199 in
    let payoffs = match option_type with
      | Call -> Tensor.(max (final_prices - Scalar.float strike) (Scalar.float 0.))
      | Put -> Tensor.(max (Scalar.float strike - final_prices) (Scalar.float 0.))
    in
    let discount_factor = exp (-. market_data.risk_free_rate *. maturity) in
    let mc_price = Tensor.(mean payoffs * Scalar.float discount_factor) in

    let s = Tensor.to_float0_exn market_data.index_price in
    let analytical_price = black_scholes_call s strike market_data.risk_free_rate maturity (Tensor.to_float0_exn market_data.index_volatility) in
    let mc_analytical = Tensor.(mean (max (final_prices - Scalar.float strike) (Scalar.float 0.))) in
    let beta = Tensor.((mc_price - Scalar.float analytical_price) / (mc_analytical - Scalar.float analytical_price)) in
    let adjusted_price = Tensor.(mc_price - beta * (mc_analytical - Scalar.float analytical_price)) in

    let std_dev = Tensor.std payoffs ~dim:[0] ~unbiased:true ~keepdim:false in
    let confidence_interval = Tensor.(Scalar.float 1.96 * std_dev / sqrt (Scalar.float 100000.)) in

    Lwt.return (adjusted_price, confidence_interval)

  let price_path_dependent_option params market_data option num_paths num_steps =
    let%lwt paths = simulate_parallel params market_data num_paths num_steps in
    match option with
    | Autocallable contract -> price_autocallable params market_data contract paths
    | AthenaJet contract -> price_athena_jet params market_data contract paths
    | DailyBarrierKnockIn { barrier; maturity } -> price_daily_barrier_knock_in params market_data barrier maturity paths

  let price_autocallable params market_data contract paths =
    let num_paths, num_steps, _ = Tensor.shape paths in
    let payoffs = Tensor.zeros [num_paths] in
    let observation_dates = Array.map (fun t -> int_of_float (t /. market_data.futures_maturities.(1) *. float num_steps)) contract.observation_dates in
    
    for i = 0 to num_paths - 1 do
      let mutable autocalled = false in
      let mutable j = 0 in
      while not autocalled && j < Array.length observation_dates do
        let t = observation_dates.(j) in
        let price = Tensor.get paths [|i; t; 0|] in
        if price >= contract.barriers.(j) then begin
          Tensor.set payoffs [|i|] (1.0 +. contract.coupons.(j));
          autocalled <- true
        end;
        j <- j + 1
      done;
      
      if not autocalled then
        let final_price = Tensor.get paths [|i; num_steps - 1; 0|] in
        if final_price >= contract.final_barrier then
          Tensor.set payoffs [|i|] 1.0
        else
          Tensor.set payoffs [|i|] (final_price /. contract.final_barrier)
    done;
    
    let discount_factor = exp (-. market_data.risk_free_rate *. contract.observation_dates.(Array.length contract.observation_dates - 1)) in
    Lwt.return (Tensor.(mean payoffs * Scalar.float discount_factor |> to_float0_exn))

  let price_athena_jet params market_data contract paths =
    let num_paths, num_steps, _ = Tensor.shape paths in
    let payoffs = Tensor.zeros [num_paths] in
    let early_redemption_step = int_of_float (contract.early_redemption_date /. market_data.futures_maturities.(1) *. float num_steps) in
    let maturity_step = num_steps - 1 in
    
    for i = 0 to num_paths - 1 do
      let early_price = Tensor.get paths [|i; early_redemption_step; 0|] in
      if early_price >= contract.early_redemption_barrier then
        Tensor.set payoffs [|i|] (1.0 +. contract.early_redemption_coupon)
      else
        let final_price = Tensor.get paths [|i; maturity_step; 0|] in
        if final_price >= contract.final_barrier then
          Tensor.set payoffs [|i|] (1.0 +. contract.participation_rate *. (final_price /. Tensor.get paths [|i; 0; 0|] -. 1.0))
        else if final_price >= 0.7 *. Tensor.get paths [|i; 0; 0|] then
          Tensor.set payoffs [|i|] 1.0
        else
          Tensor.set payoffs [|i|] (final_price /. Tensor.get paths [|i; 0; 0|])
    done;
    
    let discount_factor = exp (-. market_data.risk_free_rate *. contract.maturity) in
    Lwt.return (Tensor.(mean payoffs * Scalar.float discount_factor |> to_float0_exn))

  let price_daily_barrier_knock_in params market_data barrier maturity paths =
    let num_paths, num_steps, _ = Tensor.shape paths in
    let payoffs = Tensor.zeros [num_paths] in
    let maturity_step = int_of_float (maturity /. market_data.futures_maturities.(1) *. float num_steps) in
    
    for i = 0 to num_paths - 1 do
      let mutable knocked_in = false in
      let mutable t = 0 in
      while not knocked_in && t < maturity_step do
        if Tensor.get paths [|i; t; 0|] <= barrier then
          knocked_in <- true;
        t <- t + 1
      done;
      
      if knocked_in then
        let final_price = Tensor.get paths [|i; maturity_step; 0|] in
        Tensor.set payoffs [|i|] (max 0.0 (Tensor.get paths [|i; 0; 0|] -. final_price))
      else
        Tensor.set payoffs [|i|] 0.0
    done;
    
    let discount_factor = exp (-. market_data.risk_free_rate *. maturity) in
    Lwt.return (Tensor.(mean payoffs * Scalar.float discount_factor |> to_float0_exn))

  let compute_var params market_data confidence_level horizon =
    let%lwt paths = simulate_parallel params market_data 100000 (int_of_float horizon) in
    let final_prices = Tensor.select paths 1 (-1) in
    let sorted_prices = Tensor.sort final_prices ~dim:0 ~descending:false in
    let var_index = int_of_float (float 100000 *. (1. -. confidence_level)) in
    Lwt.return (Tensor.get sorted_prices [|var_index|])

  let compute_expected_shortfall params market_data confidence_level horizon =
    let%lwt var = compute_var params market_data confidence_level horizon in
    let%lwt paths = simulate_parallel params market_data 100000 (int_of_float horizon) in
    let final_prices = Tensor.select paths 1 (-1) in
    let losses = Tensor.(neg (final_prices - Scalar.float (Tensor.to_float0_exn market_data.index_price))) in
    let extreme_losses = Tensor.(mean (masked_select losses (losses >= Scalar.float var))) in
    Lwt.return extreme_losses

  let calibrate_local_volatility market_data =
    calibrate_local_volatility_surface_improved market_data.futures_prices market_data.vanilla_option_prices market_data.vanilla_option_strikes market_data.vanilla_option_maturities
end
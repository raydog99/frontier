open Torch

module Make (P : Mace_params.MACE_PARAMS) = struct
  type portfolio = {
    weights : Tensor.t;
    returns : Tensor.t;
  }

  type model = {
    f : Tensor.t -> Tensor.t;
    g : Tensor.t -> Tensor.t;
  }

  type tree_node =
    | Leaf of float
    | Node of {
        feature: int;
        threshold: float;
        left: tree_node;
        right: tree_node;
      }

  let create_portfolio weights returns =
    { weights; returns }

  let create_model f g =
    { f; g }

  let mean tensor =
    Tensor.mean tensor |> Tensor.float_value

  let mse left_indices right_indices targets =
    let left_mean = mean (Tensor.index_select targets 0 left_indices) in
    let right_mean = mean (Tensor.index_select targets 0 right_indices) in
    let left_mse = Tensor.sub (Tensor.index_select targets 0 left_indices) (Tensor.full_like (Tensor.index_select targets 0 left_indices) left_mean) |> Tensor.pow_scalar 2.0 |> mean in
    let right_mse = Tensor.sub (Tensor.index_select targets 0 right_indices) (Tensor.full_like (Tensor.index_select targets 0 right_indices) right_mean) |> Tensor.pow_scalar 2.0 |> mean in
    (Tensor.shape left_indices |> List.hd |> float_of_int) *. left_mse +. (Tensor.shape right_indices |> List.hd |> float_of_int) *. right_mse

  let best_split features targets =
    let num_samples, num_features = Tensor.shape features in
    let mtrys = Array.init (int_of_float (float_of_int num_features *. P.mtry)) (fun _ -> Random.int num_features) in
    Array.fold_left (fun (best_feature, best_threshold, best_mse) feature ->
      let feature_values = Tensor.select features 1 feature in
      let sorted_indices = Tensor.argsort feature_values in
      let sorted_features = Tensor.index_select feature_values 0 sorted_indices in
      let sorted_targets = Tensor.index_select targets 0 sorted_indices in
      Tensor.fold_left (fun (best_threshold, best_mse) threshold ->
        let left_indices = Tensor.lt feature_values threshold in
        let right_indices = Tensor.gt feature_values threshold in
        let current_mse = mse left_indices right_indices targets in
        if current_mse < best_mse then (threshold, current_mse) else (best_threshold, best_mse)
      ) (best_threshold, best_mse) sorted_features
      |> fun (threshold, mse) -> if mse < best_mse then (feature, threshold, mse) else (best_feature, best_threshold, best_mse)
    ) (-1, 0.0, Float.infinity) mtrys

  let rec build_tree features targets depth =
    let num_samples = Tensor.shape features |> List.hd in
    if num_samples <= P.min_samples_leaf || depth = 0 then
      Leaf (mean targets)
    else
      let feature, threshold, _ = best_split features targets in
      let left_mask = Tensor.lt (Tensor.select features 1 feature) (Tensor.scalar threshold) in
      let right_mask = Tensor.gt (Tensor.select features 1 feature) (Tensor.scalar threshold) in
      let left_features = Tensor.masked_select features left_mask |> Tensor.reshape [-1; Tensor.shape features |> List.tl |> List.hd] in
      let right_features = Tensor.masked_select features right_mask |> Tensor.reshape [-1; Tensor.shape features |> List.tl |> List.hd] in
      let left_targets = Tensor.masked_select targets left_mask in
      let right_targets = Tensor.masked_select targets right_mask in
      Node {
        feature;
        threshold;
        left = build_tree left_features left_targets (depth - 1);
        right = build_tree right_features right_targets (depth - 1);
      }

  let rec predict_tree tree features =
    match tree with
    | Leaf value -> Tensor.full [Tensor.shape features |> List.hd] value
    | Node { feature; threshold; left; right } ->
        let mask = Tensor.lt (Tensor.select features 1 feature) (Tensor.scalar threshold) in
        let left_predictions = predict_tree left (Tensor.masked_select features mask |> Tensor.reshape [-1; Tensor.shape features |> List.tl |> List.hd]) in
        let right_predictions = predict_tree right (Tensor.masked_select features (Tensor.logical_not mask) |> Tensor.reshape [-1; Tensor.shape features |> List.tl |> List.hd]) in
        Tensor.where mask left_predictions right_predictions

  let random_forest_step features targets =
    let trees = Array.init P.num_trees (fun _ ->
      let sample_indices = Tensor.randint (Tensor.shape features |> List.hd) ~high:(Tensor.shape features |> List.hd) [Tensor.shape features |> List.hd] in
      let sample_features = Tensor.index_select features 0 sample_indices in
      let sample_targets = Tensor.index_select targets 0 sample_indices in
      build_tree sample_features sample_targets (int_of_float (log (float_of_int (Tensor.shape features |> List.hd))))
    ) in
    let predict x =
      let predictions = Array.map (fun tree -> predict_tree tree x) trees in
      Tensor.stack (Array.to_list predictions) ~dim:0 |> Tensor.mean ~dim:[0]
    in
    let predictions = predict features in
    (predict, predictions)

  let ridge_regression_step features targets lambda =
    let num_samples, num_features = Tensor.shape features in
    let x_t = Tensor.transpose features ~dim0:0 ~dim1:1 in
    let x_t_x = Tensor.matmul x_t features in
    let identity = Tensor.eye num_features in
    let regularized_x_t_x = Tensor.add x_t_x (Tensor.mul_scalar identity lambda) in
    let x_t_y = Tensor.matmul x_t targets in
    let weights = Tensor.matmul (Tensor.inverse regularized_x_t_x) x_t_y in
    let predict x = Tensor.matmul x weights in
    (predict, weights)

  let marx_transform features lags =
    let num_samples, num_features = Tensor.shape features in
    let marx_features = ref [] in
    for i = 1 to lags do
      let ma = Tensor.avg_pool2d features ~ksize:[i; 1] ~stride:[1; 1] ~padding:[0; 0] in
      marx_features := ma :: !marx_features
    done;
    Tensor.cat (List.rev !marx_features) ~dim:1

  let nonlinear_mean_reversion_machine portfolio features =
    let num_samples, num_features = Tensor.shape features in
    let portfolio_returns = Tensor.matmul portfolio.returns (Tensor.unsqueeze portfolio.weights 1) in
    let lagged_returns = Tensor.cat [
      Tensor.narrow portfolio_returns ~dim:0 ~start:0 ~length:(num_samples - 1);
      Tensor.zeros [1; 1]
    ] ~dim:0 in
    let marx_features = marx_transform lagged_returns P.marx_lags in
    Tensor.cat [features; marx_features] ~dim:1

  let train_mace portfolio features =
    let features = nonlinear_mean_reversion_machine portfolio features in
    let train_size = int_of_float (float_of_int (Tensor.shape features |> List.hd) *. (1. -. P.validation_split)) in
    let train_features = Tensor.narrow features ~dim:0 ~start:0 ~length:train_size in
    let train_targets = Tensor.narrow portfolio.returns ~dim:0 ~start:0 ~length:train_size in
    let val_features = Tensor.narrow features ~dim:0 ~start:train_size ~length:((Tensor.shape features |> List.hd) - train_size) in
    let val_targets = Tensor.narrow portfolio.returns ~dim:0 ~start:train_size ~length:((Tensor.shape portfolio.returns |> List.hd) - train_size) in
    
    let num_features = Tensor.shape features |> List.tl |> List.hd in
    let weights = Tensor.ones [num_features; 1] in
    let f = fun x -> x in
    let g = fun y -> y in

    let rec iterate f g weights iter best_val_loss best_weights patience =
      if iter >= P.max_iterations || patience <= 0 then (f, g, best_weights)
      else
        let f_new, f_predictions = random_forest_step train_features (Tensor.matmul train_targets weights) in
        let g_new, g_weights = ridge_regression_step train_features f_predictions in
        let weights_new = Tensor.add weights (Tensor.mul_scalar (Tensor.sub g_weights weights) P.learning_rate) in
        
        let val_predictions = f_new val_features |> g_new in
        let val_loss = Tensor.mse_loss val_predictions val_targets |> Tensor.float_value in
        
        if val_loss < best_val_loss then
          iterate f_new g_new weights_new (iter + 1) val_loss weights_new P.early_stopping_rounds
        else
          iterate f_new g_new weights_new (iter + 1) best_val_loss best_weights (patience - 1)
    in

    let f_final, g_final, weights_final = iterate f g weights 0 Float.infinity weights P.early_stopping_rounds in
    create_model f_final g_final

  let predict model features =
    let f_pred = model.f features in
    model.g f_pred

  let calculate_r_squared predictions targets =
    let residuals = Tensor.sub predictions targets in
    let ss_res = Tensor.sum (Tensor.pow residuals 2) in
    let ss_tot = Tensor.sum (Tensor.pow (Tensor.sub targets (Tensor.mean targets)) 2) in
    1.0 -. (Tensor.float_value ss_res /. Tensor.float_value ss_tot)

  let mean_variance_optimization returns risk_free_rate =
    let num_assets = Tensor.shape returns |> List.tl |> List.hd in
    let mean_returns = Tensor.mean returns ~dim:[0] in
    let cov_matrix = Tensor.cov returns in
    
    let optimize weights =
      let portfolio_return = Tensor.dot mean_returns weights in
      let portfolio_risk = Tensor.(sqrt (dot weights (matmul cov_matrix weights))) in
      let sharpe_ratio = (portfolio_return -. risk_free_rate) /. portfolio_risk in
      -. sharpe_ratio  (* Negative to maximize *)
    in
    
    let initial_weights = Tensor.full [num_assets] (1.0 /. float_of_int num_assets) in
    let optimizer = Optimizer.adam [initial_weights] ~lr:0.01 in
    
    let rec optimize_iter iter best_weights best_score =
      if iter >= 1000 then best_weights
      else begin
        Optimizer.zero_grad optimizer;
        let loss = optimize initial_weights in
        Tensor.backward loss;
        Optimizer.step optimizer;
        let current_score = -. (Tensor.float_value loss) in
        if current_score > best_score then
          optimize_iter (iter + 1) (Tensor.copy initial_weights) current_score
        else
          optimize_iter (iter + 1) best_weights best_score
      end
    in
    
    optimize_iter 0 initial_weights Float.neg_infinity

  let compare_mace_vs_single_stock features targets =
    let mace_model = train_mace (create_portfolio (Tensor.ones [Tensor.shape targets |> List.tl |> List.hd]) targets) features in
    let mace_predictions = predict mace_model features in
    let mace_r_squared = calculate_r_squared mace_predictions targets in
    
    let single_stock_r_squared = 
      let num_stocks = Tensor.shape targets |> List.tl |> List.hd in
      let r_squared_sum = ref 0.0 in
      for i = 0 to num_stocks - 1 do
        let stock_targets = Tensor.select targets 1 i in
        let stock_model = train_mace (create_portfolio (Tensor.ones [1]) stock_targets) features in
        let stock_predictions = predict stock_model features in
        r_squared_sum := !r_squared_sum +. calculate_r_squared stock_predictions stock_targets
      done;
      !r_squared_sum /. float_of_int num_stocks
    in
    
    (mace_r_squared, single_stock_r_squared)

  let hyperparameter_tuning features targets =
    let learning_rates = [0.01; 0.05; 0.1; 0.2] in
    let ridge_lambdas = [0.1; 1.0; 10.0; 100.0] in
    let num_trees_list = [50; 100; 200; 500] in
    
    let best_params = ref (0.0, 0.0, 0, Float.neg_infinity) in
    
    List.iter (fun lr ->
      List.iter (fun lambda ->
        List.iter (fun num_trees ->
          let module CustomParams = struct
            include P
            let learning_rate = lr
            let ridge_lambda = lambda
            let num_trees = num_trees
          end in
          let module CustomMACE = Make(CustomParams) in
          let model = CustomMACE.train_mace (create_portfolio (Tensor.ones [Tensor.shape targets |> List.tl |> List.hd]) targets) features in
          let predictions = CustomMACE.predict model features in
          let r_squared = CustomMACE.calculate_r_squared predictions targets in
          if r_squared > (let _, _, _, best_r_squared = !best_params in best_r_squared) then
            best_params := (lr, lambda, num_trees, r_squared)
        ) num_trees_list
      ) ridge_lambdas
    ) learning_rates;
    
    !best_params

  let daily_trading_strategy model portfolio features risk_aversion =
    let num_samples = Tensor.shape features |> List.hd in
    let trading_positions = Tensor.zeros [num_samples; P.portfolio_size] in
    let returns = ref [] in

    for t = 0 to num_samples - 1 do
      let current_features = Tensor.narrow features ~dim:0 ~start:t ~length:1 in
      let prediction = predict model current_features in
      let current_return = Tensor.narrow portfolio.returns ~dim:0 ~start:t ~length:1 in
      let position = Tensor.div prediction (Tensor.mul_scalar (Tensor.pow current_return 2.0) risk_aversion) in
      let position = Tensor.clamp position ~min:(-2.0) ~max:2.0 in
      Tensor.copy_ (Tensor.narrow trading_positions ~dim:0 ~start:t ~length:1) position;
      let realized_return = Tensor.sum (Tensor.mul position current_return) |> Tensor.float_value in
      returns := realized_return :: !returns
    done;

    let trading_returns = Tensor.of_float1 (Array.of_list (List.rev !returns)) in
    (trading_positions, trading_returns)

  let loose_bag_mace portfolio features =
    let models = Array.init P.num_trees (fun _ ->
      let sample_indices = Tensor.randint (Tensor.shape features |> List.hd) ~high:(Tensor.shape features |> List.hd) [Tensor.shape features |> List.hd] in
      let sample_features = Tensor.index_select features 0 sample_indices in
      let sample_returns = Tensor.index_select portfolio.returns 0 sample_indices in
      let sample_portfolio = { portfolio with returns = sample_returns } in
      
      let rec train_until_target () =
        let model = train_mace sample_portfolio sample_features in
        let predictions = predict model sample_features in
        let r_squared = calculate_r_squared predictions sample_returns in
        if r_squared >= P.loose_bag_r_squared_target then model
        else train_until_target ()
      in
      
      train_until_target ()
    ) in
    
    let predict_loose_bagged x =
      let predictions = Array.map (fun model -> predict model x) models in
      Tensor.mean (Tensor.stack (Array.to_list predictions) ~dim:0) ~dim:[0]
    in
    
    create_model predict_loose_bagged (fun x -> x)

  let factor_based_analysis predictions returns =
    let num_factors = P.num_factors in
    let factor_returns = Tensor.randn [Tensor.shape returns |> List.hd; num_factors] in
    
    let x = Tensor.cat [Tensor.ones [Tensor.shape factor_returns |> List.hd; 1]; factor_returns] ~dim:1 in
    let y = returns in
    let x_t = Tensor.transpose x ~dim0:0 ~dim1:1 in
    let beta = Tensor.matmul (Tensor.matmul (Tensor.inverse (Tensor.matmul x_t x)) x_t) y in
    
    let alpha = Tensor.get beta [0] |> Tensor.float_value in
    let factor_loadings = Tensor.narrow beta ~dim:0 ~start:1 ~length:num_factors |> Tensor.to_float1 in
    
    let residuals = Tensor.sub y (Tensor.matmul x beta) in
    let r_squared = 1.0 -. (Tensor.sum (Tensor.pow residuals 2.0) |> Tensor.float_value) /. (Tensor.sum (Tensor.pow (Tensor.sub y (Tensor.mean y)) 2.0) |> Tensor.float_value) in
    
    (alpha, factor_loadings, r_squared)

  let calculate_sharpe_ratio returns risk_free_rate =
    let excess_returns = Tensor.sub returns (Tensor.full_like returns risk_free_rate) in
    let mean_excess_return = Tensor.mean excess_returns in
    let std_dev = Tensor.std excess_returns in
    Tensor.div mean_excess_return std_dev |> Tensor.float_value

  let calculate_max_drawdown returns =
    let cumulative_returns = Tensor.cumprod (Tensor.add returns 1.0) ~dim:0 in
    let peak = Tensor.cummax cumulative_returns ~dim:0 |> fst in
    let drawdown = Tensor.div (Tensor.sub peak cumulative_returns) peak in
    Tensor.max drawdown |> Tensor.float_value

  let calculate_omega_ratio returns threshold =
    let gains = Tensor.relu (Tensor.sub returns threshold) in
    let losses = Tensor.relu (Tensor.sub threshold returns) in
    let expected_gain = Tensor.mean gains in
    let expected_loss = Tensor.mean losses in
    Tensor.div expected_gain expected_loss |> Tensor.float_value

  let run_mace_strategy portfolio features =
    let model = loose_bag_mace portfolio features in
    let (trading_positions, trading_returns) = daily_trading_strategy model portfolio features P.risk_aversion in
    let sharpe_ratio = calculate_sharpe_ratio trading_returns 0.02 in (* Assuming 2% risk-free rate *)
    let max_drawdown = calculate_max_drawdown trading_returns in
    let omega_ratio = calculate_omega_ratio trading_returns 0.02 in (* Assuming 2% threshold *)
    let factor_analysis = factor_based_analysis trading_returns portfolio.returns in
    (trading_returns, trading_positions, sharpe_ratio, max_drawdown, omega_ratio, factor_analysis)
end
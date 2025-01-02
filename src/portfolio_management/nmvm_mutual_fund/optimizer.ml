open Torch

type strategy =
  | Unconstrained
  | Constrained of Portfolio.constraint_type list
  | MeanVariance of float
  | MaximumSharpe
  | MinimumVariance
  | EqualWeight
  | RiskParity
  | BlackLitterman of (Tensor.t * Tensor.t)

let optimize model strategy utility_opt initial_wealth risk_free_rate =
  let n = Tensor.shape model.Nmvm.mu |> List.hd in
  match strategy with
  | Unconstrained ->
      (match utility_opt with
       | Some utility -> 
           let weights = Tensor.randn [n] ~requires_grad:true in
           let optimizer = Optimizer.adam [weights] ~lr:0.01 in
           for _ = 1 to 1000 do
             Optimizer.zero_grad optimizer;
             let portfolio = Portfolio.create weights in
             let wealth = Portfolio.expected_wealth model portfolio initial_wealth risk_free_rate in
             let utility_value = utility wealth in
             let loss = Tensor.neg utility_value in
             Tensor.backward loss;
             Optimizer.step optimizer
           done;
           Portfolio.create weights
       | None -> failwith "Utility function is required for unconstrained optimization")
  | Constrained constraints ->
      (match utility_opt with
       | Some utility -> 
           let weights = Tensor.randn [n] ~requires_grad:true in
           let optimizer = Optimizer.adam [weights] ~lr:0.01 in
           for _ = 1 to 1000 do
             Optimizer.zero_grad optimizer;
             let portfolio = Portfolio.create weights |> Portfolio.apply_constraints constraints in
             let wealth = Portfolio.expected_wealth model portfolio initial_wealth risk_free_rate in
             let utility_value = utility wealth in
             let loss = Tensor.neg utility_value in
             Tensor.backward loss;
             Optimizer.step optimizer
           done;
           Portfolio.create weights |> Portfolio.apply_constraints constraints
       | None -> failwith "Utility function is required for constrained optimization")
  | MeanVariance target_return ->
      let weights = Tensor.randn [n] ~requires_grad:true in
      let optimizer = Optimizer.adam [weights] ~lr:0.01 in
      for _ = 1 to 1000 do
        Optimizer.zero_grad optimizer;
        let portfolio = Portfolio.create weights in
        let expected_return = Tensor.(dot portfolio (Nmvm.expected_return model)) in
        let variance = Portfolio.variance model portfolio in
        let loss = Tensor.(add variance (mul (pow (sub expected_return (Scalar.f target_return)) (Scalar.f 2.0)) (Scalar.f 100.0))) in
        Tensor.backward loss;
        Optimizer.step optimizer
      done;
      Portfolio.create weights
  | MaximumSharpe ->
      let weights = Tensor.randn [n] ~requires_grad:true in
      let optimizer = Optimizer.adam [weights] ~lr:0.01 in
      for _ = 1 to 1000 do
        Optimizer.zero_grad optimizer;
        let portfolio = Portfolio.create weights in
        let sharpe = Portfolio.sharpe_ratio model portfolio risk_free_rate in
        let loss = Tensor.neg (Tensor.scalar_float sharpe) in
        Tensor.backward loss;
        Optimizer.step optimizer
      done;
      Portfolio.create weights
  | MinimumVariance ->
      let weights = Tensor.randn [n] ~requires_grad:true in
      let optimizer = Optimizer.adam [weights] ~lr:0.01 in
      for _ = 1 to 1000 do
        Optimizer.zero_grad optimizer;
        let portfolio = Portfolio.create weights in
        let variance = Portfolio.variance model portfolio in
        Tensor.backward variance;
        Optimizer.step optimizer
      done;
      Portfolio.create weights
  | EqualWeight ->
      Tensor.ones [n] |> Tensor.div_scalar (Float.of_int n) |> Portfolio.create
  | RiskParity ->
      let weights = Tensor.randn [n] ~requires_grad:true in
      let optimizer = Optimizer.adam [weights] ~lr:0.01 in
      for _ = 1 to 1000 do
        Optimizer.zero_grad optimizer;
        let portfolio = Portfolio.create weights in
        let cov = Nmvm.covariance model in
        let risk_contributions = Tensor.(mul portfolio (matmul cov portfolio)) in
        let total_risk = Tensor.sum risk_contributions in
        let target_risk = Tensor.div total_risk (Tensor.scalar_float (Float.of_int n)) in
        let loss = Tensor.(sum (pow (sub risk_contributions target_risk) (Scalar.f 2.0))) in
        Tensor.backward loss;
        Optimizer.step optimizer
      done;
      Portfolio.create weights
  | BlackLitterman (views, confidences) ->
      let prior_returns = Nmvm.expected_return model in
      let prior_cov = Nmvm.covariance model in
      let tau = 0.05 in  (* Scalar representing the uncertainty in the prior *)
      let p = Tensor.eye n in  (* Assuming each view corresponds to a single asset *)
      let omega = Tensor.(mul_scalar (diag confidences) tau) in
      let posterior_cov = Tensor.(
        inverse (add (inverse prior_cov) (matmul (transpose 0 1 p) (matmul (inverse omega) p)))
      ) in
      let posterior_returns = Tensor.(
        matmul posterior_cov 
          (add (matmul (inverse prior_cov) prior_returns)
               (matmul (transpose 0 1 p) (matmul (inverse omega) views)))
      ) in
      let black_litterman_model = Nmvm.create posterior_returns model.Nmvm.gamma model.Nmvm.sigma model.Nmvm.z_dist in
      optimize black_litterman_model (MeanVariance (Tensor.mean posterior_returns |> Tensor.to_float0)) utility_opt initial_wealth risk_free_rate

let cross_validate model strategy utility initial_wealth risk_free_rate k_folds =
  let n = Tensor.shape model.Nmvm.mu |> List.hd in
  let fold_size = n / k_folds in
  let performances = List.init k_folds (fun i ->
    let test_indices = List.init fold_size (fun j -> i * fold_size + j) in
    let train_indices = List.filter (fun j -> not (List.mem j test_indices)) (List.init n (fun x -> x)) in
    let train_model = Nmvm.create
      (Tensor.index_select model.Nmvm.mu ~dim:0 ~index:(Tensor.of_int1 train_indices))
      (Tensor.index_select model.Nmvm.gamma ~dim:0 ~index:(Tensor.of_int1 train_indices))
      (Tensor.index_select model.Nmvm.sigma ~dim:0 ~index:(Tensor.of_int1 train_indices))
      model.Nmvm.z_dist
    in
    let optimal_portfolio = optimize train_model strategy (Some utility) initial_wealth risk_free_rate in
    let test_performance = Portfolio.sharpe_ratio model optimal_portfolio risk_free_rate in
    test_performance
  ) in
  List.fold_left (+.) 0. performances /. float_of_int k_folds
open Torch

type t = Tensor.t
type constraint_type = 
  | NoShortSelling
  | SectorExposure of (int list * float * float) list
  | MaxWeight of float
  | MinWeight of float
  | TurnoverLimit of float

let create weights =
  if Tensor.(min weights) |> Tensor.to_float < 0. then
    failwith "Portfolio weights must be non-negative";
  let sum = Tensor.sum weights in
  if Tensor.to_float0 sum = 0. then
    failwith "Portfolio weights sum to zero";
  Tensor.div weights sum

let expected_wealth model portfolio initial_wealth risk_free_rate =
  if initial_wealth <= 0. then failwith "Initial wealth must be positive";
  let returns = Nmvm.expected_return model in
  Tensor.(initial_wealth * (one_scalar + risk_free_rate + dot portfolio returns))

let variance model portfolio =
  let cov = Nmvm.covariance model in
  Tensor.(dot portfolio (matmul cov portfolio))

let sharpe_ratio model portfolio risk_free_rate =
  let expected_return = Tensor.(dot portfolio (Nmvm.expected_return model)) in
  let std_dev = Tensor.sqrt (variance model portfolio) in
  Tensor.((sub expected_return (Scalar.f risk_free_rate)) / std_dev)
  |> Tensor.to_float0

let apply_constraints portfolio constraints =
  List.fold_left (fun p constraint_type ->
    match constraint_type with
    | NoShortSelling -> 
        Tensor.max p (Tensor.zeros_like p)
    | SectorExposure sectors ->
        List.fold_left (fun p (indices, min_exposure, max_exposure) ->
          let sector_weights = Tensor.index_select p ~dim:0 ~index:(Tensor.of_int1 indices) in
          let sector_sum = Tensor.sum sector_weights in
          if Tensor.to_float0 sector_sum < min_exposure then
            Tensor.index_copy p ~dim:0 ~index:(Tensor.of_int1 indices) 
              (Tensor.mul sector_weights (Tensor.scalar_float (min_exposure /. Tensor.to_float0 sector_sum)))
          else if Tensor.to_float0 sector_sum > max_exposure then
            Tensor.index_copy p ~dim:0 ~index:(Tensor.of_int1 indices)
              (Tensor.mul sector_weights (Tensor.scalar_float (max_exposure /. Tensor.to_float0 sector_sum)))
          else
            p
        ) p sectors
    | MaxWeight max_weight ->
        Tensor.min p (Tensor.scalar_float max_weight)
    | MinWeight min_weight ->
        Tensor.max p (Tensor.scalar_float min_weight)
    | TurnoverLimit limit ->
        let sum = Tensor.sum p in
        let scaled_p = Tensor.div p sum in
        let diff = Tensor.sub scaled_p p in
        let turnover = Tensor.sum (Tensor.abs diff) |> Tensor.to_float0 in
        if turnover > limit then
          Tensor.add p (Tensor.mul diff (Tensor.scalar_float (limit /. turnover)))
        else
          p
  ) portfolio constraints

let rebalance portfolio model constraints =
  let n = Tensor.shape portfolio |> List.hd in
  let weights = Tensor.randn [n] ~requires_grad:true in
  
  let optimizer = Optimizer.adam [weights] ~lr:0.01 in
  
  for _ = 1 to 100 do
    Optimizer.zero_grad optimizer;
    let new_portfolio = create weights |> apply_constraints constraints in
    let turnover = Tensor.(sum (abs (sub new_portfolio portfolio))) in
    let expected_return = Tensor.(dot new_portfolio (Nmvm.expected_return model)) in
    let variance = variance model new_portfolio in
    let loss = Tensor.(neg expected_return + (Scalar.f 0.5) * variance + (Scalar.f 10.0) * turnover) in
    Tensor.backward loss;
    Optimizer.step optimizer
  done;
  
  create weights |> apply_constraints constraints

let turnover old_portfolio new_portfolio =
  Tensor.(sum (abs (sub new_portfolio old_portfolio))) |> Tensor.to_float0

let tracking_error model portfolio benchmark =
  let portfolio_return = Tensor.(dot portfolio (Nmvm.expected_return model)) in
  let benchmark_return = Tensor.(dot benchmark (Nmvm.expected_return model)) in
  Tensor.(sub portfolio_return benchmark_return) |> Tensor.to_float0
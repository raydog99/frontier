open Torch

type t = {
  s0: float;
  r: float;
  sigma: float;
  t: float;
  n: int;
  u: float;
  d: float;
  pu: float;
  pm: float;
  pd: float;
  dt: float;
}

type multi_asset_t = {
  assets: t array;
  correlation_matrix: float array array;
}

let create s0 r sigma t n =
  let dt = t /. float_of_int n in
  let u = exp (sigma *. sqrt dt) in
  let d = 1. /. u in
  let pu = (exp (r *. dt) -. d) /. (u -. d) in
  let pd = (u -. exp (r *. dt)) /. (u -. d) in
  let pm = 1. -. pu -. pd in
  { s0; r; sigma; t; n; u; d; pu; pm; pd; dt }

let create_with_parameters s0 r sigma t n u d pu pm pd =
  let dt = t /. float_of_int n in
  { s0; r; sigma; t; n; u; d; pu; pm; pd; dt }

let create_multi_asset assets correlation_matrix =
  { assets; correlation_matrix }

let stock_price tree i j =
  tree.s0 *. (tree.u ** float_of_int i) *. (tree.d ** float_of_int (j - i))

let discount_factor tree = exp (-. tree.r *. tree.dt)

let risk_neutral_probabilities tree =
  let df = discount_factor tree in
  let qu = tree.pu *. df in
  let qm = tree.pm *. df in
  let qd = tree.pd *. df in
  (qu, qm, qd)

let update_parameters tree r sigma pu pm pd =
  let u = exp (sigma *. sqrt tree.dt) in
  let d = 1. /. u in
  { tree with r; sigma; u; d; pu; pm; pd }

let expected_return tree =
  tree.pu *. tree.u +. tree.pm +. tree.pd *. tree.d -. 1.

let variance tree =
  let er = expected_return tree in
  tree.pu *. (tree.u -. 1. -. er) ** 2. +.
  tree.pm *. (0. -. er) ** 2. +.
  tree.pd *. (tree.d -. 1. -. er) ** 2.

let generate_correlated_returns multi_asset_tree =
  let n = Array.length multi_asset_tree.assets in
  let uncorrelated_returns = Tensor.rand [n] in
  let correlation_matrix = Tensor.of_float2 multi_asset_tree.correlation_matrix in
  let cholesky = Tensor.cholesky correlation_matrix in
  let correlated_returns = Tensor.mm cholesky uncorrelated_returns in
  Tensor.to_float1 correlated_returns

let create_from_stochastic_process process s0 r t n =
  let dt = t /. float_of_int n in
  let path = Stochastic_process.simulate process dt n s0 in
  let u = Array.fold_left max 0. path /. s0 in
  let d = Array.fold_left min max_float path /. s0 in
  let p = (exp(r *. dt) -. d) /. (u -. d) in
  create_with_parameters s0 r ((log u -. log d) /. sqrt dt) t n u d p (1. -. p) 0.

let log_likelihood tree returns =
  Array.fold_left (fun acc return ->
    acc +. log (
      if return <= tree.d -. 1. then tree.pd
      else if return >= tree.u -. 1. then tree.pu
      else tree.pm
    )
  ) 0. returns

let optimize_parameters returns initial_tree =
  let objective params =
    let [| r; sigma; pu; pm |] = params in
    let pd = 1. -. pu -. pm in
    let u = exp (sigma *. sqrt initial_tree.dt) in
    let d = 1. /. u in
    let tree = create_with_parameters initial_tree.s0 r sigma initial_tree.t initial_tree.n u d pu pm pd in
    -. log_likelihood tree returns
  in
  let initial_params = Tensor.of_float1 [| initial_tree.r; initial_tree.sigma; initial_tree.pu; initial_tree.pm |] in
  let optimizer = Optimizer.adam ~lr:0.01 [] in
  let rec optimize iter params =
    if iter = 0 then params
    else
      let loss = objective params in
      Optimizer.backward optimizer [loss];
      Optimizer.step optimizer;
      Optimizer.zero_grad optimizer;
      optimize (iter - 1) (Tensor.detach params)
  in
  let optimized_params = optimize 1000 initial_params in
  let [| r; sigma; pu; pm |] = Tensor.to_float1 optimized_params in
  let pd = 1. -. pu -. pm in
  let u = exp (sigma *. sqrt initial_tree.dt) in
  let d = 1. /. u in
  create_with_parameters initial_tree.s0 r sigma initial_tree.t initial_tree.n u d pu pm pd
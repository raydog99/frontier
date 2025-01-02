open Torch

type t = {probs: Tensor.t; mean: float; cumsum: Tensor.t}

let create probs =
  let probs = Tensor.div probs (Tensor.sum probs) in
  let n = float_of_int (Tensor.shape probs).(0) in
  let indices = Tensor.arange ~end_:(Float n) ~options:(Kind Float, Device Cpu) in
  let mean = Tensor.(sum (probs * indices) |> to_float0) in
  let cumsum = Tensor.cumsum probs ~dim:0 in
  {probs; mean; cumsum}

let get_probs dist = dist.probs
let get_mean dist = dist.mean
let get_cumsum dist = dist.cumsum

let gini_index dist =
  let n = float_of_int (Tensor.shape dist.probs).(0) in
  let indices = Tensor.arange ~end_:(Float n) ~options:(Kind Float, Device Cpu) in
  let lorenz_curve = Tensor.div dist.cumsum (Tensor.sum dist.probs) in
  let area = Tensor.trapz lorenz_curve ~x:indices ~dim:0 in
  1. -. 2. *. (Tensor.to_float0 area)

let wasserstein_distance dist1 dist2 =
  let diff = Tensor.sub (get_cumsum dist1) (get_cumsum dist2) in
  Tensor.(sum (abs diff) |> to_float0)

let l1_distance dist1 dist2 =
  let diff = Tensor.sub (get_probs dist1) (get_probs dist2) in
  Tensor.(sum (abs diff) |> to_float0)

let shifted_bernoulli mu =
  let floor_mu = float_of_int (int_of_float mu) in
  let probs = Tensor.of_float2 [|[|1. -. mu +. floor_mu; mu -. floor_mu; 0.; 0.|]|] in
  create (Tensor.squeeze probs ~dim:[0])

let entropy dist =
  let probs = get_probs dist in
  let log_probs = Tensor.log probs in
  -.(Tensor.(sum (probs * log_probs) |> to_float0))

let to_array dist =
  Tensor.to_float1 (get_probs dist)
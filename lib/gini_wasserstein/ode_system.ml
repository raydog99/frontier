open Torch
open Distribution

type model = 
  | RichBiased 
  | PersuasionPolarization 
  | StickyDispersion
  | CustomModel of (DiscreteDistribution.t -> Tensor.t)

type t = {
  distribution: DiscreteDistribution.t;
  time: float;
  model: model;
}

let create initial_dist model =
  {distribution = initial_dist; time = 0.; model}

let step system dt =
  let dist = system.distribution in
  let probs = DiscreteDistribution.get_probs dist in
  let n = float_of_int (Tensor.shape probs).(0) in
  let indices = Tensor.arange ~end_:(Float n) ~options:(Kind Float, Device Cpu) in
  
  let new_probs = match system.model with
  | RichBiased ->
      let w = Tensor.(sum (probs / indices) |> to_float0) in
      let dp0 = Tensor.get probs [|1|] -. w *. Tensor.get probs [|0|] in
      let dp = Tensor.((probs / indices) + w * (Tensor.slice probs ~dim:0 ~start:(-1) ~end_:None ~step:1) - ((1. / indices) + w) * probs) in
      Tensor.cat [Tensor.of_float0 dp0; dp] ~dim:0
  | PersuasionPolarization ->
      let dp = Tensor.((slice probs ~dim:0 ~start:(-1) ~end_:None ~step:1) * (1. - probs) - probs * (slice probs ~dim:0 ~start:0 ~end_:(-1) ~step:1)) in
      Tensor.cat [dp; Tensor.of_float0 (-.Tensor.sum dp |> to_float0)] ~dim:0
  | StickyDispersion ->
      let p0 = Tensor.get probs [|0|] in
      let dp0 = -.(DiscreteDistribution.get_mean dist -. 1. +. p0) *. p0 in
      let dp = Tensor.((indices * (slice probs ~dim:0 ~start:1 ~end_:None ~step:1) + (DiscreteDistribution.get_mean dist -. 1. +. p0) * (slice probs ~dim:0 ~start:0 ~end_:(-1) ~step:1)) - ((indices - 1.) * probs + (DiscreteDistribution.get_mean dist -. 1. +. p0) * probs)) in
      Tensor.cat [Tensor.of_float0 dp0; dp] ~dim:0
  | CustomModel f -> f dist
  in
  
  let new_probs = Tensor.add probs (Tensor.mul_scalar new_probs dt) in
  let new_probs = Tensor.relu new_probs in
  let new_probs = Tensor.div new_probs (Tensor.sum new_probs) in
  
  {distribution = DiscreteDistribution.create new_probs;
   time = system.time +. dt;
   model = system.model}

let theorem1 system =
  let dist = system.distribution in
  let mu = DiscreteDistribution.get_mean dist in
  let gini_diff = DiscreteDistribution.gini_index dist -. DiscreteDistribution.gini_index (DiscreteDistribution.shifted_bernoulli mu) in
  let w_distance = DiscreteDistribution.wasserstein_distance dist (DiscreteDistribution.shifted_bernoulli mu) in
  let bound = 2. *. mu *. gini_diff in
  w_distance, bound

let theorem2 system =
  let dist = system.distribution in
  let gini = DiscreteDistribution.gini_index dist in
  let l1_distance = DiscreteDistribution.l1_distance dist (DiscreteDistribution.create (Tensor.of_float1 [|1.; 0.; 0.; 0.|])) in
  let bound = 2. *. sqrt (DiscreteDistribution.get_mean dist *. (1. -. gini)) in
  l1_distance, bound

let simulate_system initial_dist model num_steps dt =
  let rec loop system steps results =
    if steps = 0 then List.rev results
    else
      let new_system = ODESystem.step system dt in
      let gini = DiscreteDistribution.gini_index new_system.distribution in
      let entropy = DiscreteDistribution.entropy new_system.distribution in
      let result = (new_system.time, gini, entropy, new_system) in
      loop new_system (steps - 1) (result :: results)
  in
  loop (ODESystem.create initial_dist model) num_steps []

let model_to_string = function
  | ODESystem.RichBiased -> "Rich Biased"
  | ODESystem.PersuasionPolarization -> "Persuasion Polarization"
  | ODESystem.StickyDispersion -> "Sticky Dispersion"
  | ODESystem.CustomModel _ -> "Custom Model"
open Torch

type nonlinear_params = {
  rho: float;
  gamma: float;
  alpha1: float;
  alpha2: float;
  beta: float array;
  beta_r: float;
  nu: float;
  sobolev_s: float;
}

module SobolevSpace = struct
  type t = {
    data: Tensor.t;
    order: float;
    weight: Tensor.t;
  }

  let compute_weight tensor order =
    let n = (Tensor.shape tensor).(0) in
    let freqs = Tensor.arange ~start:0 ~end_:(float_of_int n) in
    Tensor.pow (Tensor.add freqs (Tensor.ones_like freqs)) order

  let create tensor order =
    {data = tensor; 
     order; 
     weight = compute_weight tensor order}

  let norm space =
    let weighted = Tensor.mul space.data space.weight in
    Tensor.norm weighted Norm_Float |> Tensor.float_value

  let project space cutoff =
    let mask = Tensor.le space.weight (Tensor.float_vec [|cutoff|]) in
    {space with 
     data = Tensor.where mask space.data (Tensor.zeros_like space.data)}
end

module FrequencyDecomposition = struct
  type localized_component = {
    frequency: float;
    scale: float;
    support: (float * float);
    data: Tensor.t
  }

  type spectral_analysis = {
    components: localized_component list;
    base_scale: float;
    total_mass: float;
    resonant_part: Tensor.t;
    nonresonant_part: Tensor.t;
  }

  let decompose_frequency space =
    let ft = Tensor.fft space.data ~normalized:true in
    let n = (Tensor.shape space.data).(0) in
    let scales = List.init (int_of_float (log2 (float_of_int n))) float_of_int in
    
    List.map (fun scale ->
      let lower = 2.0 ** scale in
      let upper = 2.0 ** (scale +. 1.0) in
      let mask = Tensor.logical_and 
        (Tensor.ge (Tensor.abs ft) (Tensor.float_vec [|lower|]))
        (Tensor.lt (Tensor.abs ft) (Tensor.float_vec [|upper|])) in
      let localized = Tensor.where mask ft (Tensor.zeros_like ft) in
      {
        frequency = (lower +. upper) /. 2.0;
        scale;
        support = (lower, upper);
        data = Tensor.ifft localized ~normalized:true;
      }
    ) scales

  let estimate_bilinear params comp1 comp2 =
    let scale_diff = abs_float (comp1.scale -. comp2.scale) in
    let interaction = Tensor.mul comp1.data comp2.data in
    let high_freq = max comp1.frequency comp2.frequency in
    high_freq ** (-. params.alpha1 *. params.rho) *.
    Tensor.norm interaction Norm_Float |> Tensor.float_value

  let decompose_paraproduct space =
    let components = decompose_frequency space in
    let partition_freq v = v.frequency > 1.0 /. params.gamma in
    let high_modes, low_modes = List.partition partition_freq components in
    let recombine comps =
      List.fold_left (fun acc comp ->
        Tensor.add acc comp.data
      ) (Tensor.zeros_like space.data) comps in
    {
      components;
      base_scale = log2 (float_of_int (Tensor.shape space.data).(0));
      total_mass = SobolevSpace.norm space;
      resonant_part = recombine low_modes;
      nonresonant_part = recombine high_modes;
    }

  let analyze_resonance params space =
    let spec = decompose_paraproduct space in
    let resonant_norm = Tensor.norm spec.resonant_part Norm_Float |> Tensor.float_value in
    let nonres_norm = Tensor.norm spec.nonresonant_part Norm_Float |> Tensor.float_value in
    (resonant_norm, nonres_norm)
end

module NonlinearOperator = struct
  type operator_component = [
    | `N0 of float
    | `R  of float
  ]

  type multilinear_form = {
    beta_j0: float;
    beta_j1: float;
    alpha1_rho: float;
    alpha2_rho: float;
  }

  let estimate_strongly_nonresonant params space inputs =
    let ft = Tensor.fft space.data ~normalized:true in
    let fts = List.map (fun s -> Tensor.fft s.data ~normalized:true) inputs in
    let n_star = 
      let norms = List.map (fun f -> 
        Tensor.norm f Norm_Float |> Tensor.float_value) (ft :: fts) in
      List.sort (fun a b -> compare b a) norms in
    
    let n0 = List.hd n_star in
    let n1 = List.nth n_star 1 in
    let n2 = List.nth n_star 2 in
    
    n0 ** (params.beta.(0) +. params.beta.(1) -. params.alpha1 *. params.rho) *.
    n1 ** (params.beta.(2) +. 0.5 -. params.alpha2 *. params.rho) *.
    n2 ** (List.fold_left (fun acc i -> 
      acc +. params.beta.(i) +. 0.5
    ) 0.0 (range 3 (List.length inputs)))

  let estimate_perturbed params space inputs =
    let (res, nonres) = FrequencyDecomposition.analyze_resonance params space in
    let nonres_bound = estimate_strongly_nonresonant params space inputs in
    if res > params.gamma then
      res ** params.beta_r *. nonres_bound
    else
      nonres_bound *. (1.0 +. params.gamma *. res)

  let apply params space =
    let ft = Tensor.fft space.data ~normalized:true in
    let n = (Tensor.shape space.data).(0) in
    let freqs = Tensor.arange ~start:0 ~end_:(float_of_int n) in
    
    let n0_part =
      let weighted = Tensor.pow (Tensor.abs ft) (params.rho -. 1.0) in
      Tensor.mul ft weighted in
    
    let r_part =
      let resonant_mask = Tensor.eq (Tensor.fmod freqs params.gamma) (Tensor.zeros_like freqs) in
      let resonant = Tensor.where resonant_mask ft (Tensor.zeros_like ft) in
      Tensor.mul resonant (Tensor.pow freqs params.beta_r) in
    
    [
      `N0 (Tensor.norm n0_part Norm_Float |> Tensor.float_value);
      `R  (Tensor.norm r_part Norm_Float |> Tensor.float_value)
    ]

  let multilinear_estimate params space inputs multi_form =
    let base_est = estimate_strongly_nonresonant params space inputs in
    let beta_contribution = 
      multi_form.beta_j0 +. multi_form.beta_j1 -. 
      multi_form.alpha1_rho *. params.rho -. 
      multi_form.alpha2_rho *. params.rho in
    base_est ** beta_contribution

  let analyze_operator params space =
    let components = apply params space in
    let estimates = List.map (function
      | `N0 v -> (v, estimate_strongly_nonresonant params space [space])
      | `R v  -> (v, v ** params.beta_r)
    ) components in
    
    let total_bound = List.fold_left (fun acc (v, est) -> 
      acc +. v *. est) 0.0 estimates in
    
    (components, total_bound)

  let evolution_estimate params space dt =
    let (comps, _) = analyze_operator params space in
    let time_weight = exp (-. params.gamma *. dt) in
    List.map (function
      | `N0 v -> `N0 (v *. time_weight)
      | `R v  -> `R  (v *. time_weight)
    ) comps
end

module TimeEvolution = struct
  type time_step = {
    dt: float;
    order: int;
    stages: int;
  }

  type evolution_mode = [
    | `NonResonant
    | `Resonant
    | `Mixed
  ]

  type evolution_scheme = {
    mode: evolution_mode;
    time_step: time_step;
    stability_factor: float;
  }

  type solution = {
    sobolev: SobolevSpace.t;
    time: float;
    dt: float;
  }

  let compute_timestep params space =
    let norm_s = SobolevSpace.norm space in
    let nonres_est = NonlinearOperator.estimate_strongly_nonresonant params space [space] in
    let (res_norm, _) = FrequencyDecomposition.analyze_resonance params space in
    let base_dt = 1.0 /. (params.gamma *. max norm_s nonres_est) in
    if res_norm > params.gamma then base_dt /. (1.0 +. res_norm)
    else base_dt

  let evolve_nonresonant params space dt =
    let ft = Tensor.fft space.data ~normalized:true in
    let n0_part = NonlinearOperator.apply params space in
    let evolved = List.fold_left (fun acc -> function
      | `N0 v -> Tensor.add acc (Tensor.mul_scalar ft v)
      | `R _  -> acc
    ) (Tensor.zeros_like ft) n0_part in
    let phase = Tensor.scalar_tensor (-. params.gamma *. dt) in
    let modulation = Tensor.exp phase in
    let modulated = Tensor.mul evolved modulation in
    SobolevSpace.create (Tensor.ifft modulated ~normalized:true) space.order

  let evolve_resonant params space dt =
    let ft = Tensor.fft space.data ~normalized:true in
    let n = (Tensor.shape space.data).(0) in
    let freqs = Tensor.arange ~start:0 ~end_:(float_of_int n) in
    let resonant_mask = Tensor.eq (Tensor.fmod freqs params.gamma) (Tensor.zeros_like freqs) in
    let resonant = Tensor.where resonant_mask ft (Tensor.zeros_like ft) in
    let phase = dt *. params.beta_r in
    let evolved = Tensor.mul resonant (Tensor.scalar_tensor (exp phase)) in
    SobolevSpace.create (Tensor.ifft evolved ~normalized:true) space.order

  let evolve_mixed params space dt =
    let nonres = evolve_nonresonant params space (dt /. 2.0) in
    let res = evolve_resonant params nonres dt in
    evolve_nonresonant params res (dt /. 2.0)

  let step params sol scheme =
    let next_space = match scheme.mode with
      | `NonResonant -> evolve_nonresonant params sol.sobolev scheme.time_step.dt
      | `Resonant -> evolve_resonant params sol.sobolev scheme.time_step.dt
      | `Mixed -> evolve_mixed params sol.sobolev scheme.time_step.dt in
    {sol with 
      sobolev = next_space;
      time = sol.time +. scheme.time_step.dt}

  let evolve_multistage params sol scheme final_time =
    let rec evolve current_sol =
      if current_sol.time >= final_time then current_sol
      else
        let dt = compute_timestep params current_sol.sobolev in
        let adjusted_scheme = {scheme with time_step = {scheme.time_step with dt}} in
        let stages = List.init scheme.time_step.stages (fun i ->
          let stage_dt = dt *. float_of_int i /. float_of_int scheme.time_step.stages in
          step params current_sol {adjusted_scheme with 
            time_step = {adjusted_scheme.time_step with dt = stage_dt}}
        ) in
        let combined = List.fold_left (fun acc stage ->
          {acc with sobolev = SobolevSpace.create 
            (Tensor.add acc.sobolev.data stage.sobolev.data) acc.sobolev.order}
        ) (List.hd stages) (List.tl stages) in
        let next_sol = {combined with
          sobolev = SobolevSpace.create 
            (Tensor.div_scalar combined.sobolev.data 
              (float_of_int scheme.time_step.stages)) combined.sobolev.order} in
        evolve next_sol in
    evolve sol

  let analyze_stability params sol scheme =
    let base_norm = SobolevSpace.norm sol.sobolev in
    let evolved = step params sol scheme in
    let evolved_norm = SobolevSpace.norm evolved.sobolev in
    evolved_norm /. base_norm < scheme.stability_factor
end

module Solver = struct
  type regularity_class = [
    | `Subcritical of float
    | `Critical of float
    | `Supercritical of float
  ]

  type existence_class = [
    | `Strong of float
    | `Weak of float
    | `Local of float
  ]

  type solution_properties = {
    regularity: regularity_class;
    existence: existence_class;
    persistence: bool;
    resonance_type: [`Strong | `Weak | `None];
  }

  let verify_strongly_nonresonant params space =
    let norm_s = SobolevSpace.norm space in
    let critical_index = params.sobolev_s /. 2.0 in
    let regularity = 
      if norm_s < critical_index then
        `Subcritical (critical_index -. norm_s)
      else if abs_float (norm_s -. critical_index) < 1e-10 then
        `Critical critical_index
      else 
        `Supercritical (norm_s -. critical_index) in
    
    let nonres_est = NonlinearOperator.estimate_strongly_nonresonant params space [space] in
    let existence =
      if nonres_est < params.gamma then
        `Strong (1.0 /. params.gamma)
      else if nonres_est < 1.0 then
        `Local (1.0 /. nonres_est)
      else
        `Weak (1.0 /. (nonres_est ** params.rho)) in
    
    let (res_norm, _) = FrequencyDecomposition.analyze_resonance params space in
    let resonance_type =
      if res_norm > params.gamma then `Strong
      else if res_norm > 0.0 then `Weak
      else `None in

    {
      regularity;
      existence;
      persistence = norm_s <= params.sobolev_s;
      resonance_type;
    }

  let verify_perturbed_existence params space =
    let base_props = verify_strongly_nonresonant params space in
    let spec = FrequencyDecomposition.decompose_paraproduct space in
    let existence = match base_props.existence with
      | `Strong t -> 
          let res_correction = spec.total_mass ** params.beta_r in
          `Strong (t /. (1.0 +. res_correction))
      | `Local t ->
          let nonres_correction = 
            Tensor.norm spec.nonresonant_part Norm_Float |> Tensor.float_value in
          `Local (t /. (1.0 +. nonres_correction))
      | `Weak t -> `Weak t in
    {base_props with existence}

  let estimate_energy params space =
    let ft = Tensor.fft space.data ~normalized:true in
    let n = (Tensor.shape space.data).(0) in
    let freqs = Tensor.arange ~start:0 ~end_:(float_of_int n) in
    
    let kinetic = 
      let weighted = Tensor.mul (Tensor.pow freqs 2.0) (Tensor.pow (Tensor.abs ft) 2.0) in
      0.5 *. (Tensor.sum weighted |> Tensor.float_value) in
    
    let potential =
      let nonlinear = NonlinearOperator.apply params space in
      List.fold_left (fun acc -> function
        | `N0 v -> acc +. v ** 2.0
        | `R v  -> acc +. v ** 2.0 *. params.gamma
      ) 0.0 nonlinear in
    
    (kinetic, potential)

  let verify_persistence params sol dt =
    let space = sol.sobolev in
    let props = verify_strongly_nonresonant params space in
    
    match props.regularity with
    | `Subcritical _ | `Critical _ ->
        let (k0, p0) = estimate_energy params space in
        let evolved = TimeEvolution.{
          sobolev = space;
          time = 0.0;
          dt
        } |> TimeEvolution.step params {mode = `Mixed; 
                                      time_step = {dt; order = 1; stages = 1};
                                      stability_factor = 1.0} in
        let (k1, p1) = estimate_energy params evolved.sobolev in
        abs_float (k1 +. p1 -. (k0 +. p0)) < params.gamma *. dt
    | `Supercritical _ -> false

  let verify_bounds params sol =
    let props = verify_perturbed_existence params sol.sobolev in
    match props.existence with
    | `Strong t | `Local t -> sol.time < t
    | `Weak _ -> false

  let estimate_lifespan params space =
    let props = verify_perturbed_existence params space in
    let base_time = match props.existence with
      | `Strong t | `Local t -> t
      | `Weak t -> t in
    
    match props.resonance_type with
    | `Strong -> base_time /. 2.0
    | `Weak -> base_time *. (1.0 -. params.gamma)
    | `None -> base_time
end
open Torch

type nonlinear_params = {
  rho: float;          (* ρ parameter *)
  gamma: float;        (* γ parameter *)
  alpha1: float;       (* α₁ parameter *)
  alpha2: float;       (* α₂ parameter *)
  beta: float array;   (* β parameters *)
  beta_r: float;       (* βᵣ resonant parameter *)
  nu: float;          (* ν frequency parameter *)
  sobolev_s: float;   (* Sobolev regularity s *)
}

module FrequencyDecomposition : sig
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

  val decompose_frequency : SobolevSpace.t -> localized_component list
  val estimate_bilinear : nonlinear_params -> localized_component -> localized_component -> float
  val decompose_paraproduct : SobolevSpace.t -> spectral_analysis
  val analyze_resonance : nonlinear_params -> SobolevSpace.t -> float * float
end

module SobolevSpace : sig
  type t = {
    data: Tensor.t;
    order: float;
    weight: Tensor.t;
  }

  val create : Tensor.t -> float -> t
  val norm : t -> float
  val project : t -> float -> t
end

module NonlinearOperator : sig
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

  val estimate_strongly_nonresonant : nonlinear_params -> SobolevSpace.t -> SobolevSpace.t list -> float
  val estimate_perturbed : nonlinear_params -> SobolevSpace.t -> SobolevSpace.t list -> float
  val apply : nonlinear_params -> SobolevSpace.t -> operator_component list
  val multilinear_estimate : nonlinear_params -> SobolevSpace.t -> SobolevSpace.t list -> multilinear_form -> float
  val analyze_operator : nonlinear_params -> SobolevSpace.t -> operator_component list * float
  val evolution_estimate : nonlinear_params -> SobolevSpace.t -> float -> operator_component list
end

module TimeEvolution : sig
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

  val compute_timestep : nonlinear_params -> SobolevSpace.t -> float
  val evolve_nonresonant : nonlinear_params -> SobolevSpace.t -> float -> SobolevSpace.t
  val evolve_resonant : nonlinear_params -> SobolevSpace.t -> float -> SobolevSpace.t
  val evolve_mixed : nonlinear_params -> SobolevSpace.t -> float -> SobolevSpace.t
  val step : nonlinear_params -> solution -> evolution_scheme -> solution
  val evolve_multistage : nonlinear_params -> solution -> evolution_scheme -> float -> solution
  val analyze_stability : nonlinear_params -> solution -> evolution_scheme -> bool
end

module Solver : sig
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

  val verify_strongly_nonresonant : nonlinear_params -> SobolevSpace.t -> solution_properties
  val verify_perturbed_existence : nonlinear_params -> SobolevSpace.t -> solution_properties
  val estimate_energy : nonlinear_params -> SobolevSpace.t -> float * float
  val verify_persistence : nonlinear_params -> TimeEvolution.solution -> float -> bool
  val verify_bounds : nonlinear_params -> TimeEvolution.solution -> bool
  val estimate_lifespan : nonlinear_params -> SobolevSpace.t -> float
end
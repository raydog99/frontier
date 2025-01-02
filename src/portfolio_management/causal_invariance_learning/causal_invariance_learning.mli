open Torch

type environment = {
  id: int;                  (* Unique environment identifier *)
  covariates: Tensor.t;     (* Input features *)
  outcomes: Tensor.t;       (* Observed outcomes *)
  sample_size: int;         (* Number of samples *)
  intervention_strength: float; (* Strength of potential interventions *)
  metadata: (string * string) list; (* Flexible metadata *)
}

type intervention = 
  | NoIntervention
  | DirectIntervention of {
      target_variable: int;
      strength: float;
      distribution: [`Gaussian | `Uniform | `Custom]
    }
  | IndirectIntervention of {
      mediating_variables: int list;
      effect_size: float
    }

type model_parameters = {
  causal_coefficients: Tensor.t;  (* Causal effect coefficients *)
  noise_variance: float;           (* Model noise variance *)
  complexity_penalty: float;       (* Regularization strength *)
}

type causal_graph = {
  adjacency_matrix: Tensor.t;  (* Directed graph representation *)
  variable_names: string array;
  has_cycles: bool;
}

type statistical_summary = {
  mean: Tensor.t;
  variance: Tensor.t;
  skewness: Tensor.t;
  kurtosis: Tensor.t;
}

type convergence_summary = {
  average_loss: float;
  loss_variance: float;
  num_iterations: int;
  converged: bool;
} option

type intervention_sensitivity = {
  intervention: intervention;
  total_impact: float;
  max_impact: float;
  min_impact: float;
}

type model_diagnosis = {
  model_parameters: model_parameters;
  coefficient_stability: statistical_summary list;
  convergence_analysis: convergence_summary;
  sensitivity_analysis: intervention_sensitivity list;
  causal_graph: causal_graph;
}

type causal_hypothesis = {
  source_variable: int;
  target_variable: int;
  correlation: float;
  p_value: float;
  causal_strength: float;
  significant: bool;
}

type nonconvex_dro_model = {
  parameters: model_parameters;
  environments: environment list;
  causal_graph: causal_graph;
  training_history: float list;
}

type training_config = {
  max_iterations: int;
  learning_rate: float;
  regularization_strength: float;
  early_stopping_tolerance: float;
  intervention_strategy: intervention;
}

type experiment_results = {
  initial_environments: environment list;
  trained_model: nonconvex_dro_model;
  causal_graph: causal_graph;
  model_diagnosis: model_diagnosis;
  causal_hypotheses: causal_hypothesis list;
}

val compute_moments : Tensor.t -> statistical_summary
val detect_distribution_shift : 
  environment -> environment -> {
    environment_1_id: int;
    environment_2_id: int;
    kl_divergence: float;
    is_significant_shift: bool;
  }
val partial_correlations : Tensor.t -> Tensor.t
  
val create_model : 
  environment list -> 
  ?initial_graph:causal_graph -> 
  unit -> 
  nonconvex_dro_model

val train_model : 
  nonconvex_dro_model -> 
  training_config -> 
  nonconvex_dro_model

val infer_causal_graph : 
  environment list -> 
  causal_graph

val analyze_intervention : 
  nonconvex_dro_model -> 
  intervention -> 
  (float * Tensor.t) list

val predict_outcome : 
  nonconvex_dro_model -> 
  environment -> 
  Tensor.t

val test_causal_hypothesis : 
  nonconvex_dro_model -> 
  unit -> 
  causal_hypothesis list

val diagnose_model : 
  nonconvex_dro_model -> 
  model_diagnosis

module Runner = sig
  type simulation_config = {
    num_environments: int;
    sample_size: int;
    input_dimension: int;
    noise_level: float;
    intervention_probability: float;
  }
  
  val generate_synthetic_environments : 
    simulation_config -> 
    environment list
  
  val run_experiment : 
    simulation_config -> 
    experiment_results
  
  val export_results : 
    experiment_results -> 
    string
  
  val default_experiment : 
    unit -> 
    experiment_results
end
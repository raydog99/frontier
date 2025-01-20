open Torch

(** Feature space representation *)
type feature_space = {
  structured: Tensor.t;   (** s ∈ M *)
  unstructured: Tensor.t; (** x ∈ X ⊆ Rp *)
  dim_structured: int;
  dim_unstructured: int;
}

(** Model variant types for alpha parameters *)
type model_variant = 
  | S2BAMDT of float  (** Single alpha parameter *)
  | SkBAMDT of float array (** Multiple fixed alphas *)

(** Model parameters configuration *)
type model_params = {
  num_trees: int;
  gamma: float;        (** Tree split probability control *)
  delta: float;        (** Tree depth penalty *)
  alpha_mu: float;     (** Inverse-gamma shape for leaf params *)
  beta_mu: float;      (** Inverse-gamma scale for leaf params *)
  sigma_sq: float;     (** Residual variance *)
  sigma_mu_sq: float;  (** Leaf parameter variance *)
  variant: model_variant;
  alpha_g: float;      (** Gamma shape for alpha prior *)
  beta_g: float;       (** Gamma rate for alpha prior *)
  nu: float;           (** Degrees of freedom for sigma_sq prior *)
  lambda: float;       (** Scale parameter for sigma_sq prior *)
  k_prior_counts: float array; (** Prior counts for Sk-BAMDT *)
}

(** Split type definitions *)
type split_type = 
  | UnivariateRule of {
      feature_idx: int;
      threshold: float;
    }
  | MultivariateRule of {
      reference_points: Tensor.t;
      left_knots: Tensor.t;
      right_knots: Tensor.t;
    }

(** Decision rule with probabilistic splitting *)
type decision_rule = {
  split: split_type;
  prob_type: [`Hard | `Soft of float]; (** alpha parameter *)
  normalization_const: float;
}

(** Tree node type *)
type tree_node =
  | Terminal of {
      indices: int array;
      parameter: float option;
    }
  | Internal of {
      left: tree;
      right: tree;
      rule: decision_rule;
      indices: int array;
    }
and tree = {
  node: tree_node;
  depth: int;
}

(** Full model structure *)
type model = {
  trees: tree list;
  params: model_params;
  feature_space: feature_space;
  observed_data: Tensor.t option;
}

(** Random number generation utilities *)
module Random : sig
  val gaussian : unit -> float
  val gamma : float -> float -> float
  val inverse_gamma : float -> float -> float
  val dirichlet : float array -> float array
end

(** Tensor operation utilities *)
module TensorOps : sig
  val gather : Tensor.t -> int array -> Tensor.t
  val compute_distances : Tensor.t -> Tensor.t -> Tensor.t
end

(** Laplacian-based embedding operations *)
module LaplacianOps : sig
  val compute_similarity_matrix : Tensor.t -> float -> Tensor.t
  val compute_normalized_laplacian : Tensor.t -> Tensor.t
  val compute_embedding : Tensor.t -> int -> float -> Tensor.t
  val manifold_distance : Tensor.t -> float -> Tensor.t
end

(** Reference point selection and management *)
module ReferencePoints : sig
  val select_kmeans_refs : Tensor.t -> int -> int -> Tensor.t
  val bipartition_refs : Tensor.t -> Tensor.t -> 'a * 'b
end

(** Decision rule creation and probability *)
module DecisionRules : sig
  type split_prob = {
    left_prob: float;
    right_prob: float;
  }

  val create_univariate_rule : feature_space -> int array -> split_type
  val create_multivariate_rule : feature_space -> int array -> float -> split_type
  val create_decision_rule : model_params -> feature_space -> int array -> decision_rule
  val compute_split_probability : decision_rule -> Tensor.t -> float
  val update_normalization_constant : decision_rule -> Tensor.t -> decision_rule
end

(** Tree node operations *)
module TreeOps : sig
  val create_tree : int -> int -> model_params -> feature_space -> int array -> tree
  val find_leaf_paths : tree -> int list list
  val find_leaf_parent_paths : tree -> int list list
  val find_internal_paths : tree -> int list list
  val get_node_at_path : tree -> int list -> tree_node option
  val update_node_at_path : tree -> int list -> tree_node -> tree
  val grow_at_path : tree -> int list -> model_params -> tree
  val prune_at_path : tree -> int list -> tree
  val change_rule_at_path : tree -> int list -> decision_rule -> tree
  val predict_tree : tree -> Tensor.t -> Tensor.t
end

(** Markov Chain Monte Carlo (MCMC) sampling *)
module MCMC : sig
  type proposal = 
    | Grow of {depth: int; target_node: int list}
    | Prune of {target_node: int list}
    | Change of {node_path: int list; new_rule: decision_rule}

  val compute_likelihood : tree -> Tensor.t -> model_params -> float
  val compute_tree_prior : tree -> model_params -> float
  val sample_proposal : tree -> proposal
  val metropolis_hastings_step : tree -> Tensor.t -> model_params -> tree
end

(** Parameter update operations *)
module ParameterUpdates : sig
  val update_sigma_sq : model -> Tensor.t -> float
  val collect_leaf_parameters : tree list -> float list
  val update_sigma_mu_sq : model -> float
  val update_alpha : model -> Tensor.t -> float array
  val update_leaf_parameters : tree -> Tensor.t -> model_params -> tree
  val compute_soft_likelihood : tree -> Tensor.t -> float
  val find_alpha_index : float -> float array -> int
end
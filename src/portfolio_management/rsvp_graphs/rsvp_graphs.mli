open Torch

type data_matrix = Tensor.t
type covariance = Tensor.t

(** Configuration for estimation algorithms *)
type config = {
  tol: float;  (** Convergence tolerance *)
  max_iter: int;  (** Maximum number of iterations *)
  min_eigenval: float;  (** Minimum eigenvalue threshold *)
  regularization: float;  (** Regularization parameter *)
}

(** Default configuration values *)
val default_config : config

(** Result type for estimation procedures *)
type estimation_result = {
  estimate: Tensor.t;  (** Estimated matrix *)
  iterations: int;  (** Number of iterations performed *)
  error: float;  (** Final error value *)
  converged: bool;  (** Whether the algorithm converged *)
}

(** Result type for rank estimation *)
type rank_result = {
  rank: int;  (** Estimated rank *)
  eigenvalues: Tensor.t;  (** Eigenvalues *)
  explained_variance: float array;  (** Explained variance ratios *)
}

(** Center a data matrix by subtracting column means *)
val center_data : data_matrix -> data_matrix

(** Compute sample covariance matrix *)
val sample_covariance : data_matrix -> covariance

(** Compute stable SVD with regularization *)
val stable_svd : data_matrix -> config -> (Tensor.t * Tensor.t * Tensor.t) option

(** Basic RSVP estimation *)
val rsvp : data_matrix -> covariance

(** Subsampling RSVP estimation *)
val rsvp_subsample : data_matrix -> m:int -> b:int -> covariance

(** Sample splitting RSVP estimation *)
val rsvp_split : data_matrix -> m:int -> covariance

(** Clear first ell eigenvalues *)
val clear_prefix_eigenvals : int -> Tensor.t -> Tensor.t

(** Transform spectrum using different methods *)
val transform_spectrum : Tensor.t -> 
  [`PCA of int | `Threshold of float | `Shrinkage of float] -> Tensor.t

(** PC removal estimation *)
val pc_removal : Tensor.t -> num_components:int -> Tensor.t

(** Nodewise regression for sparse precision matrix estimation *)
val nodewise_regression : Tensor.t -> int -> float -> Tensor.t

(** Estimate conditional independence graph *)
val estimate_cig : Tensor.t -> float -> Tensor.t

(** Nodewise regression with inference *)
val nodewise_regression_inference : Tensor.t -> int -> float -> Tensor.t * Tensor.t

(** Compute partial correlation *)
val partial_correlation : Tensor.t -> int -> int -> int list -> float

(** Build conditional independence graph *)
val build_conditional_graph : Tensor.t -> float -> Tensor.t * (int * int * int list) list

(** Identify v-structures in graph *)
val identify_v_structures : Tensor.t -> (int * int * int list) list -> Tensor.t

(** Orient edges in CPDAG *)
val orient_edges : Tensor.t -> Tensor.t

(** Get descendants of a node *)
val get_descendants : Tensor.t -> int -> int list

(** Compute path coefficient between nodes *)
val compute_path_coefficient : Tensor.t -> Tensor.t -> int -> int -> float

(** Find all directed paths between nodes *)
val find_directed_paths : Tensor.t -> int -> int -> int list list
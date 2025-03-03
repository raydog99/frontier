open Torch

(* Convert vector to diagonal matrix *)
val diag : Tensor.t -> Tensor.t

(* Log-sum-exp trick for numerical stability *)
val log_sum_exp : Tensor.t -> Tensor.t

(* Create a block diagonal matrix from a list of matrices *)
val block_diag : Tensor.t list -> Tensor.t

(* Extract half-vectorization (vech) of a symmetric matrix *)
val vech : Tensor.t -> Tensor.t

(* Inverse of vech: convert half-vectorization to symmetric matrix *)
val vech_to_matrix : Tensor.t -> Tensor.t

(* Vectorize a matrix (column-wise) *)
val vec : Tensor.t -> Tensor.t

(* Reshape a vector to a matrix *)
val vec_to_matrix : Tensor.t -> int -> int -> Tensor.t

(* Compute Cholesky decomposition for positive definite matrix *)
val cholesky : Tensor.t -> Tensor.t

(* Compute log determinant of a matrix from its Cholesky factor *)
val log_det_from_cholesky : Tensor.t -> Tensor.t

(* Convert precision matrix to covariance matrix *)
val precision_to_covariance : Tensor.t -> Tensor.t

(* Calculate KL divergence between two multivariate normal distributions *)
val kl_mvn_from_params : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

(* Create a mask for sparse Cholesky factor *)
val create_sparse_cholesky_mask : int -> int -> Tensor.t

(* Create a mask for state space model Cholesky factor *)
val create_state_space_cholesky_mask : int -> int -> Tensor.t

(* Apply sparsity mask to a matrix *)
val apply_mask : Tensor.t -> Tensor.t -> Tensor.t

(* ADAM optimizer parameters *)
type adam_params = {
  beta1: float;
  beta2: float;
  epsilon: float;
  learning_rate: float;
}

(* Default ADAM parameters *)
val default_adam_params : adam_params

(* ADAM optimizer state *)
type adam_state = {
  m: Tensor.t;  (* First moment *)
  v: Tensor.t;  (* Second moment *)
  t: int;       (* Timestep *)
}

(* Initialize ADAM state *)
val init_adam_state : int list -> adam_state

(* ADAM update step *)
val adam_update : adam_params -> adam_state -> Tensor.t -> adam_state * Tensor.t

(* Convergence assessment *)
val assess_convergence : float -> float -> float -> bool

(* Calculate RMSE between tensors *)
val rmse : Tensor.t -> Tensor.t -> float

(* Generate reparameterized sample from Gaussian component *)
val reparameterize_gaussian : Tensor.t -> Tensor.t -> Tensor.t

(* Reparameterization-based gradient for mean *)
val reparam_grad_mean : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> int -> Tensor.t

(* Reparameterization-based gradient for Cholesky factor *)
val reparam_grad_cholesky : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> int -> Tensor.t

(* Control variates for variance reduction in gradient estimation *)
val control_variate_gradient : Tensor.t list -> (Tensor.t -> float) -> Tensor.t

(* Variance reduction technique from Ranganath et al. (2014) *)
val ranganath_control_variate : 
  Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> int -> Tensor.t

(* Combine multiple variance reduction techniques *)
val combined_variance_reduction : 
  Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> int -> Tensor.t * Tensor.t

(* Initialization strategies for mean vectors *)
type mean_init_strategy = [
  | `Random
  | `KMeans of Tensor.t * int
  | `Factor
]

(* Initialization strategies for Cholesky factors *)
type chol_init_strategy = [
  | `Identity
  | `Perturb
  | `Diagonal of float
  | `StateSpace
]

(* Initialize mean vector for a new component *)
val initialize_mean : 
  base_mean:Tensor.t -> 
  n_latent:int -> 
  n_global:int -> 
  strategy:[< `Factor | `KMeans of Tensor.t * int | `Random | `StateSpace ] -> 
  scale:float -> 
  Tensor.t

(* Initialize Cholesky factor for a new component *)
val initialize_cholesky : 
  base_chol:Tensor.t -> 
  n_latent:int -> 
  n_global:int -> 
  strategy:[< `Diagonal of float | `Identity | `Perturb | `StateSpace ] -> 
  scale:float -> 
  Tensor.t

(* Gaussian component parameters for variational approximation *)
type gaussian_component = {
  mu: Tensor.t;           (* Mean vector *)
  l_chol: Tensor.t;       (* Cholesky factor of precision matrix *)
  mu_g: Tensor.t;         (* Global parameters mean *)
  mu_b: Tensor.t;         (* Local parameters (latent variables) mean *)
  l_g: Tensor.t;          (* Cholesky factor for global parameters *)
  l_b: Tensor.t list;     (* Cholesky factors for each latent variable block *)
  l_gb: Tensor.t list;    (* Cross-covariance terms *)
  l_tilde: Tensor.t list option; (* State transition matrices for state space models (optional) *)
}

(* Gaussian mixture distribution *)
type gaussian_mixture = {
  components: gaussian_component array;
  weights: Tensor.t;       (* Mixture weights *)
  n_components: int;
  n_global: int;           (* Number of global parameters *)
  n_latent: int;           (* Number of latent variables *)
  n_subjects: int;         (* Number of subjects/data points *)
  latent_dim: int;         (* Dimension of each latent variable *)
}

(* Initialize a Gaussian component with the proper structure *)
val init_gaussian_component : 
  n_global:int -> 
  n_subjects:int -> 
  latent_dim:int ->
  use_sparse_cholesky:bool ->
  gaussian_component

(* Initialize a state space Gaussian component *)
val init_state_space_component :
  n_time_points:int ->
  latent_dim:int ->
  global_dim:int ->
  use_sparse_cholesky:bool ->
  gaussian_component

(* Initialize a Gaussian mixture distribution *)
val init_gaussian_mixture :
  n_components:int ->
  n_global:int ->
  n_subjects:int ->
  latent_dim:int ->
  use_sparse_cholesky:bool ->
  gaussian_mixture

(* Initialize a state space mixture distribution *)
val init_state_space_mixture :
  n_components:int ->
  n_time_points:int ->
  latent_dim:int ->
  global_dim:int ->
  use_sparse_cholesky:bool ->
  gaussian_mixture

(* Component initialization strategies *)
type component_init_strategy = [
  | `Random
  | `KMeans of Tensor.t
  | `Factor
  | `StateSpace
]

(* Initialize a new mixture component with various strategies *)
val initialize_component :
  base_component:gaussian_component ->
  n_latent:int ->
  n_global:int ->
  strategy:[< `Factor | `KMeans of Tensor.t | `Random | `StateSpace ] ->
  gaussian_component

(* Calculate log-likelihood of a sample under a Gaussian component *)
val log_prob_gaussian_component : gaussian_component -> Tensor.t -> Tensor.t

(* Calculate log-likelihood of a sample under the Gaussian mixture *)
val log_prob_gaussian_mixture : gaussian_mixture -> Tensor.t -> Tensor.t

(* Get conditional distribution for latent variables given global parameters *)
val conditional_latent_given_global : 
  gaussian_mixture -> Tensor.t -> (Tensor.t * Tensor.t list * Tensor.t list) array

(* Get marginal distribution for global parameters *)
val marginal_global_distribution : 
  gaussian_mixture -> (Tensor.t * Tensor.t * Tensor.t) array

(* Get conditional distribution for state n given state n+1 and global params *)
val conditional_state_given_next_and_global : 
  gaussian_component -> int -> Tensor.t -> Tensor.t * Tensor.t

(* Sample from the Gaussian mixture *)
val sample_gaussian_mixture : 
  gaussian_mixture -> n_samples:int -> Tensor.t

(* Calculate ELBO (Evidence Lower Bound) for variational inference *)
val calculate_elbo : 
  gaussian_mixture -> (Tensor.t -> Tensor.t) -> Tensor.t
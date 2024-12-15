open Torch

type precision_matrix = {
  matrix: Tensor.t;
  dim: int;
  sparsity: float;
}

type design_matrix = {
  matrix: Tensor.t;
  n_obs: int;
  n_params: int;
}

type glmm_params = {
  tau: float;
  theta: Tensor.t;
  prior_mean: Tensor.t;
  prior_precision: Tensor.t;
}

type model_config = {
  n_factors: int;
  factor_sizes: int array;
  fixed_effects: int;
}

type observation = {
  response: float;
  factor_levels: int array;
}

val get_block : Tensor.t -> row_start:int -> row_end:int -> 
               col_start:int -> col_end:int -> Tensor.t

val normalize_adjacency : Tensor.t -> Tensor.t

val sparse_multiply : Tensor.t -> Tensor.t -> Tensor.t

val compute_sparsity : Tensor.t -> float

val factorize : precision_matrix -> precision_matrix option

val column_wise_cholesky : precision_matrix -> 
                          precision_matrix option

val solve : l:Tensor.t -> b:Tensor.t -> Tensor.t

val estimate_cost : precision_matrix -> (float * float) option

module ConjugateGradient : sig
  type convergence_info = {
    iterations: int;
    errors: float array;
    bound_ratio: float;
  }

  val solve : q:precision_matrix -> b:Tensor.t -> 
             ?max_iter:int -> ?tol:float -> 
             Tensor.t * convergence_info
  
  val sample : q:precision_matrix -> mean:Tensor.t -> Tensor.t
end

module RandomIntercept : sig
  type model = {
    config: model_config;
    observations: observation array;
    design: design_matrix;
    params: glmm_params;
    tau_response: float;
    prior_taus: float array;
  }

  val create : model_config -> int -> float -> model
  
  val compute_posterior_precision : model -> precision_matrix
  
  val sample_posterior : model -> observation array -> Tensor.t
  
  val update_hyperparameters : model -> observation array -> 
                              float array * float
end

module SpectralAnalysis : sig
  type eigenvalue_distribution = {
    bulk: float array;
    outliers: float array;
    condition_number: float;
  }

  type spectral_gap = {
    value: float;
    location: int * int;
    relative_size: float;
  }

  val compute_eigenvalues : precision_matrix -> Tensor.t
  
  val analyze_spectrum : precision_matrix -> 
                        model_config -> 
                        eigenvalue_distribution
  
  val effective_condition_number : eigenvalue_distribution -> 
                                 s:int -> r:int -> float
  
  val find_spectral_gaps : float array -> float -> spectral_gap list
end

module GraphAnalysis : sig
  type graph = {
    vertices: int;
    edges: (int * int) list;
    adjacency: Tensor.t;
  }

  val create_from_precision : precision_matrix -> graph
  
  val is_conditionally_independent : graph -> int -> int -> 
                                   int list -> bool
  
  val find_potential_fillins : graph -> int array -> 
                              (int * int) list
  
  val analyze_connectivity : graph -> model_config -> 
                           int array * float
  
  val reduce_bandwidth : graph -> int array
end
open Torch

val unit_frechet_cdf : Tensor.t -> Tensor.t
val sample_unit_frechet : n:int -> Tensor.t
val wasserstein_distance : Tensor.t -> Tensor.t -> Tensor.t
val compute_spectral_measure : Tensor.t -> Tensor.t
val simulate_poisson_process : intensity:float -> n_points:int -> Tensor.t
val robustify_cdf : baseline_dist:Tensor.t -> x:Tensor.t -> epsilon:float -> float
val robustify_rare_set_prob : baseline_dist:Tensor.t -> set_a:Tensor.t -> epsilon:float -> float
val robustify_cvar : baseline_dist:Tensor.t -> alpha:float -> epsilon:float -> float
val primal_problem : loss_fn:(Tensor.t -> Tensor.t) -> baseline_dist:Tensor.t -> epsilon:float -> Tensor.t
val dual_problem : loss_fn:(Tensor.t -> Tensor.t) -> baseline_dist:Tensor.t -> epsilon:float -> Tensor.t
val generate_synthetic_data : n:int -> d:int -> Tensor.t
val tensor_to_float_list : Tensor.t -> float list
val float_list_to_tensor : float list -> Tensor.t
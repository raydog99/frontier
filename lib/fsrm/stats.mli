open Torch

val acf : Tensor.t -> int -> Tensor.t
val shannon_entropy : Tensor.t -> float
val conditional_entropy : Tensor.t -> Tensor.t -> float
val joint_probability : Tensor.t -> Tensor.t -> Tensor.t
val conditional_probability : Tensor.t -> Tensor.t -> Tensor.t
val normal_cdf : float -> float
val jarque_bera_test : Tensor.t -> float * float
val ljung_box_test : Tensor.t -> int -> float * float
val kolmogorov_smirnov_test : Tensor.t -> (float -> float) -> float * float
val akaike_information_criterion : float -> int -> float
val bayesian_information_criterion : float -> int -> int -> float
val quantile : Tensor.t -> float -> float
val chi2_cdf : float -> int -> float
val ks_test_p_value : float -> float -> float
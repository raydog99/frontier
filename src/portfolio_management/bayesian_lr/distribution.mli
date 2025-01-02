val normal_cdf : float -> float -> float -> float
(** [normal_cdf x mu sigma] computes standard normal CDF *)

val normal_ppf : float -> float -> float -> float
(** [normal_ppf p mu sigma] computes normal percent point function (inverse CDF) *)

val chi_square_cdf : float -> float -> float
(** [chi_square_cdf x df] computes chi-square CDF with df degrees of freedom *)
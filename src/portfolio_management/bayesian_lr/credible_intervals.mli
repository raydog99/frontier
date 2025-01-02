val compute_component_intervals : 
  Type.posterior_sample list -> int -> float -> Type.credible_interval
(** [compute_component_intervals samples j alpha] computes component-wise credible intervals *)

val compute_credible_ellipsoid :
  Type.posterior_sample list -> int list -> float -> Type.credible_ellipsoid
(** [compute_credible_ellipsoid samples selected_vars alpha] computes post-selection credible ellipsoid *)
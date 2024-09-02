val normal_quantile : float -> float
val confidence_interval : float -> float -> int -> float -> float * float
val t_test : float list -> float list -> float * int
val wilcoxon_test : float list -> float list -> float
val friedman_test : float list list -> float * int
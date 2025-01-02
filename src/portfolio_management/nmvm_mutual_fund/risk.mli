val value_at_risk : Nmvm.t -> Portfolio.t -> float -> int -> float
val conditional_value_at_risk : Nmvm.t -> Portfolio.t -> float -> int -> float
val expected_shortfall : Nmvm.t -> Portfolio.t -> float -> int -> float
val stress_test : Nmvm.t -> Portfolio.t -> (Torch.Tensor.t * float * float) list -> float list
val portfolio_beta : Nmvm.t -> Portfolio.t -> Portfolio.t -> float
val tracking_error : Nmvm.t -> Portfolio.t -> Portfolio.t -> int -> float
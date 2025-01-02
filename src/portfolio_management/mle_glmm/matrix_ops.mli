open Torch

val safe_solve : Tensor.t -> Tensor.t -> Tensor.t
(** [safe_solve a b] solves the system Ax = b with numerical stability safeguards *)

val safe_cholesky : Tensor.t -> Tensor.t
(** [safe_cholesky mat] computes Cholesky decomposition with jitter for stability *)

val bessel_k : Tensor.t -> Tensor.t -> Tensor.t
(** [bessel_k nu x] computes modified Bessel function of the second kind *)

val log_det : Tensor.t -> Tensor.t
(** [log_det mat] computes log determinant of matrix *)

val moore_penrose_inverse : Tensor.t -> Tensor.t
(** [moore_penrose_inverse mat] computes the Moore-Penrose pseudoinverse *)

val standard_errors : Tensor.t -> Tensor.t
(** [standard_errors info] computes standard errors from Fisher Information matrix *)
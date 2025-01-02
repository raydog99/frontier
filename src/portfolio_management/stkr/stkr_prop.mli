open Torch

module SparseOps : sig
    type csr_matrix = {
      values: Tensor.t;
      row_ptr: int array;
      col_idx: int array;
    }

    val to_csr : Types.sparse_matrix -> csr_matrix
    val csr_matmul : csr_matrix -> Tensor.t -> Tensor.t
end

val propagate_simple_sparse : Types.kernel -> Types.polynomial_params -> 
    Types.stkr_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val propagate_inverse_sparse : Types.kernel -> Types.stkr_params -> 
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
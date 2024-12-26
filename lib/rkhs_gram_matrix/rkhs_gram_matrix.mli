open Torch

type tensor = Tensor.t            (* Tensor type from Torch *)
type scalar = float               (* Scalar values *)
type kernel_function = tensor -> tensor -> tensor  (* Kernel function type *)
type feature_map = tensor -> tensor                (* Feature map type *)

(* Core vector space operations *)
val linear_combination : tensor list -> float list -> tensor
  (* Compute linear combinations of tensors with given coefficients *)

val check_axioms : tensor -> tensor list -> bool
  (* Verify vector space axioms hold for given tensors *)

(* Finite-dimensional RKHS *)
module FiniteRKHS : sig
  type finite_rkhs = {
    dim: int;                           (* Dimension of space *)
    basis: tensor list;                 (* Orthonormal basis *)
    kernel: kernel_function;            (* Associated kernel *)
    completion: bool;                   (* Completeness property *)
  }
  
  (* Create RKHS from basis and kernel *)
  val create : tensor list -> kernel_function -> finite_rkhs
  
  (* Compute optimal approximation in RKHS *)
  val optimal_approximation : finite_rkhs -> tensor list -> (tensor -> tensor)
end

(* Kernel functions *)
module Kernel : sig
  type t = kernel_function
  
  val gaussian : float -> kernel_function    (* Gaussian kernel with given sigma *)
  val hardy : float -> kernel_function      (* Hardy kernel with given sigma *)
  val gram : kernel_function -> tensor -> tensor  (* Compute Gram matrix *)
  val is_positive_definite : kernel_function -> tensor -> bool  (* Check positive definiteness *)
end

(* Radial basis functions *)
module RBF : sig
  type t = {
    func: tensor -> tensor -> float -> tensor;  (* RBF function *)
    params: float list;                         (* Parameters *)
  }
  
  val gaussian : t  (* Gaussian RBF *)
  val hardy : t     (* Hardy RBF *)
  val is_even_polynomial : (tensor -> tensor -> float -> tensor) -> bool 
    (* Check if function is even polynomial *)
end

(* Reproducing kernel verification *)
module RKVerification : sig
  type kernel_properties = {
    symmetric: bool;           (* Symmetry property *)
    positive_definite: bool;   (* Positive definiteness *)
    continuous: bool;          (* Continuity *)
    reproducing: bool;         (* Reproducing property *)
    universal: bool;           (* Universal approximation *)
  }
  
  module FeatureSpace : sig
    type t = {
      dim: int;                           (* Feature space dimension *)
      map: feature_map;                   (* Feature mapping *)
      inner_product: tensor -> tensor -> float;  (* Inner product *)
    }
    
    val from_kernel : kernel_function -> tensor -> t  (* Construct from kernel *)
  end
  
  val verify_kernel : kernel_function -> tensor -> kernel_properties
    (* Verify all kernel properties *)
end

(* Product spaces *)
module ProductSpace : sig
  type ('a, 'b) product_space = {
    space1: 'a;
    space2: 'b;
    inner_product: tensor -> tensor -> tensor -> tensor -> float;
    norm: tensor -> tensor -> float;
  }
  
  val create : 'a -> 'b -> ('a, 'b) product_space
  val verify_properties : ('a, 'b) product_space -> tensor -> tensor -> bool
end

(* Product RKHS *)
module ProductRKHS : sig
  type t = {
    kernel_u: kernel_function;  (* Input kernel *)
    kernel_x: kernel_function;  (* State kernel *)
    sigma_u: float;            (* Input kernel parameter *)
    sigma_x: float;            (* State kernel parameter *)
  }

  val create : sigma_u:float -> sigma_x:float -> t
  val gram_matrix : t -> input_data:tensor -> state_data:tensor -> tensor
  val kernel_vector : t -> input_data:tensor -> state_data:tensor -> 
                     input:tensor -> state:tensor -> tensor
  val learn : t -> input_data:tensor -> state_data:tensor -> 
             output_data:tensor -> input:tensor -> state:tensor -> tensor
end
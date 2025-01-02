open Torch

type feature_vector = Tensor.t
type sample_vector = Tensor.t
type diffusion_operator = Tensor.t
type hyperbolic_point = Tensor.t

type tree = {
  nodes: int array;
  edges: (int * int) array;
  weights: float array;
  parents: int array;
  subtree_leaves: int list array;
}

val embed_features : 
  feature_vector array -> diffusion_operator -> int -> hyperbolic_point array
val compute_hd_lca : 
  hyperbolic_point -> hyperbolic_point -> hyperbolic_point
val construct_binary_tree : 
  hyperbolic_point array -> tree
val compute_twd : 
  ?use_sliced:bool -> sample_vector -> sample_vector -> tree -> float
val compute_twd_gpu : 
  ?use_sliced:bool -> sample_vector -> sample_vector -> tree -> Config.t -> float
val compute_scale_embedding_sparse : 
  feature_vector array -> Sparse_ops.sparse_tensor -> int -> hyperbolic_point array
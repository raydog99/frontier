open Torch

val construct_factors : ?tucker_rank:(int * int) -> ?cp_rank:int -> ?verbose:bool -> Tensor.t -> ((Tucker.t * CP.t), string) result
val evaluate_factors : Tensor.t -> Tucker.t -> CP.t -> (float * float, string) result
val cross_validate : ?n_folds:int -> ?tucker_ranks:(int * int) list -> ?cp_ranks:int list -> ?verbose:bool -> Tensor.t -> ((int * int) * int * float * float, string) result
val compare_models : Tensor.t -> Tucker.t -> CP.t -> (string, string) result

val center_data : Tensor.t -> Tensor.t
val scale_data : Tensor.t -> Tensor.t
val preprocess_data : Tensor.t -> Tensor.t
val split_train_test : Tensor.t -> float -> Tensor.t * Tensor.t
val explained_variance_ratio : Tensor.t -> Tensor.t -> float
val pca : Tensor.t -> int -> Tensor.t
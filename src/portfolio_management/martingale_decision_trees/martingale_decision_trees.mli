open Torch

(* Utility functions for tensor operations *)
val variance : Tensor.t -> Tensor.t
val conditional_expectation : Tensor.t -> Tensor.t -> Tensor.t
val conditional_variance : Tensor.t -> Tensor.t -> Tensor.t
val l1_deviation : Tensor.t -> Tensor.t
val mask_lt : Tensor.t -> int -> float -> Tensor.t
val mask_ge : Tensor.t -> int -> float -> Tensor.t

(* Generate synthetic datasets *)
val generate_sinusoidal_data : int -> float -> Tensor.t * Tensor.t * Tensor.t
val generate_2d_function_data : int -> Tensor.t * Tensor.t
val train_test_split : Tensor.t -> Tensor.t -> float -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)

(* Evaluation utilities *)
val evaluate_model : 'a -> ('a -> Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> float
val compare_tree_algorithms : Tensor.t -> Tensor.t -> int -> float -> (string * float) list
val compare_forest_algorithms : Tensor.t -> Tensor.t -> int -> int -> float -> (string * float) list
val compare_algorithms : Tensor.t -> Tensor.t -> int -> (string * float) list
val compare_forest_configs : Tensor.t -> Tensor.t -> int -> int -> (string * float) list
val compare_empirical_with_theoretical : Tensor.t -> Tensor.t -> int list -> int -> (int * float * float) list

(* Bounds *)
val variance_martingale_bound : int -> float
val simons_martingale_bound : int -> float
val minimax_martingale_bound : int -> float
val median_martingale_bound : int -> float
val non_uniform_martingale_bound : int -> r:float -> c:float -> float
val asymptotic_minimax_bound : int -> r:float -> c:float -> law_dependent:bool -> float
val cyclic_minimax_bound : int -> int -> float -> float -> float -> float
val empirical_cyclic_minimax_bound : int -> int -> float -> float -> float -> int -> float
val oracle_inequality : int -> int -> int -> float -> float

(* Martingale module - for partition-based martingale approximations *)
module Martingale : sig
  (* Types of splitting rules for partitioning *)
  type split_rule = 
    | Variance   (* Minimize total variance in children *)
    | Simons     (* Split at conditional mean *)
    | Minimax    (* Minimize maximum variance in children *)
    | Median     (* Split at median point *)

  (* Find the best split point for a given array using the variance rule *)
  val find_variance_split : float array -> float * float -> float

  (* Find the best split point for a given array using the Simons rule *)
  val find_simons_split : float array -> float * float -> float

  (* Find the best split point for a given array using the minimax rule *)
  val find_minimax_split : float array -> float * float -> float

  (* Find the best split point for a given array using the median rule *)
  val find_median_split : float array -> float * float -> float

  (* Create a martingale approximation function for the given data
      @param data The array of data values
      @param rule The splitting rule to use
      @param k The depth of the approximation
      @return A function that maps a value to its conditional expectation
  *)
  val create_martingale : float array -> split_rule -> int -> (float -> float)
end

(* DecisionTree module - for tree-based regression models *)
module DecisionTree : sig
  (* Types of split criteria for decision trees *)
  type split_criterion = 
    | VarianceSplit         (* Traditional CART variance reduction *)
    | MinimaxSplit          (* Minimize maximum variance *)
    | CyclicMinimaxSplit    (* Cycle through dimensions with minimax criterion *)
    | L1VarianceSplit       (* Use L1 norm with variance split *)
    | L1MinimaxSplit        (* Use L1 norm with minimax split *)
    | L1CyclicMinimaxSplit  (* Use L1 norm with cyclic minimax split *)

  (* Decision tree node type *)
  type node = {
    region: (float * float) array;  (* min and max for each dimension *)
    value: float;                   (* prediction value for this node *)
    feature: int option;            (* split feature index *)
    threshold: float option;        (* split threshold *)
    left: node option;              (* left child *)
    right: node option;             (* right child *)
    samples: int;                   (* number of samples in this node *)
    depth: int;                     (* depth of this node in the tree *)
    mse: float;                     (* mean squared error at this node *)
  }

  (* Summary metrics for a tree *)
  type node_metrics = {
    n_samples: int;             (* number of samples *)
    n_leaves: int;              (* number of leaf nodes *)
    max_depth: int;             (* maximum depth *)
    avg_depth: float;           (* average depth of leaves *)
    mse: float;                 (* mean squared error *)
    tv_norm: float option;      (* total variation norm estimate *)
  }

  (* Calculate metrics for a decision tree *)
  val calc_node_metrics : node -> node_metrics

  (* Check if a region is splittable *)
  val is_splittable : Tensor.t -> Tensor.t -> (float * float) array -> bool

  (* Find best split using the VarianceSplit criterion *)
  val find_variance_split : Tensor.t -> Tensor.t -> int * float * float

  (* Find best split using the MinimaxSplit criterion *)
  val find_minimax_split : Tensor.t -> Tensor.t -> int * float * float

  (* Find best split using the CyclicMinimaxSplit criterion *)
  val find_cyclic_minimax_split : Tensor.t -> Tensor.t -> int -> int * float * float

  (* Find best split using the L1VarianceSplit criterion *)
  val find_l1_variance_split : Tensor.t -> Tensor.t -> int * float * float

  (* Find best split using the L1MinimaxSplit criterion *)
  val find_l1_minimax_split : Tensor.t -> Tensor.t -> int * float * float

  (* Find best split using the L1CyclicMinimaxSplit criterion *)
  val find_l1_cyclic_minimax_split : Tensor.t -> Tensor.t -> int -> int * float * float

  (* Fit a decision tree on the data
      @param x The input features tensor (n_samples x n_features)
      @param y The target values tensor (n_samples)
      @param max_depth The maximum depth of the tree
      @param criterion The splitting criterion to use
      @return The fitted decision tree
  *)
  val fit : Tensor.t -> Tensor.t -> int -> split_criterion -> node

  (* Predict a single sample using the tree
      @param tree The decision tree model
      @param x The input feature array
      @return The predicted value
  *)
  val predict_sample : node -> float array -> float

  (* Predict multiple samples
      @param tree The decision tree model
      @param x The input features tensor (n_samples x n_features)
      @return The predicted values tensor (n_samples)
  *)
  val predict : node -> Tensor.t -> Tensor.t

  (* Calculate mean squared error between true and predicted values *)
  val mse : Tensor.t -> Tensor.t -> Tensor.t

  (* Calculate mean absolute error between true and predicted values *)
  val mae : Tensor.t -> Tensor.t -> Tensor.t

  (* Calculate R^2 score between true and predicted values *)
  val r2_score : Tensor.t -> Tensor.t -> Tensor.t

  (* Calculate explained variance score between true and predicted values *)
  val explained_variance_score : Tensor.t -> Tensor.t -> Tensor.t

  (* Calculate total variation norm of a function *)
  val tv_norm : float array -> float

  (* Perform k-fold cross-validation
      @param x The input features tensor
      @param y The target values tensor
      @param folds The number of folds
      @param max_depth The maximum depth of the tree
      @param criterion The splitting criterion to use
      @return The mean and standard deviation of the MSE scores
  *)
  val cross_validate : Tensor.t -> Tensor.t -> int -> int -> split_criterion -> float * float

  (* RandomForest module for ensemble methods *)
  module RandomForest : sig
    (* Random forest model type *)
    type forest = {
      trees: node array;                     (* Array of decision trees *)
      criteria: split_criterion array;       (* Criteria used for each tree *)
      feature_importances: float array option; (* Feature importance scores *)
    }

    (* Create bootstrap samples from the data *)
    val bootstrap : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t

    (* Calculate feature importances from a random forest *)
    val calculate_feature_importances : node array -> int -> float array

    (* Fit a random forest
        @param x The input features tensor
        @param y The target values tensor
        @param n_trees The number of trees in the forest
        @param max_depth The maximum depth of each tree
        @param criteria The splitting criteria to use (cycling through them)
        @return The fitted random forest
    *)
    val fit : Tensor.t -> Tensor.t -> int -> int -> split_criterion array -> forest

    (* Predict with a random forest
        @param forest The random forest model
        @param x The input features tensor
        @return The predicted values tensor
    *)
    val predict : forest -> Tensor.t -> Tensor.t
  end
end
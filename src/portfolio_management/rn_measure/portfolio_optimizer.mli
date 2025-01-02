open Torch

type problem_type = MeanVariance | RiskMinimization
type constraint_type = NoConstraint | BoxConstraint of float * float | GroupConstraint of int list list * float list * float list

type t

(** [create n returns covariance risk_aversion cardinality problem_type ?max_community_size ?constraint ()] 
    creates a new portfolio optimization problem 
    @param n Number of assets
    @param returns Tensor of asset returns
    @param covariance Covariance matrix of asset returns
    @param risk_aversion Risk aversion coefficient
    @param cardinality Desired proportion of assets to include
    @param problem_type Type of optimization problem (MeanVariance or RiskMinimization)
    @param max_community_size Optional maximum size for communities
    @param constraint Optional portfolio constraint
*)
val create : 
  int -> 
  float tensor -> 
  float tensor -> 
  float -> 
  float -> 
  problem_type -> 
  ?max_community_size:int -> 
  ?constraint:constraint_type ->
  unit -> t

(** [preprocess_correlation_matrix t] preprocesses the correlation matrix *)
val preprocess_correlation_matrix : t -> float tensor

(** [fit_marchenko_pastur t] fits the Marchenko-Pastur distribution *)
val fit_marchenko_pastur : t -> float * float

(** [clean_correlation_matrix t] cleans the correlation matrix *)
val clean_correlation_matrix : t -> float tensor

(** [cluster t] performs clustering on the assets *)
val cluster : t -> int tensor

(** [risk_rebalance t communities] performs risk rebalancing *)
val risk_rebalance : t -> int tensor -> float

(** [optimize_subproblem t community risk_aversion] optimizes a subproblem *)
val optimize_subproblem : t -> int tensor -> float -> (float tensor, string) result Lwt.t

(** [aggregate_results t communities solutions] aggregates subproblem solutions *)
val aggregate_results : t -> int tensor -> float tensor list -> float tensor

(** [optimize t] performs the full portfolio optimization *)
val optimize : t -> (float tensor, string) result Lwt.t

(** [portfolio_return t weights] calculates the portfolio return *)
val portfolio_return : t -> float tensor -> float

(** [portfolio_risk t weights] calculates the portfolio risk *)
val portfolio_risk : t -> float tensor -> float

(** [sharpe_ratio t weights] calculates the Sharpe ratio *)
val sharpe_ratio : t -> float tensor -> float

(** [get_communities t communities] returns the communities and their assets *)
val get_communities : t -> int tensor -> (int * float tensor) list

(** [get_community_statistics t communities] returns statistics for each community *)
val get_community_statistics : t -> int tensor -> (int * float * float * float) list

(** [apply_constraint t weights] applies the portfolio constraints to the given weights *)
val apply_constraint : t -> float tensor -> float tensor

(** [test t] runs basic tests on the portfolio optimizer *)
val test : t -> unit Lwt.t
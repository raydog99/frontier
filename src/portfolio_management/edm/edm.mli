open Torch

module Edm : sig
  type stock = {
    id: string;
    returns: Tensor.t;
    sector: string;
  }

  type network = {
    stocks: stock list;
    edges: (string * string) list;
  }

  type risk_measure = VaR | ES

  type community = stock list

  val calculate_edm : stock -> stock -> Tensor.t
  val construct_network : stock list -> float -> network
  val find_max_independent_set : network -> stock list
  val optimize_portfolio : stock list -> risk_measure -> float -> float -> float list
  val load_stock_data : string -> stock list
  val calculate_var : Tensor.t -> float -> Tensor.t
  val calculate_es : Tensor.t -> float -> Tensor.t
  val calculate_portfolio_return : Tensor.t -> Tensor.t -> Tensor.t
  val calculate_portfolio_risk : Tensor.t -> Tensor.t -> risk_measure -> float -> Tensor.t
end

module Analysis : sig
  open Edm

  type network_stats = {
    num_nodes: int;
    num_edges: int;
    avg_degree: float;
    max_degree: int;
    min_degree: int;
  }

  type descriptive_stats = {
    mean: float;
    median: float;
    std: float;
    min: float;
    max: float;
    skewness: float;
    kurtosis: float;
  }

  val betweenness_centrality : network -> (string, float) Hashtbl.t
  val detect_communities : network -> community list
  val calculate_modularity : network -> community list -> float
  val network_statistics : network -> network_stats
  val descriptive_statistics : Tensor.t -> descriptive_stats
  val classify_by_sector : stock list -> (string * stock list) list
  val calculate_portfolio_performance : float list -> stock list -> int -> int -> float list
  val backtest_portfolio : (stock list -> risk_measure -> float -> float -> float list) -> stock list -> risk_measure -> float -> float -> int -> int -> float list
  val compare_strategies : (stock list -> risk_measure -> float -> float -> float list) list -> stock list -> risk_measure -> float -> float -> int -> int -> ((stock list -> risk_measure -> float -> float -> float list) * float list) list
  val optimize_portfolio_subset : stock list -> stock list -> risk_measure -> float -> float -> float list
  val create_combined_strategy : (stock list -> risk_measure -> float -> float -> float list) -> (stock list -> risk_measure -> float -> float -> float list) -> stock list -> risk_measure -> float -> float -> float list
end
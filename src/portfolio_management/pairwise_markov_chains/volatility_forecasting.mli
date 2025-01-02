open Torch

type observation = {
    volatility: float;
    log_return: float;
  }

type garch_params = {
    omega: float;
    alpha: float;
    beta: float;
  }

type activation =
    | Tanh
    | ReLU
    | Identity

type layer = {
    weights: Tensor.t;
    bias: Tensor.t;
    activation: activation;
  }

type fnn_config = {
    input_dim: int;
    hidden_dims: int list;
    output_dim: int;
  }

type fnn = {
    layers: layer list;
    config: fnn_config;
  }

type loss =
    | MSE
    | MAE
    | RMSE

type training_config = {
    max_epochs: int;
    batch_size: int;
    learning_rate: float;
    momentum: float;
    early_stopping_patience: int;
    loss_fn: loss;
  }

type training_state = {
    epoch: int;
    loss: float;
    best_loss: float;
    patience_counter: int;
  }

val safe_log : Tensor.t -> Tensor.t
val safe_div : Tensor.t -> Tensor.t -> Tensor.t
val safe_exp : Tensor.t -> Tensor.t
val stable_softmax : Tensor.t -> Tensor.t
val validate_tensor : ?name:string -> Tensor.t -> (Tensor.t, string) result
val calc_log_returns : Tensor.t -> Tensor.t
val calc_volatility : Tensor.t -> int -> Tensor.t
val calc_volatility_robust : Tensor.t -> int -> Tensor.t
val create_windows : Types.observation list -> int -> Types.observation list list
val data_to_tensor : Types.observation list -> Tensor.t
val data_to_tensors : Types.observation list -> Tensor.t * Tensor.t

module FinancialData : sig
  type market_data = {
    open_prices: Tensor.t;
    high_prices: Tensor.t;
    low_prices: Tensor.t;
    close_prices: Tensor.t;
    volume: Tensor.t;
    timestamp: float array;
  }

  type market_features = {
    returns: Tensor.t;
    volatility: Tensor.t;
    ranges: Tensor.t;
    volume_ratio: Tensor.t;
  }

  val load_market_data : string -> market_data
  val create_features : market_data -> int -> market_features
  val calc_range_volatility : market_data -> int -> Tensor.t
  val calc_volume_adjusted_volatility : market_data -> Tensor.t -> int -> Tensor.t
  val calc_realized_volatility : Tensor.t -> int -> Tensor.t
end

module Garch : sig
  val predict : Types.garch_params -> prev_volatility:float -> prev_return:float -> float
  val train : Types.observation list -> float -> int -> Types.garch_params
  val train_with_constraints : Types.observation list -> float -> int -> Types.garch_params
  val predict_batch : Types.garch_params -> Tensor.t -> Tensor.t
end

module FNN : sig
  val create_layer : int -> int -> Types.activation -> Types.layer
  val init_network : Types.fnn_config -> Types.fnn
  val forward : Types.fnn -> Tensor.t -> Tensor.t
  val create_fnn2 : unit -> Types.fnn
  val create_fnn3 : unit -> Types.fnn
  val create_fnn2_3 : unit -> Types.fnn
  val train : Types.fnn -> Types.training_config -> Tensor.t * Tensor.t -> Types.training_state
end

module PMC : sig
  type state = {
    id: int;
    model: [ `GARCH of Types.garch_params | `FNN of Types.fnn ]
  }

  type t = {
    states: state array;
    transition_probs: Tensor.t;
    initial_probs: Tensor.t;
  }

  module StateConstraints : sig
    type constraint_type =
      | NonNegative
      | Sumto1
      | Range of float * float
      | Custom of (Tensor.t -> bool)

    val apply_constraints : Tensor.t -> constraint_type -> Tensor.t
  end

  module EmissionModel : sig
    type emission_type =
      | Gaussian
      | StudentT of float
      | GaussianMixture of int

    val compute_emission : state -> Types.observation -> emission_type -> Tensor.t
  end

  val create : int -> t
  val predict : t -> Types.observation list -> Tensor.t
  val train : t -> Types.training_config -> Types.observation list -> t
  val compute_state_probs : t -> Types.observation -> Tensor.t
end

module PMCGarch : sig
  type model = {
    pmc: PMC.t;
    base_garch: Types.garch_params;
  }

  val create : int -> model
  val predict : model -> Types.observation list -> Tensor.t
  val train : model -> Types.training_config -> Types.observation list -> model
end

module Metrics : sig
  type evaluation = {
    mape: float;
    rmse: float;
    dir_acc: float;
  }

  val evaluate : Tensor.t -> Tensor.t -> evaluation
  val mape : Tensor.t -> Tensor.t -> Tensor.t
  val rmse : Tensor.t -> Tensor.t -> Tensor.t
  val directional_accuracy : Tensor.t -> Tensor.t -> float
end

module Testing : sig
  type test_result = {
    model_name: string;
    metrics: Metrics.evaluation;
    predictions: Tensor.t;
    actual: Tensor.t;
  }

  val split_data : Types.observation list -> float -> float -> 
    (Types.observation list * Types.observation list * Types.observation list)

  val cross_validate : 
    [ `PMCGarch of PMCGarch.model | `GARCH of Types.garch_params | `FNN of Types.fnn ] ->
    Types.observation list -> int -> Types.training_config -> 
    Metrics.evaluation list

  val detect_regimes : Tensor.t -> Tensor.t

  module StatTests : sig
    val diebold_mariano : Tensor.t -> Tensor.t -> Tensor.t -> float

    val model_confidence_set : 
      (string * [ `PMCGarch of PMCGarch.model 
                | `GARCH of Types.garch_params 
                | `FNN of Types.fnn ]) list ->
      Tensor.t list ->
      Tensor.t ->
      float ->
      (string * [ `PMCGarch of PMCGarch.model 
                | `GARCH of Types.garch_params 
                | `FNN of Types.fnn ]) list
  end

  val compare_models : 
    (string * [ `PMCGarch of PMCGarch.model | `GARCH of Types.garch_params | `FNN of Types.fnn ]) list ->
    Types.observation list -> Types.training_config -> test_result list

  val create_model_suite : int -> FinancialData.market_features -> 
    (string * [ `PMCGarch of PMCGarch.model | `GARCH of Types.garch_params | `FNN of Types.fnn ]) list
  
  val run_volatility_experiment : 
    ?n_states:int -> ?window_size:int -> string -> 
    test_result list * (string * Metrics.evaluation list) list

  val run_comprehensive_experiment :
    ?n_states:int -> ?window_size:int -> string ->
    test_result list * 
    (string * Metrics.evaluation list) list * 
    Tensor.t *
    float list *
    (string * [ `PMCGarch of PMCGarch.model | `GARCH of Types.garch_params | `FNN of Types.fnn ]) list
end
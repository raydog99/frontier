val create_folds : int -> int -> Types.fold array
val create_stratified_folds : Types.observation array -> int -> Types.fold array
val cross_fit : Types.observation array -> int -> Models.model_type -> Types.fold array * Types.nuisance_estimates array
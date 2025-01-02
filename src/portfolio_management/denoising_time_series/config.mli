type t = {
  data_file: string;
  input_channels: int;
  num_clusters: int;
  num_contexts: int;
  num_epochs: int;
  learning_rate: float;
  start_index: int;
  end_index: int;
  horizon: int;
}

(** 
    @param data_file Path to the CSV data file
    @param input_channels Number of input channels for the autoencoder
    @param num_clusters Number of clusters for K-means
    @param num_contexts Number of context variables
    @param num_epochs Number of training epochs
    @param learning_rate Learning rate for training
    @param start_index Start index for backtesting
    @param end_index End index for backtesting
    @param horizon Prediction horizon
*)
val create :
  ?data_file:string ->
  ?input_channels:int ->
  ?num_clusters:int ->
  ?num_contexts:int ->
  ?num_epochs:int ->
  ?learning_rate:float ->
  ?start_index:int ->
  ?end_index:int ->
  ?horizon:int ->
  unit -> t

val to_string : t -> string
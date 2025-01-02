open Factor_construction

type config = {
  name: string;
  sort_method: sort_method;
  learning_rate: float;
  num_epochs: int;
  batch_size: int;
}

val compare_models : config list -> stock_data list -> (string * float * float array) list
val run_analysis : stock_data list -> int -> int -> int -> unit
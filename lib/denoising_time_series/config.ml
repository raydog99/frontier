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

let create 
  ?(data_file="cape_y10.csv")
  ?(input_channels=2)
  ?(num_clusters=10)
  ?(num_contexts=7)
  ?(num_epochs=100)
  ?(learning_rate=1e-3)
  ?(start_index=1000)
  ?(end_index=1999)
  ?(horizon=24)
  () =
  {
    data_file;
    input_channels;
    num_clusters;
    num_contexts;
    num_epochs;
    learning_rate;
    start_index;
    end_index;
    horizon;
  }

let to_string t =
  Printf.sprintf
    "Config { data_file: %s; input_channels: %d; num_clusters: %d; num_contexts: %d; num_epochs: %d; learning_rate: %f; start_index: %d; end_index: %d; horizon: %d }"
    t.data_file t.input_channels t.num_clusters t.num_contexts t.num_epochs t.learning_rate t.start_index t.end_index t.horizon
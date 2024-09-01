open Lwt

type stock_data = {
  ticker: string;
  price: float;
  volume: int;
  timestamp: float;
}

val fetch_stock_data : string -> stock_data Lwt.t
val start_data_stream : string list -> (stock_data list -> unit) -> unit
val process_real_time_data : stock_data Queue.t -> stock_data -> unit
val start_pipeline : string list -> unit
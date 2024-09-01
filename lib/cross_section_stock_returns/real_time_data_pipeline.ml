open Lwt
open Lwt_unix
open Yojson.Basic.Util

type stock_data = {
  ticker: string;
  price: float;
  volume: int;
  timestamp: float;
}

let fetch_stock_data ticker =
  let url = Printf.sprintf "https://api.website.com/stocks/%s" ticker in
  Lwt_io.with_connection_url url (fun (ic, oc) ->
    Lwt_io.write_line oc "GET /latest HTTP/1.1" >>= fun () ->
    Lwt_io.write_line oc (Printf.sprintf "Host: api.website.com") >>= fun () ->
    Lwt_io.write_line oc "" >>= fun () ->
    Lwt_io.read ic
  ) >>= fun body ->
  let json = Yojson.Basic.from_string body in
  Lwt.return {
    ticker = json |> member "ticker" |> to_string;
    price = json |> member "price" |> to_float;
    volume = json |> member "volume" |> to_int;
    timestamp = json |> member "timestamp" |> to_float;
  }

let start_data_stream tickers callback =
  let rec stream_loop () =
    Lwt_list.map_p fetch_stock_data tickers >>= fun data ->
    callback data;
    Lwt_unix.sleep 60.0 >>= fun () ->
    stream_loop ()
  in
  Lwt.async stream_loop

let process_real_time_data data_queue stock_data =
  Queue.add stock_data data_queue;
  if Queue.length data_queue > 1000 then
    ignore (Queue.take data_queue);
  ()

let start_pipeline tickers =
  let data_queue = Queue.create () in
  start_data_stream tickers (process_real_time_data data_queue)
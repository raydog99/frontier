open Torch

type country = US | Japan | UK | Canada | France | Germany
[@@deriving show, eq]

type industry = Technology | Healthcare | Finance | ConsumerGoods | Energy
[@@deriving show, eq]

type stock_data = {
  date: int;
  country: country;
  industry: industry;
  size: float;
  bm: float;
  op: float;
  inv: float;
  returns: float;
  exchange: string;
}
[@@deriving show, eq]

type sort_method = TwoByThree | TwoByTwo | TwoByTwoByTwoByTwo
[@@deriving show, eq]

let split_by_nyse_median stocks key =
  let nyse_stocks = List.filter (fun s -> s.exchange = "NYSE") stocks in
  let nyse_median = List.nth (List.sort (fun a b -> compare (key a) (key b)) nyse_stocks) (List.length nyse_stocks / 2) |> key in
  List.partition (fun s -> key s <= nyse_median) stocks

let split_by_percentiles stocks key percentiles =
  let sorted = List.sort (fun a b -> compare (key a) (key b)) stocks in
  let n = List.length sorted in
  let split_points = List.map (fun p -> int_of_float (float_of_int n *. p)) percentiles in
  List.fold_left (fun (acc, rest) split_point ->
    let group, new_rest = List.split_at split_point rest in
    (group :: acc, new_rest)
  ) ([], sorted) split_points
  |> fst
  |> List.rev

let average_returns stocks =
  let sum = List.fold_left (fun acc s -> acc +. s.returns) 0. stocks in
  sum /. float_of_int (List.length stocks)

let construct_smb stocks =
  let small, big = split_by_nyse_median stocks (fun s -> s.size) in
  average_returns small -. average_returns big

let construct_hml stocks sort_method =
  let _, big = split_by_nyse_median stocks (fun s -> s.size) in
  match sort_method with
  | TwoByThree ->
      let low, mid, high = split_by_percentiles big (fun s -> s.bm) [0.3; 0.7] in
      average_returns high -. average_returns low
  | TwoByTwo | TwoByTwoByTwoByTwo ->
      let low, high = split_by_nyse_median big (fun s -> s.bm) in
      average_returns high -. average_returns low

let construct_rmw stocks sort_method =
  let _, big = split_by_nyse_median stocks (fun s -> s.size) in
  match sort_method with
  | TwoByThree ->
      let weak, mid, robust = split_by_percentiles big (fun s -> s.op) [0.3; 0.7] in
      average_returns robust -. average_returns weak
  | TwoByTwo | TwoByTwoByTwoByTwo ->
      let weak, robust = split_by_nyse_median big (fun s -> s.op) in
      average_returns robust -. average_returns weak

let construct_cma stocks sort_method =
  let _, big = split_by_nyse_median stocks (fun s -> s.size) in
  match sort_method with
  | TwoByThree ->
      let aggressive, mid, conservative = split_by_percentiles big (fun s -> s.inv) [0.3; 0.7] in
      average_returns conservative -. average_returns aggressive
  | TwoByTwo | TwoByTwoByTwoByTwo ->
      let aggressive, conservative = split_by_nyse_median big (fun s -> s.inv) in
      average_returns conservative -. average_returns aggressive

let construct_factors stocks sort_method =
  let smb = construct_smb stocks in
  let hml = construct_hml stocks sort_method in
  let rmw = construct_rmw stocks sort_method in
  let cma = construct_cma stocks sort_method in
  (smb, hml, rmw, cma)

let factors_to_tensor factors =
  Tensor.of_float2 [Array.of_list factors]
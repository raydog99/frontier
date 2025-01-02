open Torch

type garch_model =
  | GARCH
  | EGARCH
  | GJR_GARCH

type t = {
  mutable volatility: Tensor.t;
  mutable drift: Tensor.t;
  mutable f: float;
}

type historical_data = {
  dates: string array;
  opens: Tensor.t;
  highs: Tensor.t;
  lows: Tensor.t;
  closes: Tensor.t;
}

let create ~volatility ~drift ~f =
  { volatility; drift; f }
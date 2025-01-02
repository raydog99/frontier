open Types
open Torch

module MicroModel : MODEL

val calibrate_local_volatility : market_data -> (float -> float -> float)
open Factor_construction
open Torch

type input_format = CSV | JSON | SQLite

val load_data : string -> input_format -> stock_data list
val prepare_factors : stock_data list -> sort_method -> Tensor.t * Tensor.t
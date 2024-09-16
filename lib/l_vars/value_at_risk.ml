open Torch

let calculate expected_cost variance confidence_level =
    let standard_normal = Tensor.normal ~mean:0. ~std:1. [1] in
    let z_score = Tensor.quantile standard_normal (Tensor.of_float0 confidence_level) in
    Tensor.(to_float0_exn (expected_cost + (sqrt variance * z_score)))
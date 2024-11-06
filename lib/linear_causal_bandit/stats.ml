let mean tensor =
  let open Tensor in
  sum tensor / float_value (size tensor 0 |> of_int) 
  
let sample_mean samples =
  Tensor.(stack samples ~dim:0 |> mean)
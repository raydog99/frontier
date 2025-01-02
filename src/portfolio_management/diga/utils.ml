open Torch

let preprocess_data data =
  let mean = Tensor.mean data ~dim:[0; 1] ~keepdim:true in
  let std = Tensor.std data ~dim:[0; 1] ~keepdim:true in
  Tensor.((data - mean) / std)

let calculate_return prices =
  let log_returns = Tensor.log (Tensor.slice prices ~dim:0 ~start:1 ~end:None /
                                Tensor.slice prices ~dim:0 ~start:0 ~end:(-1)) in
  Tensor.sum log_returns

let calculate_amplitude prices =
  let high = Tensor.max prices ~dim:0 |> fst in
  let low = Tensor.min prices ~dim:0 |> fst in
  Tensor.((high - low) / low)

let calculate_volatility prices =
  let returns = Tensor.log (Tensor.slice prices ~dim:0 ~start:1 ~end:None /
                            Tensor.slice prices ~dim:0 ~start:0 ~end:(-1)) in
  Tensor.std returns

let evaluate_control_error target generated =
  Tensor.mse_loss target generated ~reduction:Mean

let evaluate_fidelity real_data generated_data =
  let real_return = calculate_return real_data in
  let generated_return = calculate_return generated_data in
  let return_error = Tensor.mse_loss real_return generated_return ~reduction:Mean in
  
  let real_amplitude = calculate_amplitude real_data in
  let generated_amplitude = calculate_amplitude generated_data in
  let amplitude_error = Tensor.mse_loss real_amplitude generated_amplitude ~reduction:Mean in
  
  let real_volatility = calculate_volatility real_data in
  let generated_volatility = calculate_volatility generated_data in
  let volatility_error = Tensor.mse_loss real_volatility generated_volatility ~reduction:Mean in
  
  Tensor.(return_error + amplitude_error + volatility_error)
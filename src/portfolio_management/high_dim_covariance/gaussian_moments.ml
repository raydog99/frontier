open Torch

let check_bounded sigma =
  let eigenvals = Tensor.eigenvalues sigma in
  let max_eigenval = Tensor.max eigenvals |> Tensor.float_value in
  max_eigenval <= 1.0 +. 1e-10

let check_identity_proximity sigma tau =
  let d = Tensor.size sigma 0 in
  let diff = Tensor.sub sigma (Tensor.eye d) in
  let spec_norm = Numerical_stability.compute_spectral_norm diff in
  spec_norm <= tau

let compute_fourth_moment sigma =
  let d = Tensor.size sigma 0 in
  let sigma_sqrt = Numerical_stability.stable_matrix_sqrt sigma in
  
  let result = Tensor.zeros [|d * d; d * d|] in
  for i = 0 to d - 1 do
    for j = 0 to d - 1 do
      for k = 0 to d - 1 do
        for l = 0 to d - 1 do
          let idx1 = i * d + j in
          let idx2 = k * d + l in
          let val1 = Tensor.get sigma_sqrt i k |> 
            Tensor.float_value in
          let val2 = Tensor.get sigma_sqrt j l |> 
            Tensor.float_value in
          let val3 = Tensor.get sigma_sqrt i l |> 
            Tensor.float_value in
          let val4 = Tensor.get sigma_sqrt j k |> 
            Tensor.float_value in
          let value = val1 *. val2 +. val3 *. val4 in
          Tensor.set result [|idx1; idx2|] value
        done
      done
    done
  done;
  result
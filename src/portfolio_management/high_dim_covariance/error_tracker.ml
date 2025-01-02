open Torch

let compute_bounds ~estimate ~true_cov =
  let diff = Tensor.sub estimate true_cov in
  let frob_norm = Tensor.frobenius_norm diff |> 
    Tensor.float_value in
  let spec_norm = Numerical_stability.compute_spectral_norm diff in
  
  (* Compute multiplicative error *)
  let true_inv_sqrt = Numerical_stability.stable_inverse_sqrt true_cov in
  let normalized = Tensor.mm 
    (Tensor.mm true_inv_sqrt estimate) true_inv_sqrt in
  let mult_error = Tensor.frobenius_norm 
    (Tensor.sub normalized 
       (Tensor.eye (Tensor.size normalized 0))) |>
    Tensor.float_value in
  
  (* Compute relative error *)
  let true_norm = Tensor.frobenius_norm true_cov |>
    Tensor.float_value in
  let rel_error = frob_norm /. true_norm in
  
  { Types.
    frobenius = frob_norm;
    spectral = spec_norm;
    multiplicative = mult_error;
    relative = rel_error;
  }

let track_convergence ~current ~previous ~epsilon =
  let diff = Tensor.sub current previous in
  let rel_change = Tensor.frobenius_norm diff |>
    Tensor.float_value in
  let prev_norm = Tensor.frobenius_norm previous |>
    Tensor.float_value in
  
  let relative_error = rel_change /. prev_norm in
  let converged = relative_error <= epsilon *. 0.1 in
  (converged, relative_error)

let verify_error_rate ~history ~epsilon =
  let errors = List.map (fun est ->
    Tensor.frobenius_norm est |> Tensor.float_value
  ) history in
  
  (* Check if errors are decreasing geometrically *)
  let rec check_geometric = function
    | e1 :: e2 :: rest ->
        if e1 >= e2 *. 0.9 then false
        else check_geometric (e2 :: rest)
    | _ -> true in
  
  check_geometric errors
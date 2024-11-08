open Torch
open Types
open Linalg

type transform_aware_model = {
  base_model: Stkr_core.model;
  polynomial_degree: int;
  optimal_rate: float ref;
}

(* Compute optimal learning rate from eigenspectrum *)
let compute_optimal_rate eigenvalues =
  let max_eval = Tensor.max eigenvalues in
  let min_eval = Tensor.min eigenvalues in
  2. /. (Tensor.float_value (Tensor.add max_eval min_eval))

(* Create polynomial transformation *)
let create_polynomial_transform degree =
  fun x -> 
    let rec compute_term i acc =
      if i > degree then acc
      else compute_term (i + 1) (acc +. (Float.pow x (float_of_int i)))
    in
    compute_term 1 0.

(* Enhanced fitting with optimal rate adaptation *)
let fit model x_labeled y_labeled x_unlabeled =
  (* Compute kernel matrix and its eigendecomposition *)
  let k_matrix = model.base_model.kernel x_labeled x_labeled in
  let eigenvalues, _ = eigensystem k_matrix in
  
  (* Update learning rate *)
  let optimal_rate = compute_optimal_rate eigenvalues in
  model.optimal_rate := optimal_rate;
  
  (* Update model parameters *)
  let updated_params = {
    model.base_model.params with learning_rate = optimal_rate 
  } in
  let updated_model = {
    model.base_model with params = updated_params
  } in
  
  (* Fit with updated model *)
  Stkr_core.fit updated_model x_labeled y_labeled x_unlabeled
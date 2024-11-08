open Torch
open Types
open Linalg

type model = {
  kernel: kernel;
  transform: transform_fn;
  params: stkr_params;
}

let create kernel transform params = 
  { kernel; transform; params }

(* Compute transformed gram matrix *)
let compute_transformed_gram model x_train =
  let k_matrix = model.kernel x_train x_train in
  let eigenvalues, eigenvectors = eigensystem k_matrix in
  let transformed_eigenvalues = Tensor.map eigenvalues ~f:model.transform in
  let v = eigenvectors in
  let d = Tensor.diag transformed_eigenvalues in
  Tensor.matmul (Tensor.matmul v d) (Tensor.transpose v 0 1)

(* Core fitting implementation *)
let fit model x_labeled y_labeled x_unlabeled =
  let n_labeled = Tensor.size x_labeled 0 in
  let x_all = Tensor.cat [x_labeled; x_unlabeled] 0 in
  
  (* Compute transformed kernel matrix *)
  let k_transformed = compute_transformed_gram model x_all in
  let k_labeled = Tensor.narrow k_transformed 0 0 n_labeled in
  let k_labeled = Tensor.narrow k_labeled 1 0 n_labeled in
  
  (* Add regularization and solve system *)
  let reg_matrix = Tensor.add k_labeled 
    (Tensor.eye n_labeled |> Tensor.mul model.params.lambda) in
  let alpha = solve_conjugate_gradient 
    reg_matrix y_labeled model.params.epsilon model.params.max_iter in
  
  (x_labeled, alpha)

(* Prediction implementation *)
let predict model (x_train, alpha) x_test =
  let k_test = model.kernel x_train x_test in
  Tensor.matmul k_test (Tensor.unsqueeze alpha 1)
  |> Tensor.squeeze ~dim:[1]
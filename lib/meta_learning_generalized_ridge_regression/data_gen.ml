open Torch
open Types
open Matrix_ops

(* Generate Toeplitz matrix *)
let generate_toeplitz ~dim ~a ~b =
  let matrix = Tensor.zeros [dim; dim] in
  for i = 0 to dim - 1 do
    for j = 0 to dim - 1 do
      if i = j then
        Tensor.set matrix [|i; j|] a
      else if abs (i - j) = 1 then
        Tensor.set matrix [|i; j|] b
      else
        Tensor.set matrix [|i; j|] 0.
    done
  done;
  nearest_positive_definite matrix

(* Generate random Gaussian data *)
let generate_gaussian_data ~rows ~cols =
  Tensor.randn [rows; cols]

(* Generate synthetic task data *)
let generate_task_data ~dim ~num_samples ~omega ~sigma ~sigma_sq =
  (* Generate random design matrix *)
  let z = generate_gaussian_data ~rows:num_samples ~cols:dim in
  let x = Tensor.mm z (Tensor.cholesky sigma) in
  
  (* Generate coefficients *)
  let beta = Tensor.mm 
    (generate_gaussian_data ~rows:dim ~cols:1)
    (Tensor.cholesky (Tensor.mul_scalar omega (1. /. float_of_int dim))) in
  
  (* Generate response with noise *)
  let noise = generate_gaussian_data ~rows:num_samples ~cols:1 |> 
              Tensor.mul_scalar (sqrt sigma_sq) in
  let y = Tensor.(add (mm x beta) noise) in
  {x; y}

(* Generate multiple tasks *)
let generate_tasks ~config ~omega ~sigma =
  List.init config.num_tasks (fun _ ->
    generate_task_data 
      ~dim:config.dim 
      ~num_samples:config.samples_per_task
      ~omega ~sigma 
      ~sigma_sq:config.sigma_sq
  )

(* Generate data for varying sample sizes *)
let generate_varying_samples_tasks ~config ~l0_samples ~omega ~sigma =
  let l0 = int_of_float (0.2 *. float_of_int config.num_tasks) in
  let tasks1 = List.init l0 (fun _ ->
    generate_task_data 
      ~dim:config.dim 
      ~num_samples:l0_samples
      ~omega ~sigma 
      ~sigma_sq:config.sigma_sq
  ) in
  let tasks2 = List.init (config.num_tasks - l0) (fun _ ->
    generate_task_data 
      ~dim:config.dim 
      ~num_samples:config.samples_per_task
      ~omega ~sigma 
      ~sigma_sq:config.sigma_sq
  ) in
  tasks1 @ tasks2
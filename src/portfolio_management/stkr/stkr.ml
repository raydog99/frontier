open Torch
open Types

type model = [
  | `Base of Stkr_core.model
  | `Optimized of OptimizedSTKR.optimized_model
]

(* Smart parameter selection based on data size *)
module AutoParams = struct
  let estimate_kernel_cache_size n_samples feature_dim =
    (* Estimate bytes needed for kernel matrix *)
    let matrix_size = n_samples * n_samples * 8 in  (* 8 bytes per float *)
    let overhead = 1.2 in  (* 20% overhead *)
    int_of_float (float_of_int matrix_size *. overhead /. (1024. *. 1024.))  (* In MB *)

  let estimate_chunk_size n_samples available_memory =
    let mem_per_sample = 8 * n_samples in  (* 8 bytes per float *)
    min n_samples (available_memory * 1024 * 1024 / mem_per_sample)

  let should_use_parallel n_samples =
    n_samples > 5000  (* Threshold for parallel processing *)

  let optimize_params n_samples feature_dim =
    let available_memory = 4096 in  (* 4GB default *)
    let cache_size = estimate_kernel_cache_size n_samples feature_dim in
    let chunk_size = estimate_chunk_size n_samples available_memory in
    let use_parallel = should_use_parallel n_samples in
    cache_size, chunk_size, use_parallel
end

(* Create base model *)
let create kernel transform params = 
  `Base (Stkr_core.create kernel transform params)

(* Create optimized model with automatic parameter selection *)
let create_optimized kernel transform params cache_size chunk_size use_parallel =
  let base_model = Stkr_core.create kernel transform params in
  `Optimized (OptimizedSTKR.create base_model cache_size chunk_size use_parallel)

(* Automatic model creation based on data characteristics *)
let create_auto kernel transform params x_labeled =
  let n_samples = Tensor.size x_labeled 0 in
  let feature_dim = Tensor.size x_labeled 1 in
  let cache_size, chunk_size, use_parallel = 
    AutoParams.optimize_params n_samples feature_dim in
  create_optimized kernel transform params cache_size chunk_size use_parallel

(* Transform selection utilities *)
module Transforms = struct
  let polynomial degree =
    Stkr_transform_aware.create_polynomial_transform degree

  let inverse_laplacian eta epsilon =
    Stkr_transform_agnostic.create_stable_inverse_laplacian eta epsilon

  let adaptive eigenvalues degree =
    let max_eval = Tensor.max eigenvalues |> Tensor.float_value in
    let scale = 1. /. max_eval in
    fun x -> 
      let scaled_x = x *. scale in
      polynomial degree scaled_x
end

(* Kernel selection utilities *)
module Kernels = struct
  let rbf sigma = Base_kernel.rbf_kernel sigma
  
  let composite k1 k2 alpha =
    fun x y -> 
      let k1_xy = k1 x y in
      let k2_xy = k2 x y in
      Tensor.add (Tensor.mul k1_xy alpha) (Tensor.mul k2_xy (1. -. alpha))

  let centered kernel = Base_kernel.center_kernel kernel
end

(* Smart model selection and fitting *)
let smart_fit model x_labeled y_labeled x_unlabeled =
  let n_labeled = Tensor.size x_labeled 0 in
  let n_unlabeled = Tensor.size x_unlabeled 0 in
  let n_total = n_labeled + n_unlabeled in

  (* Choose appropriate implementation based on data size and model type *)
  match model, n_total with
  | `Optimized m, _ -> 
      OptimizedSTKR.fit m x_labeled y_labeled x_unlabeled
  | `Base m, n when n > 10000 ->
      (* Automatically upgrade to optimized version for large datasets *)
      let cache_size, chunk_size, use_parallel = 
        AutoParams.optimize_params n_total (Tensor.size x_labeled 1) in
      let opt_model = OptimizedSTKR.create m cache_size chunk_size use_parallel in
      OptimizedSTKR.fit opt_model x_labeled y_labeled x_unlabeled
  | `Base m, _ ->
      Stkr_core.fit m x_labeled y_labeled x_unlabeled

(* Smart prediction *)
let smart_predict model trained_model x_test =
  let n_test = Tensor.size x_test 0 in
  
  match model with
  | `Optimized m -> 
      OptimizedSTKR.predict m trained_model x_test
  | `Base m when n_test > 5000 ->
      (* Upgrade to optimized for large test sets *)
      let cache_size, chunk_size, use_parallel = 
        AutoParams.optimize_params n_test (Tensor.size x_test 1) in
      let opt_model = OptimizedSTKR.create m cache_size chunk_size use_parallel in
      OptimizedSTKR.predict opt_model trained_model x_test
  | `Base m ->
      Stkr_core.predict m trained_model x_test

(* Validation utilities *)
module Validation = struct
  let cross_validate model x y folds =
    let n = Tensor.size x 0 in
    let fold_size = n / folds in
    
    let errors = Array.make folds 0. in
    for i = 0 to folds - 1 do
      (* Create train/test split *)
      let start_idx = i * fold_size in
      let end_idx = if i = folds - 1 then n else (i + 1) * fold_size in
      
      let x_test = Tensor.narrow x 0 start_idx (end_idx - start_idx) in
      let y_test = Tensor.narrow y 0 start_idx (end_idx - start_idx) in
      
      let x_train = Tensor.cat [
        Tensor.narrow x 0 0 start_idx;
        Tensor.narrow x 0 end_idx (n - end_idx)
      ] 0 in
      let y_train = Tensor.cat [
        Tensor.narrow y 0 0 start_idx;
        Tensor.narrow y 0 end_idx (n - end_idx)
      ] 0 in
      
      (* Fit and predict *)
      let trained_model = smart_fit model x_train y_train Tensor.empty in
      let predictions = smart_predict model trained_model x_test in
      
      (* Compute MSE *)
      let diff = Tensor.sub predictions y_test in
      let mse = Tensor.mean (Tensor.pow diff 2.) in
      errors.(i) <- Tensor.float_value mse
    done;
    
    (* Return mean and std of errors *)
    let mean_error = Array.fold_left (+.) 0. errors /. float_of_int folds in
    let std_error = 
      Array.fold_left (fun acc e -> 
        acc +. ((e -. mean_error) *. (e -. mean_error))
      ) 0. errors in
    let std_error = sqrt (std_error /. float_of_int folds) in
    mean_error, std_error

  (* Model selection using validation set *)
  let select_best_model x_train y_train x_val y_val models =
    List.map (fun model ->
      let trained_model = smart_fit model x_train y_train Tensor.empty in
      let predictions = smart_predict model trained_model x_val in
      let diff = Tensor.sub predictions y_val in
      let mse = Tensor.mean (Tensor.pow diff 2.) |> Tensor.float_value in
      model, mse
    ) models
    |> List.sort (fun (_, mse1) (_, mse2) -> compare mse1 mse2)
    |> List.hd |> fst
end

(* Pipeline construction utilities *)
module Pipeline = struct
  type transform_type = [
    | `Polynomial of int
    | `InverseLaplacian of float * float
    | `Adaptive of int
  ]

  type pipeline = {
    kernel: kernel;
    transform: transform_type;
    params: stkr_params;
  }

  let create_pipeline kernel transform params = {
    kernel;
    transform;
    params;
  }

  let fit_pipeline pipeline x_labeled y_labeled x_unlabeled =
    (* Create appropriate transform *)
    let transform = match pipeline.transform with
    | `Polynomial degree -> Transforms.polynomial degree
    | `InverseLaplacian (eta, epsilon) -> 
        Transforms.inverse_laplacian eta epsilon
    | `Adaptive degree ->
        let k_matrix = pipeline.kernel x_labeled x_labeled in
        let eigenvalues, _ = Linalg.eigensystem k_matrix in
        Transforms.adaptive eigenvalues degree
    in
    
    (* Create and fit model *)
    let model = create_auto pipeline.kernel transform pipeline.params x_labeled in
    smart_fit model x_labeled y_labeled x_unlabeled
end
open Torch

module RegularizationFamily = struct
  type t = {
    g_lambda: float -> float;
    qualification: float;
    lambda_bound: float;
  }

  let verify_conditions g_lambda =
    let check_sup f bound =
      let samples = List.init 1000 (fun i -> float_of_int i /. 1000.0) in
      List.fold_left (fun acc t -> 
        max acc (abs_float (f t))
      ) 0. samples < bound
    in
    let cond1 = check_sup (fun t -> t *. g_lambda t) 1. in
    let cond2 = check_sup (fun t -> 1. -. t *. g_lambda t) 1. in
    let cond3 = check_sup g_lambda 1. in
    cond1 && cond2 && cond3

  let create g_lambda qualification lambda_bound =
    if not (verify_conditions g_lambda) then
      failwith "Invalid regularization family: conditions not satisfied"
    else
      {g_lambda; qualification; lambda_bound}

  let ridge lambda =
    create
      (fun t -> 1. /. (lambda +. t))
      1.0
      (1. /. lambda)

  let pca lambda =
    create
      (fun t -> if t > lambda then 1. /. t else 0.)
      1.0
      (1. /. lambda)

  let gradient_flow lambda =
    create
      (fun t -> (1. -. exp(-. t /. lambda)) /. t)
      1.0
      (1. /. lambda)
end

module SpectralAlgorithm = struct
  type t = {
    regularization: RegularizationFamily.t;
    truncation: int;
    lambda: float;
  }

  module SampleCovariance = struct
    type t = {
      eigenvalues: Tensor.t;
      eigenvectors: Tensor.t;
      points: Tensor.t;
      heat_kernel: HeatKernel.t;
    }

    let create points heat_kernel =
      let h = HeatKernel.build_matrix heat_kernel points 1.0 in
      let eigenvalues, eigenvectors = Tensor.linalg_eigh h in
      {eigenvalues; eigenvectors; points; heat_kernel}

    let apply cov f =
      let h = HeatKernel.build_matrix cov.heat_kernel cov.points 1.0 in
      Tensor.(matmul h f)
  end

  let create regularization truncation lambda = 
    {regularization; truncation; lambda}

  let compute_basis_function points y heat_kernel =
    let h = HeatKernel.build_matrix heat_kernel points 1.0 in
    Tensor.(matmul h y)

  let estimate algo labeled_data unlabeled_data heat_kernel =
    let n_labeled = Tensor.size labeled_data.Dataset.x |> List.hd in
    let x_all = Tensor.cat [labeled_data.x; unlabeled_data.x] ~dim:0 in
    
    let cov = SampleCovariance.create x_all heat_kernel in
    let truncated_eigenvalues = 
      Tensor.narrow cov.eigenvalues ~dim:0 ~start:0 ~length:algo.truncation in
    let truncated_eigenvectors = 
      Tensor.narrow cov.eigenvectors ~dim:1 ~start:0 ~length:algo.truncation in
    
    let g_d = compute_basis_function labeled_data.x labeled_data.y heat_kernel in
    let regularized_eigenvalues = 
      Tensor.map truncated_eigenvalues ~f:algo.regularization.g_lambda in
    
    let coefs = Tensor.(matmul (transpose truncated_eigenvectors ~dim0:0 ~dim1:1) g_d) in
    let scaled_coefs = Tensor.(coefs * regularized_eigenvalues) in
    Tensor.(matmul truncated_eigenvectors scaled_coefs)
end

module DiffusionSpectralAlgorithm = struct
  type t = {
    config: Config.t;
    heat_kernel: HeatKernel.t option;
    eigensystem: SpectralAlgorithm.SampleCovariance.t option;
    regularization: RegularizationFamily.t;
  }

  let create config regularization = 
    {config; heat_kernel = None; eigensystem = None; regularization}

  let init_heat_kernel algo points =
    let laplacian = LaplaceBeltrami.normalized_laplacian points algo.config.epsilon in
    let eigensystem = LaplaceBeltrami.compute_eigensystem laplacian algo.config.truncation_k in
    let heat_kernel = HeatKernel.create algo.config.epsilon algo.config.truncation_k eigensystem in
    {algo with heat_kernel = Some heat_kernel}

  let compute_eigensystem algo points m =
    match algo.heat_kernel with
    | None -> failwith "Heat kernel not initialized"
    | Some heat_kernel ->
        let heat_matrix = HeatKernel.build_matrix heat_kernel points algo.config.time in
        let h_labeled = Tensor.narrow heat_matrix ~dim:0 ~start:0 ~length:m in
        let eigenvalues, eigenvectors = 
          Tensor.linalg_eigh Tensor.(h_labeled / float_of_int m) in
        let cov = SpectralAlgorithm.SampleCovariance.create points heat_kernel in
        {algo with eigensystem = Some cov}

  let fit algo labeled_data unlabeled_data =
    let x_all = Tensor.cat [labeled_data.x; unlabeled_data.x] ~dim:0 in
    let m = Tensor.size labeled_data.x |> List.hd in
    
    (* Initialize heat kernel and compute eigensystem *)
    let algo = init_heat_kernel algo x_all in
    let algo = compute_eigensystem algo x_all m in
    
    match algo.heat_kernel, algo.eigensystem with
    | Some heat_kernel, Some eigensystem ->
        let spectral = SpectralAlgorithm.create 
          algo.regularization algo.config.truncation_q algo.config.lambda in
        SpectralAlgorithm.estimate spectral labeled_data unlabeled_data heat_kernel
    | _ -> failwith "Algorithm initialization failed"
end

module RegressionPipeline = struct
  let run config labeled_data unlabeled_data =
    let regularization = RegularizationFamily.ridge config.lambda in
    let algo = DiffusionSpectralAlgorithm.create config regularization in
    DiffusionSpectralAlgorithm.fit algo labeled_data unlabeled_data
    
  let run_with_power_space config labeled_data unlabeled_data alpha =
    let x_all = Tensor.cat [labeled_data.Dataset.x; unlabeled_data.x] ~dim:0 in
    
    let laplacian = LaplaceBeltrami.normalized_laplacian x_all config.epsilon in
    let eigensystem = LaplaceBeltrami.compute_eigensystem laplacian config.truncation_k in
    let power_space = PowerSpace.create alpha 
      eigensystem.eigenvalues eigensystem.eigenfunctions config.time in
    
    let heat_kernel = HeatKernel.create config.epsilon config.truncation_k eigensystem in
    let regularization = RegularizationFamily.ridge config.lambda in
    
    let spectral = SpectralAlgorithm.create 
      regularization config.truncation_q config.lambda in
    SpectralAlgorithm.estimate spectral labeled_data unlabeled_data heat_kernel
end
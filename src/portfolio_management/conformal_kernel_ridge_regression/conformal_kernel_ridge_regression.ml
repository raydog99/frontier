open Torch

(* Numerical utilities and stability control *)
module NumericControl = struct
  type precision_level = 
    | Single
    | Double
    | Mixed

  type stability_config = {
    precision: precision_level;
    epsilon: float;
    max_condition_number: float;
    pivot_threshold: float;
    decomposition_method: [`QR | `SVD | `Cholesky];
  }

  let default_config = {
    precision = Double;
    epsilon = 1e-12;
    max_condition_number = 1e15;
    pivot_threshold = 1e-10;
    decomposition_method = `Cholesky;
  }

  let to_precision tensor = function
    | Single -> Tensor.to_type tensor ~dtype:Float
    | Double -> Tensor.to_type tensor ~dtype:Double
    | Mixed -> tensor

  (* Stable matrix decomposition *)
  let stable_decomposition config matrix =
    let matrix = to_precision matrix config.precision in
    match config.decomposition_method with
    | `QR -> 
        let q, r = Tensor.qr matrix in
        let condition = 
          let diag = Tensor.diagonal r in
          let max_d = Tensor.max diag |> Tensor.get_float0 in
          let min_d = Tensor.min diag |> Tensor.get_float0 in
          if min_d < config.epsilon then infinity
          else max_d /. min_d in
        (q, r)
    | `SVD ->
        let u, s, v = Tensor.svd matrix in
        let condition = 
          let max_s = Tensor.max s |> Tensor.get_float0 in
          let min_s = Tensor.min s |> Tensor.get_float0 in
          max_s /. min_s in
        (u, s, v)
    | `Cholesky ->
        let l = Tensor.cholesky matrix in
        (l, Tensor.transpose l ~dim0:0 ~dim1:1)

  (* Stable solver using chosen decomposition *)
  let stable_solve ?(config=default_config) a b =
    let a = to_precision a config.precision in
    let b = to_precision b config.precision in
    match config.decomposition_method with
    | `QR ->
        let q, r = stable_decomposition config a in
        let qt_b = Tensor.mm (Tensor.transpose q ~dim0:0 ~dim1:1) b in
        Tensor.triangular_solve r qt_b ~upper:true
    | `SVD ->
        let u, s, v = stable_decomposition config a in
        let s_inv = Tensor.where_scalar s 
          ~condition:(fun x -> x > config.epsilon)
          ~true_tensor:(Tensor.reciprocal s)
          ~false_tensor:(Tensor.zeros_like s) in
        let ut_b = Tensor.mm (Tensor.transpose u ~dim0:0 ~dim1:1) b in
        let s_inv_ut_b = Tensor.(ut_b * s_inv.unsqueeze ~dim:1) in
        Tensor.mm v s_inv_ut_b
    | `Cholesky ->
        let l, lt = stable_decomposition config a in
        let y = Tensor.triangular_solve l b ~upper:false in
        Tensor.triangular_solve lt y ~upper:true
end

(* Utility functions *)
module Utils = struct
  let sort_tensor ?(stable=true) t =
    if stable then
      let sorted, indices = Tensor.sort t ~dim:0 ~descending:false in
      (sorted, indices)
    else
      Tensor.sort t ~dim:0 ~descending:false

  let eye n = Tensor.eye n
  
  let range n = 
    Tensor.arange ~start:0 ~end_:(float_of_int n) ~step:1.0

  let concat_vertical t1 t2 =
    Tensor.cat [t1; t2] ~dim:0

  (* Compute quantile efficiently *)
  let compute_quantile t p =
    let sorted = fst (sort_tensor t) in
    let n = Tensor.size sorted 0 in
    let idx = int_of_float (float_of_int n *. p) in
    Tensor.get_float1 sorted idx

  (* Efficient matrix operations *)
  let solve_system = NumericControl.stable_solve

  (* Memory-efficient chunked operations *)
  let chunked_operation ~chunk_size ~f tensor =
    let n = Tensor.size tensor 0 in
    let num_chunks = (n + chunk_size - 1) / chunk_size in
    let results = ref [] in
    
    for i = 0 to num_chunks - 1 do
      let start_idx = i * chunk_size in
      let end_idx = min (start_idx + chunk_size) n in
      let chunk = Tensor.narrow tensor ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
      results := (f chunk) :: !results;
      Gc.compact ()
    done;
    List.rev !results
end

(* Kernels *)
module Kernel = struct
  type kernel_type =
    | Gaussian
    | Linear
    | Polynomial
    | Laplacian
    | RationalQuadratic
    | Periodic

  (* Kernel module signature *)
  module type S = sig
    type t
    val create : ?params:float array -> kernel_type -> t
    val compute : t -> Tensor.t -> Tensor.t -> Tensor.t
    val batch_compute : t -> Tensor.t -> Tensor.t -> Tensor.t
    val gradient : t -> Tensor.t -> Tensor.t -> Tensor.t
    val get_type : t -> kernel_type
    val to_device : t -> Device.t -> t
  end

  module KernelImpl : S = struct
    type t = {
      kernel_type: kernel_type;
      params: float array;
      device: Device.t;
    }

    let create ?(params=[|1.0|]) kernel_type = {
      kernel_type;
      params;
      device = Device.Cpu;
    }

    let get_type t = t.kernel_type

    let to_device t device = { t with device }

    let compute_gaussian t x1 x2 =
      let precision = t.params.(0) in
      let diff = Tensor.sub x1 x2 in
      let squared_dist = Tensor.(diff * diff) |> Tensor.sum ~dim:[0] ~keepdim:true in
      Tensor.exp (Tensor.mul_scalar squared_dist (-. precision))

    let compute_linear t x1 x2 =
      let scale = t.params.(0) in
      let prod = Tensor.mm x1 (Tensor.transpose x2 ~dim0:0 ~dim1:1) in
      Tensor.mul_scalar prod scale

    let compute_polynomial t x1 x2 =
      let scale = t.params.(0) in
      let degree = int_of_float t.params.(1) in
      let bias = t.params.(2) in
      let base = Tensor.mm x1 (Tensor.transpose x2 ~dim0:0 ~dim1:1) in
      let scaled = Tensor.mul_scalar base scale in
      let with_bias = Tensor.add_scalar scaled bias in
      Tensor.pow_scalar with_bias (float_of_int degree)

    let compute_laplacian t x1 x2 =
      let length_scale = t.params.(0) in
      let diff = Tensor.sub x1 x2 in
      let norm = Tensor.norm diff ~p:1. ~dim:[0] in
      Tensor.exp (Tensor.div_scalar norm (-. length_scale))

    let compute_rational_quadratic t x1 x2 =
      let length_scale = t.params.(0) in
      let alpha = t.params.(1) in
      let diff = Tensor.sub x1 x2 in
      let squared_dist = Tensor.(diff * diff) |> Tensor.sum ~dim:[0] in
      let scaled = Tensor.div_scalar squared_dist (2. *. alpha *. length_scale ** 2.) in
      Tensor.pow_scalar (Tensor.add_scalar scaled 1.) (-. alpha)

    let compute_periodic t x1 x2 =
      let length_scale = t.params.(0) in
      let period = t.params.(1) in
      let diff = Tensor.sub x1 x2 in
      let scaled = Tensor.mul_scalar diff (2. *. Float.pi /. period) in
      let sinterm = Tensor.sin (Tensor.div_scalar scaled 2.) in
      let squared = Tensor.(sinterm * sinterm) |> Tensor.sum ~dim:[0] in
      Tensor.exp (Tensor.mul_scalar squared (-2. /. length_scale ** 2.))

    let compute t x1 x2 =
      let x1 = Tensor.to_device x1 t.device in
      let x2 = Tensor.to_device x2 t.device in
      match t.kernel_type with
      | Gaussian -> compute_gaussian t x1 x2
      | Linear -> compute_linear t x1 x2
      | Polynomial -> compute_polynomial t x1 x2
      | Laplacian -> compute_laplacian t x1 x2
      | RationalQuadratic -> compute_rational_quadratic t x1 x2
      | Periodic -> compute_periodic t x1 x2

    let batch_compute t x1 x2 =
      let x1 = Tensor.to_device x1 t.device in
      let x2 = Tensor.to_device x2 t.device in
      let n1 = Tensor.size x1 0 in
      let n2 = Tensor.size x2 0 in
      
      match t.kernel_type with
      | Gaussian ->
          let x1_expanded = Tensor.unsqueeze x1 ~dim:1 in
          let x2_expanded = Tensor.unsqueeze x2 ~dim:0 in
          let diff = Tensor.sub x1_expanded x2_expanded in
          let squared_dist = Tensor.(diff * diff) |> Tensor.sum ~dim:[2] in
          Tensor.exp (Tensor.mul_scalar squared_dist (-. t.params.(0)))
      | _ ->
          (* For other kernels, compute element by element *)
          let result = Tensor.zeros [n1; n2] ~device:t.device in
          for i = 0 to n1 - 1 do
            for j = 0 to n2 - 1 do
              let x1i = Tensor.select x1 ~dim:0 ~index:i in
              let x2j = Tensor.select x2 ~dim:0 ~index:j in
              let k = compute t x1i x2j in
              Tensor.copy_ 
                (Tensor.narrow result ~dim:0 ~start:i ~length:1)
                k
            done
          done;
          result

    (* Gradient computation for kernel parameters *)
    let gradient t x1 x2 =
      let k = compute t x1 x2 in
      Tensor.requires_grad_ k true;
      let grad = Tensor.backward k;
      Tensor.grad k
  end
end

(* Kernel ridge regression *)
module KRR = struct
  type t = {
    kernel: Kernel.KernelImpl.t;
    lambda: float;
    x_train: Tensor.t;
    y_train: Tensor.t;
    alpha: Tensor.t;
    gram_matrix: Tensor.t;
    cached_residuals: (Tensor.t * float) option;
  }

  let create ?(lambda=1.0) ?(kernel=Kernel.KernelImpl.create Kernel.Gaussian) x_train y_train =
    Error.check_dimensions x_train y_train;
    let n = Tensor.size x_train 0 in
    let k_matrix = Kernel.KernelImpl.batch_compute kernel x_train x_train in
    let reg_matrix = Tensor.add k_matrix (Utils.eye n |> Tensor.mul_scalar lambda) in
    
    (* Solve (K + λI)α = y *)
    let alpha = NumericControl.stable_solve reg_matrix y_train in
    
    {
      kernel;
      lambda;
      x_train;
      y_train;
      alpha;
      gram_matrix = k_matrix;
      cached_residuals = None;
    }

  (* Compute in-sample residuals *)
  let compute_in_sample_residuals t =
    match t.cached_residuals with
    | Some (res, lambda) when Float.equal lambda t.lambda -> res
    | _ ->
        let predictions = Tensor.mm t.gram_matrix t.alpha in
        let residuals = Tensor.sub t.y_train predictions in
        residuals

  (* Compute leave-one-out residuals *)
  let compute_loo_residuals t =
    let n = Tensor.size t.x_train 0 in
    let residuals = Tensor.zeros [n] in
    
    let q_matrix = NumericControl.stable_solve 
      (Tensor.add t.gram_matrix (Utils.eye n |> Tensor.mul_scalar t.lambda))
      (Utils.eye n) in
    
    (* Compute m_i values *)
    for i = 0 to n - 1 do
      let k_i = Tensor.select t.gram_matrix ~dim:0 ~index:i in
      let q_i = Tensor.select q_matrix ~dim:0 ~index:i in
      
      (* Compute leverage score *)
      let m_i = t.lambda +. 
        (Tensor.get_float2 t.gram_matrix i i) -.
        (Tensor.dot k_i q_i |> Tensor.get_float0) in
      
      (* Compute LOO residual *)
      let in_sample_res = Tensor.get_float1 (compute_in_sample_residuals t) i in
      let lambda_m = t.lambda /. m_i in
      Tensor.set_float1 residuals i (in_sample_res /. lambda_m)
    done;
    residuals

  let predict t x_test =
    let k_test = Kernel.KernelImpl.batch_compute t.kernel x_test t.x_train in
    Tensor.mm k_test t.alpha

  (* Compute log marginal likelihood *)
  let log_likelihood t =
    let n = Tensor.size t.x_train 0 in
    let rx = Tensor.add t.gram_matrix (Utils.eye n |> Tensor.mul_scalar t.lambda) in
    
    let log_det = Tensor.logdet rx in
    let rx_inv_y = NumericControl.stable_solve rx t.y_train in
    let quad_term = Tensor.dot t.y_train rx_inv_y in
    
    -0.5 *. (
      float_of_int n *. log (2. *. Float.pi) +.
      Tensor.get_float0 log_det +.
      Tensor.get_float0 quad_term
    )
end

(* Region computation utilities *)
module RegionComputation = struct
  type interval = {
    lower: float;
    upper: float;
    coverage: float;
    score: float;
  }

  type region_type =
    | EmptyRegion
    | FullRegion
    | SingleInterval of interval
    | UnionIntervals of interval list

  (* Compute p-value *)
  let compute_pvalue residuals r =
    let n = Tensor.size residuals 0 in
    let count = Tensor.ge residuals (Tensor.full [1] r) |>
      Tensor.sum ~dtype:Float |> Tensor.get_float0 in
    (count +. 1.) /. (float_of_int n +. 1.)

  (* Efficient computation of regions Si *)
  let compute_regions residuals x_n y_n b_n =
    let n = Tensor.size residuals 0 in
    let sorted_residuals, indices = Utils.sort_tensor residuals in
    let regions = ref [] in
    
    for i = 0 to n - 1 do
      let r_i = Tensor.get_float1 sorted_residuals i in
      let b_i = Tensor.get_float1 b_n i in
      let b_n = Tensor.get_float1 b_n (n-1) in
      
      (* Check region type based on paper's conditions *)
      if Float.abs (b_i -. b_n) < NumericControl.default_config.epsilon then
        if r_i >= 0. then
          regions := FullRegion :: !regions
        else
          regions := EmptyRegion :: !regions
      else
        let p = (-. r_i) /. (b_i +. b_n) in
        let q = r_i /. (b_n -. b_i) in
        if b_i > b_n then
          regions := SingleInterval {
            lower = min p q;
            upper = max p q;
            coverage = compute_pvalue sorted_residuals r_i;
            score = abs_float r_i
          } :: !regions
        else
          regions := SingleInterval {
            lower = q;
            upper = p;
            coverage = compute_pvalue sorted_residuals r_i;
            score = abs_float r_i
          } :: !regions
    done;
    !regions

  (* Merge overlapping intervals *)
  let merge_intervals intervals =
    let sorted = List.sort 
      (fun i1 i2 -> compare i1.lower i2.lower)
      intervals in
    
    let rec merge acc = function
      | [] -> acc
      | i::rest ->
          match acc with
          | [] -> merge [i] rest
          | prev::others ->
              if prev.upper +. NumericControl.default_config.epsilon >= i.lower then
                let merged = {
                  lower = min prev.lower i.lower;
                  upper = max prev.upper i.upper;
                  coverage = max prev.coverage i.coverage;
                  score = min prev.score i.score;
                } in
                merge (merged::others) rest
              else
                merge (i::acc) rest in
    
    merge [] sorted

  (* Compute confidence region *)
  let compute_confidence_region regions alpha =
    let valid_intervals = List.filter_map (function
      | SingleInterval i when i.coverage >= 1. -. alpha -> Some i
      | _ -> None
    ) regions in
    
    match valid_intervals with
    | [] -> EmptyRegion
    | [interval] -> SingleInterval interval
    | intervals ->
        let merged = merge_intervals intervals in
        match merged with
        | [interval] -> SingleInterval interval
        | intervals -> UnionIntervals intervals
end

(* Ridge regression confidence machine *)
module RRCM = struct
  type t = {
    krr: KRR.t;
    alpha: float;
  }

  let create ?(alpha=0.05) krr = {
    krr;
    alpha;
  }

  (* Compute B_n vector *)
  let compute_bn t x_n =
    let k_n = Kernel.KernelImpl.batch_compute t.krr.kernel 
      (Tensor.unsqueeze x_n ~dim:0) t.krr.x_train in
    
    let q_n = NumericControl.stable_solve ~config:{
      NumericControl.default_config with
      decomposition_method = `QR;
    } t.krr.gram_matrix k_n in
    
    let m_n = t.krr.lambda +. 
      (Kernel.KernelImpl.compute t.krr.kernel x_n x_n |> Tensor.get_float2 0 0) -.
      (Tensor.mm k_n q_n |> Tensor.get_float2 0 0) in
    
    (* Construct B_n *)
    let b_top = Tensor.neg q_n in
    let b_bottom = Tensor.ones [1; 1] |> Tensor.div_scalar m_n in
    Tensor.cat [b_top; b_bottom] ~dim:0

  (* Compute confidence region *)
  let compute_confidence_region t x_test y_test =
    let residuals = KRR.compute_in_sample_residuals t.krr in
    let b_n = compute_bn t x_test in
    
    (* Compute regions *)
    let regions = RegionComputation.compute_regions residuals x_test y_test b_n in
    
    (* Compute final confidence region *)
    RegionComputation.compute_confidence_region regions t.alpha
end

(* Complete two-sided predictor *)
module CompleteTwoSidedPredictor = struct
  type t = {
    krr: KRR.t;
    alpha: float;
  }

  let create ?(alpha=0.05) krr = {
    krr;
    alpha;
  }

  (* Two-sided conformal prediction *)
  let compute_score t residual =
    let sorted = Utils.sort_tensor (Tensor.abs residual) in
    let n = Tensor.size residual 0 in
    
    (* Compute non-conformity scores *)
    let upper_idx = int_of_float (float_of_int n *. (1. -. t.alpha /. 2.)) in
    let lower_idx = int_of_float (float_of_int n *. (t.alpha /. 2.)) in
    
    let upper_score = Tensor.get_float1 sorted upper_idx in
    let lower_score = Tensor.get_float1 sorted lower_idx in
    
    RegionComputation.SingleInterval {
      lower = lower_score;
      upper = upper_score;
      coverage = 1. -. t.alpha;
      score = max (abs_float lower_score) (abs_float upper_score)
    }

  (* Predict region for new point *)
  let predict_region t x_test y_test =
    let residuals = KRR.compute_in_sample_residuals t.krr in
    let region = compute_score t residuals in
    
    match region with
    | RegionComputation.SingleInterval i ->
        let pred = KRR.predict t.krr (Tensor.unsqueeze x_test ~dim:0) in
        RegionComputation.SingleInterval {
          lower = Tensor.get_float2 pred 0 0 -. i.upper;
          upper = Tensor.get_float2 pred 0 0 +. i.upper;
          coverage = i.coverage;
          score = i.score
        }
    | _ -> region
end

(* Validation module for conformal predictions *)
module ConformalValidation = struct
  type validation_result = {
    empirical_coverage: float;
    average_width: float;
    confidence_interval: float * float;
  }

  (* Compute empirical coverage *)
  let compute_coverage predictions true_values =
    let n = Array.length predictions in
    let covered = ref 0 in
    let total_width = ref 0. in
    
    Array.iteri (fun i region ->
      match region with
      | RegionComputation.SingleInterval i ->
          let y = true_values.(i) in
          if y >= i.lower && y <= i.upper then
            incr covered;
          total_width := !total_width +. (i.upper -. i.lower)
      | RegionComputation.UnionIntervals intervals ->
          let y = true_values.(i) in
          if List.exists (fun i -> y >= i.lower && y <= i.upper) intervals then
            incr covered;
          total_width := !total_width +. 
            (List.fold_left (fun acc i -> acc +. (i.upper -. i.lower)) 0. intervals)
      | _ -> ()
    ) predictions;
    
    let coverage = float_of_int !covered /. float_of_int n in
    let width = !total_width /. float_of_int n in
    
    (* Compute confidence interval using normal approximation *)
    let std_err = sqrt (coverage *. (1. -. coverage) /. float_of_int n) in
    {
      empirical_coverage = coverage;
      average_width = width;
      confidence_interval = (
        coverage -. 1.96 *. std_err,
        coverage +. 1.96 *. std_err
      )
    }
end

(* Parameter tuning and optimization *)
module ParameterTuning = struct
  type parameter_space = {
    lambda_range: float list;
    kernel_params: float list array;
    kernel_types: Kernel.kernel_type list;
  }

  type tuning_config = {
    max_iter: int;
    tolerance: float;
    parallel: bool;
    num_threads: int;
  }

  type tuning_result = {
    best_lambda: float;
    best_kernel_type: Kernel.kernel_type;
    best_kernel_params: float array;
    best_score: float;
    convergence_path: float array;
  }

  (* Grid search *)
  let grid_search parameter_space config x_train y_train =
    let best_score = ref infinity in
    let best_params = ref None in
    let convergence = ref [] in

    let evaluate_params lambda kernel_type kernel_params =
      let kernel = Kernel.KernelImpl.create ~params:kernel_params kernel_type in
      let krr = KRR.create ~lambda ~kernel x_train y_train in
      let score = KRR.log_likelihood krr in
      convergence := score :: !convergence;
      if score < !best_score then begin
        best_score := score;
        best_params := Some (lambda, kernel_type, kernel_params)
      end in

    if config.parallel then
      ParallelProcessing.parallel_map ~f:(fun (lambda, kernel_type, kernel_params) ->
        evaluate_params lambda kernel_type kernel_params
      ) (List.flatten (
        List.map (fun lambda ->
          List.map (fun kernel_type ->
            List.map (fun kernel_params ->
              (lambda, kernel_type, kernel_params)
            ) (Utils.cartesian_product (Array.to_list parameter_space.kernel_params))
          ) parameter_space.kernel_types
        ) parameter_space.lambda_range
      )) ~num_threads:config.num_threads
    else
      List.iter (fun lambda ->
        List.iter (fun kernel_type ->
          List.iter (fun kernel_params ->
            evaluate_params lambda kernel_type kernel_params
          ) (Utils.cartesian_product (Array.to_list parameter_space.kernel_params))
        ) parameter_space.kernel_types
      ) parameter_space.lambda_range;

    match !best_params with
    | Some (lambda, kernel_type, kernel_params) ->
        {
          best_lambda = lambda;
          best_kernel_type = kernel_type;
          best_kernel_params = kernel_params;
          best_score = !best_score;
          convergence_path = Array.of_list (List.rev !convergence);
        }
    | None -> Error.raise_error (Error.ComputationError "No valid parameters found")
end
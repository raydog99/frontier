open Torch

type boundary_condition =
  | Dirichlet
  | Neumann 
  | Robin of float * float

type domain = {
  dimensions: int list;
  bounds: (float * float) list;
}

module FunctionSpace = struct
  type norm = L2Norm | H1Norm | H1_0Norm

  let compute_norm tensor = function
    | L2Norm ->
        Tensor.square tensor 
        |> Tensor.mean ~dim:[0] 
        |> Tensor.sqrt
    | H1Norm ->
        let derivative = numerical_derivative tensor in
        let l2_part = compute_norm tensor L2Norm in
        let der_part = compute_norm derivative L2Norm in
        Tensor.add l2_part der_part |> Tensor.sqrt
    | H1_0Norm ->
        let derivative = numerical_derivative tensor in
        compute_norm derivative L2Norm

  let numerical_derivative tensor =
    let n = Tensor.shape2_exn tensor |> snd in
    let h = 1. /. float_of_int (n - 1) in
    let central_diff = Tensor.(
      slice tensor ~dim:1 ~start:2 ~end_:n
      - slice tensor ~dim:1 ~start:0 ~end_:(n-2)
    ) |> Tensor.div_scalar (2. *. h) in
    
    (* Handle boundaries with one-sided differences *)
    let forward_diff = Tensor.(
      get tensor [|0; 1|] - get tensor [|0; 0|]
    ) |> fun x -> x /. h in
    let backward_diff = Tensor.(
      get tensor [|0; n-1|] - get tensor [|0; n-2|]
    ) |> fun x -> x /. h in
    
    Tensor.cat [
      Tensor.of_float1 [forward_diff];
      central_diff;
      Tensor.of_float1 [backward_diff];
    ] ~dim:0
end

module Transform = struct
  let dst input =
    let batch_size, n = Tensor.shape2_exn input in
    let extended = Tensor.new_zeros [batch_size; 2 * n] in
    (* Odd extension *)
    let _ = Tensor.slice_assign_ extended ~dim:1 ~start:0 ~end_:n input in
    let _ = Tensor.slice_assign_ extended ~dim:1 ~start:n ~end_:(2*n)
      (Tensor.flip (Tensor.neg input) ~dims:[1]) in
    let transformed = Tensor.fft extended ~normalized:true in
    Tensor.slice transformed ~dim:1 ~start:0 ~end_:n

  let dct input =
    let batch_size, n = Tensor.shape2_exn input in
    let extended = Tensor.new_zeros [batch_size; 2 * n] in
    (* Even extension *)
    let _ = Tensor.slice_assign_ extended ~dim:1 ~start:0 ~end_:n input in
    let _ = Tensor.slice_assign_ extended ~dim:1 ~start:n ~end_:(2*n)
      (Tensor.flip input ~dims:[1]) in
    let transformed = Tensor.fft extended ~normalized:true in
    Tensor.slice transformed ~dim:1 ~start:0 ~end_:n

  let idst = let open Tensor in function x -> 
    neg (dst (neg x))

  let idct = let open Tensor in function x ->
    dct x

  let normalize_basis_functions tensor domain =
    let norm = Tensor.sqrt (Tensor.ones [1] |> Tensor.mul_scalar 2.) in
    Tensor.div tensor norm

  let projection_filter tensor = function
    | Dirichlet ->
        let n = Tensor.shape2_exn tensor |> snd in
        let result = Tensor.clone tensor in
        Tensor.slice_assign_ result ~dim:1 ~start:0 ~end_:1 (Tensor.zeros [1]);
        Tensor.slice_assign_ result ~dim:1 ~start:(n-1) ~end_:n (Tensor.zeros [1]);
        result
    | Neumann ->
        let transformed = dct tensor in
        let filtered = transformed in (* Apply Neumann filtering *)
        idct filtered
    | Robin (a, b) ->
        let transformed = dct tensor in
        let filtered = transformed in (* Apply Robin filtering *)
        idct filtered
end

module SpectralOperator = struct
  type t = {
    transform_layer: transform_type;
    weight: Tensor.t;
    bias: Tensor.t;
    modes: int;
    width: int;
    n_dims: int;
    mode_selection: mode_selection;
  }
  
  and transform_type =
    | Fourier
    | Sine
    | Cosine

  type mode_selection = {
    max_modes: int;
    cutoff_threshold: float;
    adaptive: bool;
  }

  let create ?(mode_selection={max_modes=32; cutoff_threshold=1e-10; adaptive=true}) 
             ~width ~bc ~n_dims =
    let transform = match bc with
      | Dirichlet -> Sine
      | Neumann -> Cosine
      | Robin _ -> Fourier in
    let scale = 1. /. Float.sqrt (float_of_int width) in
    {
      transform_layer = transform;
      weight = Tensor.randn [n_dims; width; width; mode_selection.max_modes] ~scale;
      bias = Tensor.zeros [width];
      modes = mode_selection.max_modes;
      width;
      n_dims;
      mode_selection;
    }

  let enforce_boundary_conditions t x =
    match t.transform_layer with
    | Sine -> Transform.projection_filter x Dirichlet
    | Cosine -> Transform.projection_filter x Neumann
    | Fourier -> x

  let forward t x =
    let x = enforce_boundary_conditions t x in
    (* Apply spectral transform *)
    let x_transformed = match t.transform_layer with
      | Sine -> Transform.dst x
      | Cosine -> Transform.dct x
      | Fourier -> Tensor.fft x ~normalized:true in
    
    (* Truncate modes if adaptive *)
    let n_modes = if t.mode_selection.adaptive then
      min t.modes (Tensor.shape2_exn x_transformed |> snd)
    else t.modes in
    
    (* Apply weight in spectral domain *)
    let x_weighted = Tensor.(matmul x_transformed 
      (narrow t.weight ~dim:3 ~start:0 ~length:n_modes)) in
    
    (* Inverse transform *)
    let output = match t.transform_layer with
      | Sine -> Transform.idst x_weighted
      | Cosine -> Transform.idct x_weighted
      | Fourier -> Tensor.ifft x_weighted ~normalized:true in
    
    enforce_boundary_conditions t output
end

module SPFNO = struct
  type t = {
    operators: SpectralOperator.t list;
    width: int;
    bc: boundary_condition;
  }

  let create ~width ~depth ~modes ~bc =
    let operators = List.init depth (fun _ -> 
      SpectralOperator.create ~width ~bc ~n_dims:1 
        ~mode_selection:{max_modes=modes; cutoff_threshold=1e-10; adaptive=true}
    ) in
    { operators; width; bc }

  let forward t input =
    List.fold_left (fun acc op ->
      let x = SpectralOperator.forward op acc in
      Tensor.relu x
    ) input t.operators
end

module ErrorAnalysis = struct
  type error_metric = 
    | L2Error
    | H1Error 
    | MaxError
    | BoundaryError

  let compute_error ~predicted ~target = function
    | L2Error ->
        let diff = Tensor.sub predicted target in
        FunctionSpace.compute_norm diff FunctionSpace.L2Norm
        |> Tensor.to_float0_exn
    | H1Error ->
        let diff = Tensor.sub predicted target in
        FunctionSpace.compute_norm diff FunctionSpace.H1Norm
        |> Tensor.to_float0_exn
    | MaxError ->
        Tensor.sub predicted target
        |> Tensor.abs
        |> Tensor.max []
        |> Tensor.to_float0_exn
    | BoundaryError ->
        let n = Tensor.shape2_exn predicted |> snd in
        let pred_bounds = Tensor.cat [
          Tensor.slice predicted ~dim:1 ~start:0 ~end_:1;
          Tensor.slice predicted ~dim:1 ~start:(n-1) ~end_:n;
        ] ~dim:0 in
        let target_bounds = Tensor.cat [
          Tensor.slice target ~dim:1 ~start:0 ~end_:1;
          Tensor.slice target ~dim:1 ~start:(n-1) ~end_:n;
        ] ~dim:0 in
        compute_error ~predicted:pred_bounds ~target:target_bounds L2Error

  let analyze_stability model input perturbation =
    let output = SPFNO.forward model input in
    let perturbed_input = Tensor.add input perturbation in
    let perturbed_output = SPFNO.forward model perturbed_input in
    compute_error ~predicted:perturbed_output ~target:output L2Error
end

module HigherDim = struct
  type tensor_decomp = {
    core: Tensor.t;
    factors: Tensor.t list;
  }

  (* Helper functions for tensor operations *)
  let unfold_tensor tensor dim =
    let dims = Tensor.size tensor in
    let n = List.nth dims dim in
    let m = List.fold_left ( * ) 1 (List.filteri (fun i _ -> i != dim) dims) in
    Tensor.reshape tensor [n; m]

  let fold_tensor tensor orig_dims dim =
    let new_dims = List.mapi (fun i d -> 
      if i = dim then d else List.nth orig_dims i
    ) orig_dims in
    Tensor.reshape tensor new_dims

  let hosvd tensor max_rank =
    let dims = Tensor.size tensor in
    let n_dims = List.length dims in
    
    (* Compute factors for each dimension *)
    let factors = List.init n_dims (fun dim ->
      let unfolded = unfold_tensor tensor dim in
      let u, s, _ = Tensor.svd unfolded in
      Tensor.narrow u ~dim:1 ~start:0 ~length:max_rank
    ) in
    
    (* Compute core tensor through successive contractions *)
    let core = List.fold_left2 (fun acc factor dim ->
      let unfolded = unfold_tensor acc dim in
      let contracted = Tensor.matmul (Tensor.transpose factor ~dim0:0 ~dim1:1) unfolded in
      fold_tensor contracted dims dim
    ) tensor factors (List.init n_dims (fun i -> i)) in
    
    { core; factors }

  let tt_decomposition tensor epsilon =
    let dims = Tensor.size tensor in
    let n = List.hd dims in
    
    (* Reshape tensor into matrix *)
    let matrix = Tensor.reshape tensor [n; -1] in
    
    let rec decompose mat remaining_dims ranks =
      match remaining_dims with
      | [] -> []
      | d::ds ->
          let u, s, v = Tensor.svd mat in
          let rank = ref 0 in
          while !rank < Tensor.size s |> List.hd &&
                Tensor.get s [!rank] > epsilon do
            incr rank
          done;
          
          let core = Tensor.narrow u ~dim:1 ~start:0 ~length:!rank in
          let reshaped_core = Tensor.reshape core (d :: !rank :: []) in
          
          reshaped_core :: decompose 
            (Tensor.matmul (Tensor.diag s) (Tensor.transpose v ~dim0:0 ~dim1:1))
            ds
            (!rank :: ranks)
    in
    decompose matrix (List.tl dims) [1]

  let fft_nd tensor =
    let dims = Tensor.size tensor in
    List.fold_left (fun acc dim ->
      Tensor.fft acc ~dim ~normalized:true
    ) tensor (List.init (List.length dims) (fun i -> i))
end

module MultiDimStability = struct
  type stability_measure = {
    condition_number: float;
    spectral_radius: float;
    energy_ratio: float;
    max_eigenvalue: float;
  }

  let compute_fourier_symbol tensor dim dt dx =
    let n = List.nth (Tensor.size tensor) dim in
    let k = Tensor.arange n ~start:0 ~step:1 in
    let symbol = Tensor.(mul_scalar (sin (mul_scalar k (Float.pi /. float_of_int n))) 
                        (2. *. dx /. dt)) in
    Tensor.square symbol

  let von_neumann_analysis tensor dt dx =
    let dims = Tensor.size tensor in
    let symbols = List.mapi (fun dim _ ->
      compute_fourier_symbol tensor dim dt dx
    ) dims in
    
    let max_symbol = List.fold_left (fun acc symbol ->
      Tensor.max2 acc symbol |> fst
    ) (List.hd symbols) (List.tl symbols) in
    
    Tensor.to_float0_exn max_symbol

  let check_cfl coeffs dx dt =
    let max_coeff = Tensor.max coeffs [] |> Tensor.to_float0_exn in
    let cfl_number = max_coeff *. dt /. (dx *. dx) in
    cfl_number <= 1.0

  let analyze_energy_stability model input =
    let output = SPFNO.forward model input in
    let energy_in = Tensor.sum (Tensor.square input) [] |> Tensor.to_float0_exn in
    let energy_out = Tensor.sum (Tensor.square output) [] |> Tensor.to_float0_exn in
    
    let condition_number = 
      let jac = compute_jacobian model input in
      let s = Tensor.svd jac |> fun (_, s, _) -> s in
      let s_max = Tensor.max s [] |> Tensor.to_float0_exn in
      let s_min = Tensor.min s [] |> Tensor.to_float0_exn in
      s_max /. s_min in
    
    {
      condition_number;
      spectral_radius = energy_out /. energy_in;
      energy_ratio = energy_out /. energy_in;
      max_eigenvalue = condition_number;
    }
end

module AdaptiveRefinement = struct
  type refinement_criterion =
    | EnergyBased of float
    | ErrorBased of float
    | HybridCriterion of float * float

  let select_modes tensor = function
    | EnergyBased threshold ->
        let spectrum = Tensor.fft tensor ~normalized:true in
        let energy = Tensor.square spectrum |> Tensor.sum ~dim:[1] in
        let total_energy = Tensor.sum energy [] |> Tensor.to_float0_exn in
        let cumsum = Tensor.cumsum energy ~dim:0 in
        let n = ref 0 in
        while !n < Tensor.size cumsum |> List.hd &&
              Tensor.get cumsum [!n] /. total_energy < threshold do
          incr n
        done;
        !n
    | ErrorBased threshold ->
        let error = ErrorAnalysis.compute_error 
          ~predicted:tensor 
          ~target:(Tensor.zeros_like tensor) 
          ErrorAnalysis.L2Error in
        int_of_float (Float.ceil (1. /. threshold *. error))
    | HybridCriterion (energy_thresh, error_thresh) ->
        min 
          (select_modes tensor (EnergyBased energy_thresh))
          (select_modes tensor (ErrorBased error_thresh))

  let refine_mesh tensor error_indicator =
    let threshold = Tensor.mean error_indicator [] |> Tensor.to_float0_exn in
    let mask = Tensor.gt error_indicator (Tensor.full_like error_indicator threshold) in
    let refined = Tensor.masked_select tensor mask in
    let n = Tensor.size refined |> List.hd in
    let new_points = Tensor.linspace ~start:0. ~end_:1. (2 * n) in
    Tensor.interpolate tensor new_points
end
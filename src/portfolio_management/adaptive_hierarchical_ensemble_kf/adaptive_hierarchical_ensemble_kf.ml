open Torch

(* Stable inverse using SVD *)
let stable_inverse ?(rcond=1e-15) mat =
  let u, s, v = Tensor.svd mat in
  let s_inv = Tensor.reciprocal s in
  (* Zero out small singular values *)
  let mask = Tensor.gt s (Tensor.mul_scalar s rcond) in
  let s_inv = Tensor.where mask s_inv (Tensor.zeros_like s_inv) in
  Tensor.mm (Tensor.mm v (Tensor.diag s_inv)) (Tensor.transpose u)

(* Matrix square root via Cholesky *)
let matrix_sqrt mat =
  let n = Tensor.size mat 0 in
  let jitter = Tensor.mul_scalar (Tensor.eye n) 1e-12 in
  let reg_mat = Tensor.add mat jitter in
  Tensor.cholesky reg_mat

(* Solve linear system with regularization *)
let solve ?(alpha=1e-10) a b =
  let m = Tensor.size a 0 in
  let augmented = Tensor.cat 
    [a; Tensor.mul_scalar (Tensor.eye m) (sqrt alpha)] 1 in
  let augmented_b = Tensor.cat 
    [b; Tensor.zeros [m; Tensor.size b 1]] 0 in
  Tensor.lstsq augmented augmented_b

(* Core state space operations *)
module StateSpace = struct
  type t = {
    dim: int;                (* State dimension *)
    inner_product_op: Tensor.t option;  (* Optional inner product operator *)
  }

  (* Create state space *)
  let create ?(inner_product_op=None) dim =
    { dim; inner_product_op }

  (* Inner product computation *)
  let inner_product space x y =
    match space.inner_product_op with
    | None -> Tensor.dot x y |> Tensor.float_value
    | Some op -> 
        Tensor.mm (Tensor.mm (Tensor.reshape x [1; -1]) op) 
          (Tensor.reshape y [-1; 1])
        |> Tensor.float_value

  (* Norm computation *)
  let norm space x =
    sqrt (inner_product space x x)

  (* Orthogonalize vector against basis *)
  let orthogonalize space v basis =
    Array.fold_left (fun acc b ->
      let proj = inner_product space b v in
      Tensor.sub acc (Tensor.mul_scalar b proj)
    ) v basis

  (* Project vector onto subspace *)
  let project space v basis =
    let n = Array.length basis in
    let coeffs = Array.init n (fun i ->
      inner_product space basis.(i) v
    ) in
    Array.fold_left2 (fun acc coef b ->
      Tensor.add acc (Tensor.mul_scalar b coef)
    ) (Tensor.zeros [space.dim]) coeffs basis
end

(* Ensemble operations *)
module Ensemble = struct
  type t = {
    members: Tensor.t array;
    space: StateSpace.t;
    weights: float array option;
  }

  (* Create ensemble *)
  let create ?(weights=None) members space =
    { members; space; weights }

  (* Compute weighted ensemble mean *)
  let mean ensemble =
    match ensemble.weights with
    | None ->
        let n = float_of_int (Array.length ensemble.members) in
        Array.fold_left Tensor.add 
          (Tensor.zeros [ensemble.space.dim]) ensemble.members
        |> fun x -> Tensor.div_scalar x n
    | Some w ->
        Array.fold_left2 (fun acc x w' ->
          Tensor.add acc (Tensor.mul_scalar x w')
        ) (Tensor.zeros [ensemble.space.dim]) ensemble.members w

  (* Compute ensemble covariance *)
  let covariance ensemble =
    let mu = mean ensemble in
    let centered = Array.map (fun x -> Tensor.sub x mu) ensemble.members in
    
    let weights = match ensemble.weights with
    | None -> Array.make (Array.length ensemble.members) 
        (1. /. float_of_int (Array.length ensemble.members - 1))
    | Some w -> Array.map (fun x -> x /. 
        (1. -. Array.fold_left (+.) 0. w)) w
    in
    
    Array.fold_left2 (fun acc x w ->
      let outer = Tensor.mm (Tensor.reshape x [-1; 1]) 
        (Tensor.reshape x [1; -1]) in
      Tensor.add acc (Tensor.mul_scalar outer w)
    ) (Tensor.zeros [ensemble.space.dim; ensemble.space.dim]) 
      centered weights

  (* Compute correlation between ensembles *)
  let correlation e1 e2 =
    let cov = Array.map2 (fun x1 x2 ->
      let outer = Tensor.mm 
        (Tensor.reshape x1 [-1; 1])
        (Tensor.reshape x2 [1; -1]) in
      outer
    ) e1.members e2.members
    |> Array.fold_left Tensor.add 
      (Tensor.zeros [e1.space.dim; e2.space.dim]) in
    
    let norm1 = sqrt (StateSpace.inner_product e1.space 
      (mean e1) (mean e1)) in
    let norm2 = sqrt (StateSpace.inner_product e2.space
      (mean e2) (mean e2)) in
    
    Tensor.div_scalar cov (norm1 *. norm2)

  (* Split ensemble *)
  let split ensemble =
    let n = Array.length ensemble.members in
    let n1 = n / 2 in
    let n2 = n - n1 in
    
    let members1 = Array.sub ensemble.members 0 n1 in
    let members2 = Array.sub ensemble.members n1 n2 in
    
    let weights1 = match ensemble.weights with
    | None -> None
    | Some w -> Some (Array.sub w 0 n1)
    in
    let weights2 = match ensemble.weights with
    | None -> None
    | Some w -> Some (Array.sub w n1 n2)
    in
    
    create ~weights:weights1 members1 ensemble.space,
    create ~weights:weights2 members2 ensemble.space

  (* Sample from ensemble *)
  let sample ensemble n =
    let idx = Array.init n (fun _ -> 
      Random.int (Array.length ensemble.members)) in
    Array.map (fun i -> Array.get ensemble.members i) idx

  (* Project ensemble onto basis *)
  let project_to_basis ensemble basis =
    let projected_members = Array.map (fun x ->
      StateSpace.project ensemble.space x basis
    ) ensemble.members in
    create ?weights:ensemble.weights projected_members ensemble.space
end

(* POD and Reduced Basis *)
module POD = struct
  type t = {
    basis: Tensor.t;
    singular_values: Tensor.t;
    energy: float array;
    mean: Tensor.t;
  }

  (* Compute POD from snapshot matrix *)
  let compute ?(center=true) ?(eps=1e-10) snapshots =
    let n_snapshots = Array.length snapshots in
    
    (* Center snapshots if requested *)
    let mean = if center then
      Array.fold_left Tensor.add 
        (Tensor.zeros_like snapshots.(0)) snapshots
      |> fun x -> Tensor.div_scalar x (float_of_int n_snapshots)
    else
      Tensor.zeros_like snapshots.(0)
    in
    
    let centered = if center then
      Array.map (fun x -> Tensor.sub x mean) snapshots
    else
      snapshots
    in
    
    (* Build snapshot matrix *)
    let snapshot_matrix = Tensor.stack (Array.to_list centered) 1 in
    
    (* Compute SVD *)
    let u, s, _ = Tensor.svd snapshot_matrix in
    
    (* Compute energy distribution *)
    let total_energy = Tensor.sum s |> Tensor.float_value in
    let energy = Array.init (Tensor.size s 0) (fun i ->
      let partial = Tensor.narrow s 0 0 (i + 1) in
      Tensor.sum partial |> Tensor.float_value |> fun x -> x /. total_energy
    ) in
    
    { basis = u; singular_values = s; energy; mean }

  (* Determine number of modes for energy threshold *)
  let number_of_modes pod threshold =
    let rec find_index i =
      if i >= Array.length pod.energy then i
      else if pod.energy.(i) >= threshold then i + 1
      else find_index (i + 1)
    in
    find_index 0

  (* Project state onto POD basis *)
  let project pod state n_modes =
    let basis = Tensor.narrow pod.basis 1 0 n_modes in
    let centered = Tensor.sub state pod.mean in
    let coeffs = Tensor.mm (Tensor.transpose basis) 
      (Tensor.reshape centered [-1; 1]) in
    Tensor.mm basis coeffs |> Tensor.squeeze |> fun x ->
    Tensor.add x pod.mean

  (* Reconstruct state from coefficients *)
  let reconstruct pod coeffs =
    let n_modes = Tensor.size coeffs 0 in
    let basis = Tensor.narrow pod.basis 1 0 n_modes in
    Tensor.mm basis (Tensor.reshape coeffs [-1; 1])
    |> Tensor.squeeze |> fun x ->
    Tensor.add x pod.mean
end

(* Reduced basis *)
module ReducedBasis = struct
  type t = {
    pod: POD.t;
    n_modes: int;
    space: StateSpace.t;
  }

  (* Create reduced basis *)
  let create ?(energy_threshold=0.99) snapshots space =
    let pod = POD.compute ~center:true snapshots in
    let n_modes = POD.number_of_modes pod energy_threshold in
    { pod; n_modes; space }

  (* Create from existing matrices *)
  let create_from_matrices u s v =
    let pod = {
      POD.basis = u;
      singular_values = s;
      energy = Array.init (Tensor.size s 0) (fun i ->
        let partial = Tensor.narrow s 0 0 (i + 1) in
        Tensor.sum partial |> Tensor.float_value
      );
      mean = Tensor.zeros [Tensor.size u 0];
    } in
    {
      pod;
      n_modes = Tensor.size u 1;
      space = StateSpace.create (Tensor.size u 0);
    }

  (* Project state *)
  let project rb state =
    POD.project rb.pod state rb.n_modes

  (* Reconstruct state *)
  let reconstruct rb coeffs =
    POD.reconstruct rb.pod coeffs

  (* Extend basis *)
  let extend rb new_size =
    let current_size = rb.n_modes in
    if new_size <= current_size then rb
    else
      let extended_basis = Tensor.narrow rb.pod.basis 1 0 new_size in
      create_from_matrices extended_basis rb.pod.singular_values
        (Tensor.zeros [new_size; new_size])

  (* Truncate basis *)
  let truncate rb new_size =
    let current_size = rb.n_modes in
    if new_size >= current_size then rb
    else
      let truncated_basis = Tensor.narrow rb.pod.basis 1 0 new_size in
      create_from_matrices truncated_basis
        (Tensor.narrow rb.pod.singular_values 0 0 new_size)
        (Tensor.zeros [new_size; new_size])
end

(* Error estimation *)
module ErrorEstimation = struct
  type error_component = {
    local: float;
    global: float;
    propagation: float;
    correlation: float;
  }

  type error_bound = {
    estimate: float;
    upper_bound: float;
    confidence: float;
  }

  type error_history = {
    times: float array;
    errors: error_component array;
    trends: float array;
  }

  (* Compute error components *)
  let analyze_error ~high_fidelity ~low_fidelity ~reference =
    let local_error = Tensor.sub high_fidelity low_fidelity
      |> Tensor.norm |> Tensor.float_value in
    
    let global_error = Tensor.sub reference low_fidelity
      |> Tensor.norm |> Tensor.float_value in
    
    let prop_error = Tensor.sub reference high_fidelity
      |> Tensor.norm |> Tensor.float_value in
    
    let corr = Tensor.mm 
      (Tensor.reshape (Tensor.sub high_fidelity reference) [1; -1])
      (Tensor.reshape (Tensor.sub low_fidelity reference) [-1; 1])
      |> Tensor.float_value in
    
    {
      local = local_error;
      global = global_error;
      propagation = prop_error;
      correlation = corr;
    }

  (* Track error history *)
  let track_errors errors time =
    let n = Array.length errors in
    let times = Array.init n (fun i -> 
      time +. float_of_int i *. 0.1) in
    
    (* Compute error trends *)
    let trends = Array.init (n-1) (fun i ->
      let e1 = errors.(i) in
      let e2 = errors.(i+1) in
      (e2.global -. e1.global) /. (times.(i+1) -. times.(i))
    ) in
    
    { times; errors; trends = Array.append trends [|0.0|] }

  (* Analyze stability *)
  let analyze_stability errors =
    let n = Array.length errors in
    let stable = ref true in
    let growth_rate = ref 0.0 in
    
    (* Check for exponential growth *)
    for i = 0 to n-2 do
      let ratio = errors.(i+1).global /. errors.(i).global in
      if ratio > 1.1 then begin
        stable := false;
        growth_rate := max !growth_rate (log ratio)
      end
    done;
    
    !stable, !growth_rate

  (* Error-based adaptivity criteria *)
  module AdaptivityCriteria = struct
    type criterion = 
      | ErrorThreshold of float
      | RelativeReduction of float
      | GradientBased of float
      | Combined of criterion list

    (* Check adaptation criterion *)
    let check_criterion error_stats criterion =
      match criterion with
      | ErrorThreshold thresh ->
          error_stats.estimate > thresh
      | RelativeReduction target ->
          let reduction = (error_stats.upper_bound -. error_stats.estimate) 
            /. error_stats.upper_bound in
          reduction < target
      | GradientBased thresh ->
          let gradient = Array.map (fun err -> err.propagation) 
            error_stats.errors in
          Array.exists (fun g -> abs_float g > thresh) gradient
      | Combined criteria ->
          List.exists (check_criterion error_stats) criteria
  end
end

(* Multi-level covariance estimation *)
module MultiLevelCovariance = struct
  type level_weights = {
    principal: float;
    control: float;
    ancillary: float;
    cross_correlation: float;
  }

  type covariance_estimate = {
    mean: Tensor.t;
    covariance: Tensor.t;
    cross_terms: Tensor.t array;
    condition_number: float;
    rank: int;
  }

  (* Create level weights *)
  let create_weights ~np ~nc ~na =
    let total = float_of_int (np + nc + na) in
    {
      principal = float_of_int np /. total;
      control = float_of_int nc /. total;
      ancillary = float_of_int na /. total;
      cross_correlation = 2.0 *. sqrt (float_of_int (np * nc)) /. total;
    }

  (* Cross-covariance computation *)
  let cross_covariance ensemble1 ensemble2 =
    let n1 = Array.length ensemble1 in
    let n2 = Array.length ensemble2 in
    assert (n1 = n2);
    
    let mean1 = Ensemble.mean 
      { Ensemble.members = ensemble1; 
        space = ensemble1.(0).space; weights = None } in
    let mean2 = Ensemble.mean 
      { Ensemble.members = ensemble2;
        space = ensemble2.(0).space; weights = None } in
    
    let cov = Array.mapi (fun i x1 ->
      let x2 = ensemble2.(i) in
      let diff1 = Tensor.sub x1 mean1 in
      let diff2 = Tensor.sub x2 mean2 in
      Tensor.mm (Tensor.reshape diff1 [-1; 1]) 
        (Tensor.reshape diff2 [1; -1])
    ) ensemble1 |> Array.fold_left Tensor.add
      (Tensor.zeros [Tensor.size ensemble1.(0) 0; 
                    Tensor.size ensemble2.(0) 0]) in
    
    Tensor.div_scalar cov (float_of_int (n1 - 1))

  (* Multi-level covariance estimation *)
  let estimate ~principal ~control ~ancillary weights =
    (* Single level covariances *)
    let pp_cov = Ensemble.covariance
      { Ensemble.members = principal;
        space = principal.(0).space; weights = None } in
    let cc_cov = Ensemble.covariance 
      { Ensemble.members = control;
        space = control.(0).space; weights = None } in
    let aa_cov = Ensemble.covariance
      { Ensemble.members = ancillary;
        space = ancillary.(0).space; weights = None } in
    
    (* Cross covariances *)
    let pc_cov = cross_covariance principal control in
    let cp_cov = Tensor.transpose pc_cov in
    
    (* Combined estimate *)
    let combined = Tensor.add
      (Tensor.mul_scalar pp_cov weights.principal)
      (Tensor.add
        (Tensor.mul_scalar cc_cov weights.control)
        (Tensor.add
          (Tensor.mul_scalar aa_cov weights.ancillary)
          (Tensor.add
            (Tensor.mul_scalar pc_cov weights.cross_correlation)
            (Tensor.mul_scalar cp_cov weights.cross_correlation)))) in
    
    (* Analyze estimate *)
    let u, s, _ = Tensor.svd combined in
    let rank = Tensor.sum (Tensor.gt s 1e-10) |> Tensor.int_value in
    let condition_number = 
      let max_s = Tensor.max s |> Tensor.float_value in
      let min_s = Tensor.min s |> Tensor.float_value in
      if min_s < 1e-10 then infinity else max_s /. min_s in
    
    {
      mean = Ensemble.mean 
        { Ensemble.members = principal;
          space = principal.(0).space; weights = None };
      covariance = combined;
      cross_terms = [|pc_cov; cp_cov|];
      condition_number;
      rank;
    }
end

(* Advanced regularization techniques *)
module AdvancedRegularization = struct
  type regularization_method =
    | Tikhonov of float
    | Spectrum of float
    | LocalizedTapering of float
    | AdaptiveShrinkage of float

  type regularization_stats = {
    original_condition: float;
    regularized_condition: float;
    rank_reduction: int;
    frobenius_change: float;
  }

  (* Tikhonov regularization *)
  let tikhonov_regularize matrix alpha =
    let n = Tensor.size matrix 0 in
    let reg_term = Tensor.mul_scalar (Tensor.eye n) alpha in
    Tensor.add matrix reg_term

  (* Spectral regularization *)
  let spectral_regularize matrix threshold =
    let u, s, v = Tensor.svd matrix in
    let s_reg = Tensor.max_values s (Tensor.full_like s threshold) in
    Tensor.mm (Tensor.mm u (Tensor.diag s_reg)) (Tensor.transpose v)

  (* Localized tapering *)
  let localized_taper matrix length_scale =
    let n = Tensor.size matrix 0 in
    let taper = Tensor.zeros [n; n] in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let dist = float_of_int (abs (i - j)) in
        let factor = exp (-. (dist *. dist) /. 
          (2.0 *. length_scale *. length_scale)) in
        Tensor.set taper [|i; j|] factor
      done
    done;
    
    Tensor.mul matrix taper

  (* Apply regularization with statistics *)
  let regularize matrix method_ =
    let original_stats = {
      original_condition = 
        let u, s, _ = Tensor.svd matrix in
        let max_s = Tensor.max s |> Tensor.float_value in
        let min_s = Tensor.min s |> Tensor.float_value in
        if min_s < 1e-10 then infinity else max_s /. min_s;
      regularized_condition = 0.0;
      rank_reduction = 0;
      frobenius_change = 0.0;
    } in
    
    let regularized = match method_ with
    | Tikhonov alpha -> tikhonov_regularize matrix alpha
    | Spectrum thresh -> spectral_regularize matrix thresh
    | LocalizedTapering scale -> localized_taper matrix scale
    | AdaptiveShrinkage intensity ->
        let u, s, v = Tensor.svd matrix in
        let n = Tensor.size s 0 in
        
        let mean_s = Tensor.mean s |> Tensor.float_value in
        let var_s = Tensor.var s |> Tensor.float_value in
        
        let shrinkage = Array.init n (fun i ->
          let si = Tensor.get s [|i|] in
          let t = 1.0 -. intensity *. var_s /. 
            ((si -. mean_s) ** 2.0 +. var_s) in
          max 0.0 (min 1.0 t)
        ) in
        
        let s_shrunk = Tensor.of_float_array shrinkage in
        Tensor.mm (Tensor.mm u (Tensor.diag s_shrunk)) 
          (Tensor.transpose v)
    in
    
    (* Compute statistics *)
    let u_reg, s_reg, _ = Tensor.svd regularized in
    let rank_orig = Tensor.sum (Tensor.gt s_reg 1e-10) 
      |> Tensor.int_value in
    let rank_reg = Tensor.sum (Tensor.gt s_reg 1e-10) 
      |> Tensor.int_value in
    
    let frob_change = 
      let diff = Tensor.sub matrix regularized in
      let norm_diff = Tensor.norm diff |> Tensor.float_value in
      let norm_orig = Tensor.norm matrix |> Tensor.float_value in
      norm_diff /. norm_orig
    in
    
    let reg_cond = 
      let max_s = Tensor.max s_reg |> Tensor.float_value in
      let min_s = Tensor.min s_reg |> Tensor.float_value in
      if min_s < 1e-10 then infinity else max_s /. min_s
    in
    
    let stats = {
      original_condition = original_stats.original_condition;
      regularized_condition = reg_cond;
      rank_reduction = rank_orig - rank_reg;
      frobenius_change = frob_change;
    } in
    
    regularized, stats
end

(* Memory-efficient adaptation *)
module EfficientAdaptation = struct
  type memory_arena = {
    max_total: int;       (* Maximum total memory in bytes *)
    max_per_level: int;   (* Maximum memory per level *)
    min_required: int;    (* Minimum required memory *)
  }

  type resource_usage = {
    active_memory: int;
    cached_memory: int;
    temporary_memory: int;
  }

  (* Memory pool for efficient reuse *)
  module MemoryPool = struct
    type buffer = {
      data: Tensor.t;
      size: int;
      last_use: float;
    }

    type t = {
      active: buffer list;
      cached: buffer list;
      total_size: int;
    }

    (* Create pool *)
    let create budget =
      { 
        active = [];
        cached = [];
        total_size = 0;
      }

    (* Get memory usage *)
    let get_usage () =
      let open Unix in
      let stats = Gc.stat () in
      {
        active_memory = stats.minor_words |> int_of_float;
        cached_memory = stats.major_words |> int_of_float;
        temporary_memory = stats.promoted_words |> int_of_float;
      }

    (* Clean old buffers *)
    let cleanup pool =
      let now = Unix.gettimeofday () in
      let new_cached = List.filter
        (fun buf -> now -. buf.last_use < 300.0)
        pool.cached in
      { pool with cached = new_cached }
  end

  (* Adapt basis efficiently *)
  let adapt_basis rb_space ensemble budget =
    let dim = Tensor.size (Array.get ensemble 0) 0 in
    let n_samples = Array.length ensemble in
    
    (* Memory-efficient SVD *)
    let compute_efficient_svd data max_rank =
      let n = Tensor.size data 1 in
      let block_size = min 100 n in
      let n_blocks = (n + block_size - 1) / block_size in
      
      let q = ref (Tensor.zeros [dim; 0]) in
      
      for i = 0 to n_blocks - 1 do
        (* Process block *)
        let start_idx = i * block_size in
        let end_idx = min ((i + 1) * block_size) n in
        let block = Tensor.narrow data 1 start_idx (end_idx - start_idx) in
        
        (* Orthogonalize against existing Q *)
        if i > 0 then begin
          let proj = Tensor.mm 
            (Tensor.mm (Tensor.transpose !q) block)
            (Tensor.transpose !q) in
          block <- Tensor.sub block proj
        end;
        
        (* Local SVD *)
        let u, _, _ = Tensor.svd block in
        
        (* Update Q *)
        q := Tensor.cat [!q; u] 1
      done;
      
      (* Final SVD on reduced matrix *)
      let r = Tensor.mm (Tensor.transpose !q) data in
      let u_final, s_final, v_final = Tensor.svd r in
      
      let u = Tensor.mm !q u_final in
      Tensor.narrow u 1 0 max_rank,
      Tensor.narrow s_final 0 0 max_rank,
      Tensor.narrow v_final 1 0 max_rank
    in
    
    (* Build snapshot matrix efficiently *)
    let snapshots = ref (Tensor.zeros [dim; 0]) in
    for i = 0 to n_samples - 1 do
      snapshots := Tensor.cat 
        [!snapshots; Tensor.reshape ensemble.(i) [-1; 1]] 1
    done;
    
    (* Compute efficient POD *)
    let max_rank = min (budget.max_per_level / (dim * 8)) n_samples in
    let u, s, v = compute_efficient_svd !snapshots max_rank in
    
    (* Create new reduced basis *)
    ReducedBasis.create_from_matrices u s v
end

(* Adaptive hierarchical EnKF *)
module AdaptiveHierarchicalEnKF = struct
  type config = {
    (* Filter configuration *)
    dim: int;
    max_levels: int;
    base_ensemble_size: int;
    initial_basis_size: int;
    
    (* Adaptation parameters *)
    error_threshold: float;
    correlation_threshold: float;
    efficiency_threshold: float;
    
    (* Memory configuration *)
    memory_arena: EfficientAdaptation.memory_arena;
    
    (* Regularization *)
    regularization: AdvancedRegularization.regularization_method;
  }

  type level = {
    id: int;
    basis: ReducedBasis.t;
    ensemble: Ensemble.t;
    parent: level option;
    children: level list;
    error_history: ErrorAnalysis.error_history option;
  }

  type t = {
    config: config;
    levels: level array;
    covariance: MultiLevelCovariance.covariance_estimate;
    time: float;
    step: int;
  }

  (* Create initial filter state *)
  let create config initial_state =
    (* Create initial basis *)
    let initial_basis = ReducedBasis.create
      ~energy_threshold:0.99
      [|initial_state|]
      (StateSpace.create config.dim) in

    (* Initialize ensemble *)
    let ensemble_config = {
      InitialSampling.n_samples = config.base_ensemble_size;
      mean = initial_state;
      std = 0.1;
      method_ = `Normal;
    } in
    let initial_ensemble = InitialSampling.create_ensemble
      ensemble_config initial_basis.space in

    (* Create initial level *)
    let initial_level = {
      id = 0;
      basis = initial_basis;
      ensemble = initial_ensemble;
      parent = None;
      children = [];
      error_history = None;
    } in

    let initial_covariance = MultiLevelCovariance.estimate
      ~principal:initial_ensemble.members
      ~control:[|initial_state|]
      ~ancillary:[|initial_state|]
      (MultiLevelCovariance.create_weights 
        ~np:config.base_ensemble_size ~nc:1 ~na:1) in

    {
      config;
      levels = [|initial_level|];
      covariance = initial_covariance;
      time = 0.0;
      step = 0;
    }

  (* Analysis step *)
  let analysis_step state observation =
    (* Compute multi-level covariances *)
    let weights = Array.map (fun level ->
      let n = Array.length level.ensemble.members in
      MultiLevelCovariance.create_weights
        ~np:n
        ~nc:(n/2)
        ~na:(max 1 (n/4))
    ) state.levels in

    let updated_levels = Array.mapi (fun i level ->
      (* Level-specific analysis *)
      let cov = MultiLevelCovariance.estimate
        ~principal:level.ensemble.members
        ~control:(Array.sub level.ensemble.members 0 
          (Array.length level.ensemble.members / 2))
        ~ancillary:(Array.sub level.ensemble.members 0 
          (max 1 (Array.length level.ensemble.members / 4)))
        weights.(i) in

      (* Regularize covariance *)
      let reg_cov, stats = AdvancedRegularization.regularize
        cov.covariance state.config.regularization in

      (* Kalman update *)
      let kalman_gain = stable_inverse 
        (Tensor.add reg_cov observation.noise_cov) in
      
      let updated_ensemble = Array.map (fun member ->
        let innovation = Tensor.sub observation.data member in
        Tensor.add member (Tensor.mv kalman_gain innovation)
      ) level.ensemble.members in

      { level with 
        ensemble = { level.ensemble with members = updated_ensemble } }
    ) state.levels in

    { state with levels = updated_levels }

  (* Prediction step *)
  let prediction_step state model dt =
    (* Propagate ensembles *)
    let propagated_levels = Array.map (fun level ->
      let propagated_ensemble = Array.map (fun member ->
        let reduced = ReducedBasis.project level.basis member in
        let evolved = model reduced dt in
        ReducedBasis.reconstruct level.basis evolved
      ) level.ensemble.members in

      { level with 
        ensemble = { level.ensemble with members = propagated_ensemble } }
    ) state.levels in

    (* Adapt bases *)
    let adapted_levels = Array.mapi (fun i level ->
      let error_stats = ErrorAnalysis.analyze_error
        ~high_fidelity:(Array.get level.ensemble.members 0)
        ~low_fidelity:(ReducedBasis.project level.basis 
          (Array.get level.ensemble.members 0))
        ~reference:(Array.get state.levels.(0).ensemble.members 0) in

      let adapted_basis = if error_stats.global > state.config.error_threshold
      then
        EfficientAdaptation.adapt_basis
          level.basis level.ensemble state.config.memory_arena
      else level.basis in

      { level with basis = adapted_basis }
    ) propagated_levels in

    (* Update error histories *)
    let updated_levels = Array.mapi (fun i level ->
      let error_stats = ErrorAnalysis.analyze_error
        ~high_fidelity:(Array.get level.ensemble.members 0)
        ~low_fidelity:(ReducedBasis.project level.basis 
          (Array.get level.ensemble.members 0))
        ~reference:(Array.get state.levels.(0).ensemble.members 0) in

      let history = match level.error_history with
      | None -> 
          ErrorAnalysis.track_errors [|error_stats|] state.time
      | Some h ->
          ErrorAnalysis.track_errors 
            (Array.append h.errors [|error_stats|])
            state.time
      in

      { level with error_history = Some history }
    ) adapted_levels in

    { state with 
      levels = updated_levels;
      time = state.time +. dt;
      step = state.step + 1 }

  (* Complete filter step *)
  let step state observation model dt =
    let post_analysis = analysis_step state observation in
    let post_prediction = prediction_step post_analysis model dt in
    
    (* Check stability *)
    let stable = Array.for_all (fun level ->
      match level.error_history with
      | None -> true
      | Some h -> 
          let stable, _ = ErrorAnalysis.analyze_stability h.errors in
          stable
    ) post_prediction.levels in
    post_prediction
end
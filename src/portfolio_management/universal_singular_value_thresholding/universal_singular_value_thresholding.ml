open Torch

(* Core matrix operations *)
module Matrix = struct
  type t = Tensor.t

  let create m n = Tensor.zeros [m; n]

  let dims t =
    let shape = Tensor.shape t in
    match shape with
    | [m; n] -> (m, n)
    | _ -> failwith "Invalid matrix dimensions"

  let get t i j = Tensor.get t [i; j]

  let set t i j v =
    let t' = Tensor.copy t in
    Tensor.set t' [i; j] v;
    t'

  let matmul a b = Tensor.mm a b

  let transpose t = Tensor.transpose t ~dim0:0 ~dim1:1

  let is_symmetric t =
    let t' = transpose t in
    Tensor.equal t t'

  let is_skew_symmetric t =
    let t' = transpose t in
    Tensor.equal t (Tensor.neg t')

  let of_lists lists =
    let arr = Array.of_list (List.map Array.of_list lists) in
    Tensor.of_float_array2 arr

  let to_float_array2 t =
    Tensor.to_float_array2 t
end

(* Handling observed matrices with missing entries *)
module ObservedMatrix = struct
  type t = {
    data: Matrix.t;  (* Observed entries *)
    mask: Matrix.t;  (* 1.0 for observed, 0.0 for missing *)
  }

  let create data mask = { data; mask }

  let dims t = Matrix.dims t.data

  let observed_proportion t =
    let total = float_of_int (Tensor.numel t.mask) in
    let observed_sum = Tensor.sum t.mask |> Tensor.float_value in
    observed_sum /. total

  let get t i j =
    if Tensor.get t.mask [i; j] > 0.5 then
      Some (Matrix.get t.data i j)
    else
      None

  let to_dense t = t.data

  let get_mask t = t.mask

  let create_random_mask m n p =
    let mask = Matrix.create m n in
    let rand = Tensor.rand [m; n] in
    Tensor.lt_ rand (Tensor.float_scalar p)
    |> Tensor.to_type ~dtype:(Tensor.kind rand)
end

(* SVD and numerical stability *)
module NumericalStability = struct
  type svd_result = {
    u: Matrix.t;
    s: Matrix.t;  (* Diagonal matrix of singular values *)
    vt: Matrix.t;
  }

  let eps = 1e-10

  let stable_svd matrix ~epsilon =
    let u, s, v = Tensor.svd matrix in
    let s' = Tensor.where s (Tensor.gt s epsilon) epsilon in
    (u, s', v)

  let condition_number matrix =
    let _, s, _ = Tensor.svd matrix in
    let s_arr = Tensor.to_float_array1 s in
    if Array.length s_arr > 0 && s_arr.(Array.length s_arr - 1) > eps then
      s_arr.(0) /. s_arr.(Array.length s_arr - 1)
    else
      infinity

  let stable_matmul a b ~epsilon =
    let product = Matrix.matmul a b in
    let scale = Tensor.max (Tensor.abs product) |> Tensor.float_value in
    if scale > epsilon then
      Tensor.div_scalar product scale
    else
      product
end

(* Basic statistics *)
module Stats = struct
  let mse a b =
    let diff = Tensor.sub a b in
    let squared = Tensor.mul diff diff in
    let mean = Tensor.mean squared in
    Tensor.float_value mean

  let nuclear_norm m =
    let _, s, _ = Tensor.svd m in
    Tensor.sum s |> Tensor.float_value

  let frobenius_norm m =
    Tensor.norm m |> Tensor.float_value

  let mean matrix =
    Tensor.mean matrix |> Tensor.float_value

  let variance matrix =
    let mean_val = mean matrix in
    let centered = Tensor.sub matrix (Tensor.float_scalar mean_val) in
    let squared = Tensor.mul centered centered in
    Tensor.mean squared |> Tensor.float_value
end

(* Interval scaling and bounds handling *)
module IntervalScaling = struct
  type interval = {
    lower: float;
    upper: float;
  }

  type scaling_params = {
    original_interval: interval;
    scaled_interval: interval;
    scale_factor: float;
    offset: float;
  }

  let compute_scaling_params ~original_interval ~target_interval =
    let range_orig = original_interval.upper -. original_interval.lower in
    let range_target = target_interval.upper -. target_interval.lower in
    let scale_factor = range_target /. range_orig in
    let offset = target_interval.lower -. 
                (original_interval.lower *. scale_factor) in
    {
      original_interval;
      scaled_interval = target_interval;
      scale_factor;
      offset;
    }

  let scale_matrix matrix params =
    let scaled = Tensor.mul_scalar matrix params.scale_factor in
    Tensor.add_scalar scaled params.offset

  let inverse_scale_matrix matrix params =
    let centered = Tensor.sub_scalar matrix params.offset in
    Tensor.div_scalar centered params.scale_factor

  let clip_values matrix bounds =
    Tensor.clamp matrix 
      ~min:(Tensor.float_scalar bounds.lower)
      ~max:(Tensor.float_scalar bounds.upper)
end

(* Variance handling and analysis *)
module VarianceHandling = struct
  type variance_info = {
    known_variance: bool;
    sigma_sq: float option;
    estimated_variance: float;
    q_factor: float;  (* q := pσ² + p(1-p)(1-σ²) *)
    confidence_level: float;
  }

  let estimate_variance observed =
    let data = ObservedMatrix.to_dense observed in
    let mask = ObservedMatrix.get_mask observed in
    
    let observed_values = ref [] in
    let m, n = Matrix.dims data in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        if Matrix.get mask i j > 0.5 then
          observed_values := Matrix.get data i j :: !observed_values
      done
    done;
    
    let values = Array.of_list !observed_values in
    if Array.length values = 0 then 0.0
    else begin
      let mean = Array.fold_left (+.) 0.0 values /. 
                 float_of_int (Array.length values) in
      let var = Array.fold_left (fun acc x ->
        acc +. (x -. mean) ** 2.0) 0.0 values /. 
        float_of_int (Array.length values - 1) in
      min var 1.0  (* Ensure variance <= 1 as per paper requirements *)
    end

  let compute_q_factor p sigma_sq =
    p *. sigma_sq +. p *. (1.0 -. p) *. (1.0 -. sigma_sq)

  let create_variance_info ?sigma_sq observed =
    let p = ObservedMatrix.observed_proportion observed in
    let estimated_var = estimate_variance observed in
    
    let effective_var = match sigma_sq with
      | Some v when v <= 1.0 -> v
      | _ -> estimated_var in
    
    let q = compute_q_factor p effective_var in
    
    {
      known_variance = Option.is_some sigma_sq;
      sigma_sq;
      estimated_variance = estimated_var;
      q_factor = q;
      confidence_level = 0.95;
    }
end

(* Core USVT *)
module USVT = struct
  type config = {
    eta: float;               (* Threshold parameter η *)
    bound: float;             (* Upper bound for matrix entries *)
    use_variance: bool;       (* Whether to use variance information *)
    adaptive_threshold: bool; (* Whether to use adaptive thresholding *)
  }

  type t = {
    config: config;
    estimate: Matrix.t;
  }

  let default_config = {
    eta = 0.01;
    bound = 1.0;
    use_variance = true;
    adaptive_threshold = true;
  }

  let compute_threshold ~config ~n ~p ~variance_info =
    let base_factor = 2.0 +. config.eta in
    
    let variance_factor = match variance_info.sigma_sq with
      | Some sigma_sq -> 
          let q = VarianceHandling.compute_q_factor p sigma_sq in
          sqrt (q /. sigma_sq)
      | None -> 1.0 in
    
    let adaptive_factor =
      if config.adaptive_threshold then
        let p_factor = sqrt (p /. 0.5) in  (* Density adjustment *)
        min p_factor 1.0
      else 1.0 in
    
    base_factor *. variance_factor *. adaptive_factor *. 
    sqrt (float_of_int n *. p)

  let clip_values ~bound matrix =
    Tensor.clamp matrix 
      ~min:(Tensor.float_scalar (-.bound))
      ~max:(Tensor.float_scalar bound)

  let estimate ?(config=default_config) observed =
    let m, n = ObservedMatrix.dims observed in
    let p = ObservedMatrix.observed_proportion observed in
    
    (* Get dense representation with zeros for missing values *)
    let y = ObservedMatrix.to_dense observed in
    
    (* Compute variance information if needed *)
    let variance_info = 
      if config.use_variance then
        VarianceHandling.create_variance_info observed
      else
        VarianceHandling.create_variance_info ~sigma_sq:1.0 observed in
    
    (* Compute SVD *)
    let u, s, v = NumericalStability.stable_svd y ~epsilon:1e-10 in
    
    (* Compute threshold *)
    let threshold = compute_threshold ~config ~n ~p ~variance_info in
    
    (* Threshold singular values *)
    let s_thresholded = Tensor.where s 
      (Tensor.gt s (Tensor.float_scalar threshold))
      (Tensor.zeros_like s) in
    
    (* Reconstruct matrix *)
    let w = NumericalStability.stable_matmul u
      (NumericalStability.stable_matmul (Tensor.diag s_thresholded)
                                      (Matrix.transpose v))
      ~epsilon:1e-10 in
    
    (* Scale by observation probability *)
    let w_scaled = Tensor.div_scalar w p in
    
    (* Ensure bounds *)
    let estimate = clip_values ~bound:config.bound w_scaled in
    
    { config; estimate }
end

(* Enhanced USVT with improved variance handling *)
module EnhancedUSVT = struct
  type config = {
    base_config: USVT.config;
    min_rank: int option;
    max_rank: int option;
    convergence_threshold: float;
    use_iterative_refinement: bool;
  }

  let default_config = {
    base_config = USVT.default_config;
    min_rank = None;
    max_rank = None;
    convergence_threshold = 1e-6;
    use_iterative_refinement = true;
  }

  (* Select optimal rank based on singular values *)
  let select_rank s config =
    let s_arr = Tensor.to_float_array1 s in
    let total_variance = Array.fold_left (+.) 0.0 s_arr in
    let cumulative_variance = Array.make (Array.length s_arr) 0.0 in
    
    let _ = Array.fold_left (fun (idx, acc) sv ->
      cumulative_variance.(idx) <- acc +. sv;
      (idx + 1, acc +. sv)
    ) (0, 0.0) s_arr in
    
    let rank = ref 0 in
    while !rank < Array.length s_arr && 
          cumulative_variance.(!rank) /. total_variance < 0.95 do
      incr rank
    done;
    
    match config.min_rank, config.max_rank with
    | Some min_r, Some max_r -> min max_r (max min_r !rank)
    | Some min_r, None -> max min_r !rank
    | None, Some max_r -> min max_r !rank
    | None, None -> !rank

  (* Iterative refinement process *)
  let refine_estimate initial_est observed max_iter tol =
    let current = ref initial_est in
    let converged = ref false in
    let iter = ref 0 in
    
    while not !converged && !iter < max_iter do
      let prev = Tensor.copy !current in
      
      (* Update missing entries *)
      let m, n = Matrix.dims !current in
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          match ObservedMatrix.get observed i j with
          | None -> ()  (* Keep estimated value *)
          | Some v -> Matrix.set !current i j v |> ignore
        done
      done;
      
      (* Check convergence *)
      let diff = Tensor.sub !current prev in
      let rel_change = Tensor.norm diff /. Tensor.norm prev 
                      |> Tensor.float_value in
      
      converged := rel_change < tol;
      incr iter
    done;
    !current

  let estimate ?(config=default_config) observed =
    (* Initial USVT estimate *)
    let base_est = USVT.estimate ~config:config.base_config observed in
    
    if config.use_iterative_refinement then
      let refined = refine_estimate base_est.estimate observed 100 
                     config.convergence_threshold in
      { base_est with estimate = refined }
    else
      base_est
end

(* Stochastic Block Model *)
module StochasticBlockModel = struct
  type config = {
    n: int;                    (* Number of vertices *)
    k: int;                    (* Number of blocks *)
    sparsity: float;           (* Sparsity parameter ρ_n *)
    growing_blocks: bool;      (* Whether k grows with n *)
    min_block_size: int;       (* Minimum block size *)
  }

  type block_structure = {
    assignments: int array;    (* Block assignments *)
    sizes: int array;         (* Block sizes *)
    prob_matrix: Matrix.t;    (* Inter-block probabilities *)
  }

  let create_stochastic_model config =
    let actual_k = 
      if config.growing_blocks then
        min (max 2 (int_of_float (sqrt (float_of_int config.n)))) config.n
      else config.k in
    
    (* Initialize block assignments *)
    let assignments = Array.make config.n 0 in
    let sizes = Array.make actual_k 0 in
    
    (* Assign vertices to blocks *)
    let block_size = config.n / actual_k in
    for i = 0 to config.n - 1 do
      let block = min (i / block_size) (actual_k - 1) in
      assignments.(i) <- block;
      sizes.(block) <- sizes.(block) + 1
    done;
    
    (* Create probability matrix scaled by sparsity *)
    let prob_matrix = Matrix.create actual_k actual_k in
    for i = 0 to actual_k - 1 do
      for j = 0 to actual_k - 1 do
        let prob = config.sparsity *. Random.float 1.0 in
        Matrix.set prob_matrix i j prob |> ignore
      done
    done;
    
    { assignments; sizes; prob_matrix }

  let estimate_blocks observed config =
    let data = ObservedMatrix.to_dense observed in
    let n = fst (Matrix.dims data) in
    
    (* Compute normalized Laplacian *)
    let degree = Array.make n 0.0 in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        degree.(i) <- degree.(i) +. Matrix.get data i j
      done
    done;
    
    let laplacian = Matrix.create n n in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        if degree.(i) > 0.0 && degree.(j) > 0.0 then
          let norm_factor = 1.0 /. sqrt (degree.(i) *. degree.(j)) in
          let value = if i = j then 1.0 
                     else -. Matrix.get data i j *. norm_factor in
          Matrix.set laplacian i j value |> ignore
      done
    done;
    
    (* Spectral clustering *)
    let u, s, _ = Tensor.svd laplacian in
    let actual_k = 
      if config.growing_blocks then
        min (max 2 (int_of_float (sqrt (float_of_int n)))) n
      else config.k in
    
    let features = Matrix.create n actual_k in
    for i = 0 to n - 1 do
      for j = 0 to actual_k - 1 do
        Matrix.set features i j (Tensor.get u [i; j]) |> ignore
      done
    done;
    
    (* K-means clustering *)
    let assignments = Array.make n 0 in
    let converged = ref false in
    let iter = ref 0 in
    
    while not !converged && !iter < 100 do
      let old_assignments = Array.copy assignments in
      
      (* Update assignments *)
      for i = 0 to n - 1 do
        let min_dist = ref infinity in
        let min_cluster = ref 0 in
        
        for k = 0 to actual_k - 1 do
          let dist = ref 0.0 in
          for j = 0 to actual_k - 1 do
            let diff = Matrix.get features i j -. 
                      (float_of_int k /. float_of_int actual_k) in
            dist := !dist +. diff *. diff
          done;
          
          if !dist < !min_dist then begin
            min_dist := !dist;
            min_cluster := k
          end
        done;
        
        assignments.(i) <- !min_cluster
      done;
      
      (* Check convergence *)
      converged := true;
      for i = 0 to n - 1 do
        if assignments.(i) <> old_assignments.(i) then
          converged := false
      done;
      
      incr iter
    done;
    
    (* Compute block sizes and probability matrix *)
    let sizes = Array.make actual_k 0 in
    Array.iter (fun block -> sizes.(block) <- sizes.(block) + 1) assignments;
    
    let prob_matrix = Matrix.create actual_k actual_k in
    let counts = Matrix.create actual_k actual_k in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let value = Matrix.get data i j in
        let bi = assignments.(i) in
        let bj = assignments.(j) in
        Matrix.set prob_matrix bi bj
          (Matrix.get prob_matrix bi bj +. value) |> ignore;
        Matrix.set counts bi bj
          (Matrix.get counts bi bj +. 1.0) |> ignore
      done
    done;
    
    (* Normalize probability matrix *)
    for i = 0 to actual_k - 1 do
      for j = 0 to actual_k - 1 do
        let count = Matrix.get counts i j in
        if count > 0.0 then
          Matrix.set prob_matrix i j
            (Matrix.get prob_matrix i j /. count) |> ignore
      done
    done;
    
    { assignments; sizes; prob_matrix }
end

(* Distance Matrix *)
module DistanceMatrix = struct
  module MetricSpace = struct
    module type METRIC = sig
      type t
      val distance : t -> t -> float
      val ball_covering : t list -> float -> t list list
      val diameter : t list -> float
      val is_compact : t list -> bool
    end

    module EuclideanMetric : METRIC with type t = float array = struct
      type t = float array

      let distance x y =
        Array.fold_left2 (fun acc xi yi ->
          acc +. (xi -. yi) ** 2.0) 0.0 x y |> sqrt

      let rec ball_covering points delta =
        if points = [] then []
        else
          let center = List.hd points in
          let (in_ball, out_ball) = List.partition
            (fun p -> distance p center <= delta) points in
          in_ball :: ball_covering out_ball delta

      let diameter points =
        let max_dist = ref 0.0 in
        List.iter (fun x ->
          List.iter (fun y ->
            max_dist := max !max_dist (distance x y)
          ) points
        ) points;
        !max_dist

      let is_compact points =
        let dim = Array.length (List.hd points) in
        List.for_all (fun p -> Array.length p = dim) points &&
        List.for_all (fun p ->
          Array.for_all (fun x -> Float.is_finite x) p) points
    end
  end

  type point = float array
  
  type t = {
    points: point array;
    metric: (module MetricSpace.METRIC with type t = point);
  }

  let create points metric = { points; metric }

  let compute_distance_matrix t =
    let n = Array.length t.points in
    let matrix = Matrix.create n n in
    let module M = (val t.metric) in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        Matrix.set matrix i j 
          (M.distance t.points.(i) t.points.(j)) |> ignore
      done
    done;
    matrix

  let estimate observed =
    let est = USVT.estimate observed in
    
    (* Project onto distance matrix constraints *)
    let m, n = Matrix.dims est.estimate in
    let result = Matrix.create m n in
    
    (* Ensure symmetry and non-negativity *)
    for i = 0 to m - 1 do
      Matrix.set result i i 0.0 |> ignore;
      for j = i + 1 to n - 1 do
        let dist = max 0.0 ((Matrix.get est.estimate i j +. 
                            Matrix.get est.estimate j i) /. 2.0) in
        Matrix.set result i j dist |> ignore;
        Matrix.set result j i dist |> ignore
      done
    done;
    
    { est with estimate = result }
end

(* Graphon estimation *)
module Graphon = struct
  type t = {
    f: float -> float -> float;
    is_symmetric: bool;
    is_measurable: bool;
    support: float * float;
  }

  type discrete_graphon = {
    matrix: Matrix.t;
    n: int;
  }

  let step_graphon matrix =
    let n = fst (Matrix.dims matrix) in
    let f x y =
      let i = min (n - 1) (int_of_float (x *. float_of_int n)) in
      let j = min (n - 1) (int_of_float (y *. float_of_int n)) in
      Matrix.get matrix i j
    in
    {
      f;
      is_symmetric = true;
      is_measurable = true;
      support = (0.0, 1.0);
    }

  let estimate observed =
    let est = USVT.estimate observed in
    let n = fst (Matrix.dims est.estimate) in
    
    (* Smooth the estimate *)
    let bandwidth = 1.0 /. sqrt (float_of_int n) in
    let kernel x = exp (-. x *. x /. (2.0 *. bandwidth *. bandwidth)) in
    
    let smoothed = Matrix.create n n in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let x = float_of_int i /. float_of_int n in
        let y = float_of_int j /. float_of_int n in
        
        let sum = ref 0.0 in
        let weight_sum = ref 0.0 in
        
        for k = 0 to n - 1 do
          for l = 0 to n - 1 do
            let x_k = float_of_int k /. float_of_int n in
            let y_l = float_of_int l /. float_of_int n in
            
            let dist = sqrt ((x -. x_k) ** 2.0 +. (y -. y_l) ** 2.0) in
            let weight = kernel dist in
            
            sum := !sum +. weight *. Matrix.get est.estimate k l;
            weight_sum := !weight_sum +. weight
          done
        done;
        
        Matrix.set smoothed i j (!sum /. !weight_sum) |> ignore
      done
    done;
    
    { est with estimate = smoothed }
end

(* Bradley-Terry model *)
module BradleyTerry = struct
  type player = {
    id: int;
    name: string option;
    initial_rating: float option;
  }

  type match_result = {
    player1: int;
    player2: int;
    outcome: float;  (* 1.0 for p1 win, 0.5 for draw, 0.0 for p2 win *)
    weight: float;   (* Match importance weight *)
    timestamp: float option;
  }

  type model = {
    n_players: int;
    players: player array;
    strength_matrix: Matrix.t;  (* Pairwise winning probabilities *)
    ratings: float array;       (* Current player ratings *)
    uncertainty: float array;   (* Rating uncertainties *)
  }

  type config = {
    allow_draws: bool;
    temporal_weighting: bool;
    min_matches: int;
    regularization: float;
    convergence_tol: float;
    max_iterations: int;
  }

  let default_config = {
    allow_draws = true;
    temporal_weighting = true;
    min_matches = 5;
    regularization = 0.1;
    convergence_tol = 1e-6;
    max_iterations = 100;
  }

  let create_player ?name ?initial_rating id =
    { id; name; initial_rating }

  let create_model players =
    let n = Array.length players in
    {
      n_players = n;
      players;
      strength_matrix = Matrix.create n n;
      ratings = Array.init n (fun i ->
        match players.(i).initial_rating with
        | Some r -> r
        | None -> 0.0);
      uncertainty = Array.make n 1.0;
    }

  (* Compute temporal weights *)
  let compute_temporal_weights results current_time =
    Array.map (fun result ->
      match result.timestamp with
      | Some t -> exp (-. (current_time -. t) /. 365.25)
      | None -> 1.0
    ) results

  (* Weighted maximum likelihood estimate *)
  let weighted_mle model results weights =
    let n = model.n_players in
    let wins = Matrix.create n n in
    let matches = Matrix.create n n in
    
    (* Accumulate weighted statistics *)
    Array.iteri (fun idx result ->
      let weight = weights.(idx) *. result.weight in
      Matrix.set wins result.player1 result.player2
        (Matrix.get wins result.player1 result.player2 +. 
         result.outcome *. weight) |> ignore;
      Matrix.set matches result.player1 result.player2
        (Matrix.get matches result.player1 result.player2 +. weight) |> ignore;
      
      if result.outcome <> 0.5 then
        Matrix.set wins result.player2 result.player1
          (Matrix.get wins result.player2 result.player1 +. 
           (1.0 -. result.outcome) *. weight) |> ignore;
      Matrix.set matches result.player2 result.player1
        (Matrix.get matches result.player2 result.player1 +. weight) |> ignore
    ) results;
    
    (* MM algorithm for MLE *)
    let ratings = Array.copy model.ratings in
    let converged = ref false in
    let iter = ref 0 in
    
    while not !converged && !iter < default_config.max_iterations do
      let old_ratings = Array.copy ratings in
      
      (* Update step *)
      for i = 0 to n - 1 do
        let numerator = ref 0.0 in
        let denominator = ref 0.0 in
        
        for j = 0 to n - 1 do
          if i <> j then begin
            let matches_ij = Matrix.get matches i j in
            if matches_ij > 0.0 then begin
              let wins_ij = Matrix.get wins i j in
              let exp_rating_j = exp ratings.(j) in
              numerator := !numerator +. wins_ij;
              denominator := !denominator +. 
                (matches_ij *. exp_rating_j /. 
                 (exp ratings.(i) +. exp_rating_j))
            end
          end
        done;
        
        if !denominator > 0.0 then
          ratings.(i) <- log (!numerator /. !denominator)
      done;
      
      (* Center ratings *)
      let mean = Array.fold_left (+.) 0.0 ratings /. float_of_int n in
      Array.iteri (fun i r -> ratings.(i) <- r -. mean) ratings;
      
      (* Check convergence *)
      let max_change = ref 0.0 in
      Array.iteri (fun i r ->
        max_change := max !max_change 
          (abs_float (r -. old_ratings.(i)))) ratings;
      
      converged := !max_change < default_config.convergence_tol;
      incr iter
    done;
    
    (* Update uncertainties based on Fisher information *)
    let uncertainties = Array.make n 0.0 in
    for i = 0 to n - 1 do
      let info = ref 0.0 in
      for j = 0 to n - 1 do
        if i <> j then begin
          let matches_ij = Matrix.get matches i j in
          if matches_ij > 0.0 then begin
            let exp_i = exp ratings.(i) in
            let exp_j = exp ratings.(j) in
            let denom = exp_i +. exp_j in
            info := !info +. 
              (matches_ij *. exp_i *. exp_j /. (denom *. denom))
          end
        end
      done;
      uncertainties.(i) <- if !info > 0.0 then 1.0 /. sqrt !info 
                          else model.uncertainty.(i)
    done;
    
    (ratings, uncertainties)

  (* Full model estimation *)
  let estimate ?(config=default_config) model results =
    let current_time = Unix.time () in
    
    (* Compute weights *)
    let weights = if config.temporal_weighting then
      compute_temporal_weights results current_time
    else
      Array.make (Array.length results) 1.0 in
    
    (* Compute MLE with weights *)
    let ratings, uncertainties = weighted_mle model results weights in
    
    (* Update strength matrix *)
    let strength_matrix = Matrix.create model.n_players model.n_players in
    for i = 0 to model.n_players - 1 do
      for j = 0 to model.n_players - 1 do
        if i <> j then begin
          let exp_i = exp ratings.(i) in
          let exp_j = exp ratings.(j) in
          let prob = exp_i /. (exp_i +. exp_j) in
          Matrix.set strength_matrix i j prob |> ignore
        end
      done
    done;
    
    { model with strength_matrix; ratings; uncertainty = uncertainties }
end
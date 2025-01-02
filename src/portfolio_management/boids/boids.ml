open Torch

module Error = struct
  type t = 
    | ConfigError of string
    | OptimizationError of string
    | GPError of string
    | DataError of string
    | SubspaceError of string
    | ResourceError of string

  let to_string = function
    | ConfigError msg -> "Configuration error: " ^ msg
    | OptimizationError msg -> "Optimization error: " ^ msg
    | GPError msg -> "Gaussian Process error: " ^ msg
    | DataError msg -> "Data error: " ^ msg
    | SubspaceError msg -> "Subspace error: " ^ msg
    | ResourceError msg -> "Resource error: " ^ msg

  let raise_error e = raise (Failure (to_string e))
end

module Config = struct
  type gp_config = {
    noise_variance: float;
    length_scale: float;
    signal_variance: float;
  }

  type optimization_config = {
    max_iterations: int;
    batch_size: int;
    tolerance: float;
    patience: int;
  }

  type subspace_config = {
    initial_dim: int;
    max_dim: int;
    min_explained_variance: float;
  }

  type t = {
    input_dim: int;
    n_initial_points: int;
    gp: gp_config;
    optimization: optimization_config;
    subspace: subspace_config;
    random_seed: int option;
  }

  let default_gp_config = {
    noise_variance = 1e-6;
    length_scale = 1.0;
    signal_variance = 1.0;
  }

  let default_optimization_config = {
    max_iterations = 1000;
    batch_size = 10;
    tolerance = 1e-6;
    patience = 20;
  }

  let default_subspace_config = {
    initial_dim = 2;
    max_dim = 100;
    min_explained_variance = 0.95;
  }

  let create ~input_dim ?n_initial_points ?gp_config ?opt_config 
             ?subspace_config ?random_seed () =
    let n_initial = match n_initial_points with
      | Some n -> n
      | None -> min (input_dim * 2) 100 in
    {
      input_dim;
      n_initial_points = n_initial;
      gp = Option.value ~default:default_gp_config gp_config;
      optimization = Option.value ~default:default_optimization_config opt_config;
      subspace = Option.value ~default:default_subspace_config subspace_config;
      random_seed;
    }

  let validate config =
    try
      if config.input_dim <= 0 then 
        Error (Error.ConfigError "Input dimension must be positive")
      else if config.n_initial_points <= 0 then
        Error (Error.ConfigError "Number of initial points must be positive")
      else if config.gp.noise_variance <= 0.0 then
        Error (Error.ConfigError "GP noise variance must be positive")
      else if config.gp.length_scale <= 0.0 then
        Error (Error.ConfigError "GP length scale must be positive")
      else if config.optimization.max_iterations <= 0 then
        Error (Error.ConfigError "Maximum iterations must be positive")
      else if config.subspace.initial_dim <= 0 then
        Error (Error.ConfigError "Initial subspace dimension must be positive")
      else if config.subspace.initial_dim > config.input_dim then
        Error (Error.ConfigError "Initial subspace dimension cannot exceed input dimension")
      else Ok config
    with e -> Error (Error.ConfigError (Printexc.to_string e))
end

module Dataset = struct
  type point = Tensor.t
  type value = float

  type t = {
    points: point array;
    values: value array;
    best_value: value;
    best_point: point;
  }

  let create points values =
    if Array.length points <> Array.length values then
      Error.raise_error (Error.DataError "Points and values must have same length");
    let best_idx = ref 0 in
    let best_val = ref values.(0) in
    Array.iteri (fun i v ->
      if v < !best_val then begin
        best_val := v;
        best_idx := i
      end
    ) values;
    {
      points;
      values;
      best_value = !best_val;
      best_point = points.(!best_idx);
    }

  let add t point value =
    let new_points = Array.append t.points [|point|] in
    let new_values = Array.append t.values [|value|] in
    let new_best_value = min t.best_value value in
    let new_best_point = 
      if value < t.best_value then point else t.best_point in
    {
      points = new_points;
      values = new_values;
      best_value = new_best_value;
      best_point = new_best_point;
    }

  let add_batch t points values =
    if Array.length points <> Array.length values then
      Error.raise_error (Error.DataError "Batch points and values must have same length");
    let new_points = Array.append t.points points in
    let new_values = Array.append t.values values in
    let best_idx = ref 0 in
    let best_val = ref t.best_value in
    Array.iteri (fun i v ->
      if v < !best_val then begin
        best_val := v;
        best_idx := i + Array.length t.points
      end
    ) values;
    {
      points = new_points;
      values = new_values;
      best_value = !best_val;
      best_point = if !best_val < t.best_value then 
                    points.(!best_idx - Array.length t.points)
                  else t.best_point;
    }

  let size t = Array.length t.points

  let get_best t = (t.best_point, t.best_value)

  let to_tensor t =
    (Tensor.stack (Array.to_list t.points) ~dim:0,
     Tensor.of_float1 t.values)

  let of_tensor points values =
    let points_array = Array.init (Tensor.size points 0) (fun i ->
      Tensor.select points 0 i
    ) in
    let values_array = Tensor.to_float1 values in
    create points_array values_array
end

module GP = struct
  type t = {
    params: Config.gp_config;
    l_matrix: Tensor.t option;
    alpha: Tensor.t option;
  }

  type params = Config.gp_config

  let kernel params x1 x2 =
    let diff = Tensor.sub 
      (Tensor.unsqueeze x1 1) 
      (Tensor.unsqueeze x2 0) in
    let sq_dist = Tensor.sum (Tensor.mul diff diff) ~dim:[2] in
    let scaled_dist = Tensor.div sq_dist 
      (Tensor.scalar_tensor (2.0 *. params.length_scale *. params.length_scale)) in
    Tensor.mul_scalar 
      (Tensor.exp (Tensor.neg scaled_dist)) 
      params.signal_variance

  let create params dataset =
    let points, values = Dataset.to_tensor dataset in
    let k = kernel params points points in
    let n = Tensor.size points 0 in
    let noise_diag = Tensor.mul_scalar 
      (Tensor.eye n) params.noise_variance in
    let k_noise = Tensor.add k noise_diag in
    try
      let l = Tensor.cholesky k_noise in
      let alpha = Tensor.triangular_solve values l ~upper:false in
      {params; l_matrix = Some l; alpha = Some alpha}
    with _ -> Error.raise_error (Error.GPError "Cholesky decomposition failed")

  let predict t point dataset =
    match t.l_matrix, t.alpha with
    | Some l, Some alpha ->
        let points, _ = Dataset.to_tensor dataset in
        let k_star = kernel t.params points (Tensor.unsqueeze point 0) in
        let k_star_star = kernel t.params 
          (Tensor.unsqueeze point 0) 
          (Tensor.unsqueeze point 0) in
        let mean = Tensor.matmul 
          (Tensor.transpose k_star 0 1) alpha in
        let v = Tensor.triangular_solve k_star l ~upper:false in
        let var = Tensor.sub k_star_star 
          (Tensor.matmul (Tensor.transpose v 0 1) v) in
        (Tensor.item (Tensor.select mean 0 0),
         Tensor.item (Tensor.select var 0 0))
    | _ -> Error.raise_error (Error.GPError "GP not properly initialized")

  let update t dataset = create t.params dataset

  let optimize_hyperparams t dataset =
    let rec optimize params iter best_ll =
      if iter >= 100 then params
      else
        let try_params = {
          length_scale = params.length_scale *. (1.0 +. Random.float 0.2 -. 0.1);
          signal_variance = params.signal_variance *. (1.0 +. Random.float 0.2 -. 0.1);
          noise_variance = params.noise_variance;
        } in
        let try_gp = create try_params dataset in
        let ll = log_likelihood try_gp dataset in
        if ll > best_ll then
          optimize try_params (iter + 1) ll
        else
          optimize params (iter + 1) best_ll
    in
    let new_params = optimize t.params 0 (log_likelihood t dataset) in
    create new_params dataset

  let get_gradients t dataset =
    let points, _ = Dataset.to_tensor dataset in
    let n = Tensor.size points 0 in
    let d = Tensor.size points 1 in
    let grads = Tensor.zeros [n; d] in
    for i = 0 to n - 1 do
      let x = Tensor.select points 0 i in
      let g = Tensor.grad (fun x -> fst (predict t x dataset)) x in
      Tensor.copy_ (Tensor.select grads 0 i) g
    done;
    grads
end

module LineOpt = struct
  type line = {
    start: Dataset.point;
    direction: Dataset.point;
    length: float;
  }

  type acquisition = Dataset.point -> float

  let create start direction length = {start; direction; length}

  let sample_points line n =
    let alphas = Tensor.linspace 0.0 line.length n in
    Array.init n (fun i ->
      let alpha = Tensor.select alphas 0 i in
      Tensor.add line.start 
        (Tensor.mul_scalar line.direction (Tensor.item alpha)))

  let optimize line acquisitions config =
    let points = sample_points line config.Config.batch_size in
    let values = Array.map (fun p ->
      Array.fold_left (fun acc f -> acc +. f p) 0.0 acquisitions
    ) points in
    let best_idx = ref 0 in
    let best_val = ref values.(0) in
    Array.iteri (fun i v ->
      if v > !best_val then begin
        best_val := v;
        best_idx := i
      end
    ) values;
    points.(!best_idx)
end

module Subspace = struct
  type t = {
    dim: int;
    projection: Tensor.t;
    reconstruction: Tensor.t;
    explained_variance: float;
  }

  let create config =
    let proj = Tensor.rand [config.Config.initial_dim; config.input_dim] in
    let proj_normalized = Tensor.div proj 
      (Tensor.sqrt (Tensor.sum (Tensor.mul proj proj) ~dim:[1])) in
    {
      dim = config.initial_dim;
      projection = proj_normalized;
      reconstruction = Tensor.transpose proj_normalized 0 1;
      explained_variance = 0.0;
    }

  let project t point =
    Tensor.matmul (Tensor.unsqueeze point 0) 
      (Tensor.transpose t.projection 0 1)
    |> fun x -> Tensor.select x 0 0

  let reconstruct t point =
    Tensor.matmul (Tensor.unsqueeze point 0) t.projection
    |> fun x -> Tensor.select x 0 0

  let update t gp dataset =
    let grads = GP.get_gradients gp dataset in
    let cov = Tensor.matmul 
      (Tensor.transpose grads 0 1) grads in
    let eigenvalues, eigenvectors = Tensor.linalg.eigh cov in
    let total_var = Tensor.sum eigenvalues in
    let cum_var = Tensor.cumsum eigenvalues 0 in
    let explained = Tensor.div cum_var total_var in
    let new_dim = ref t.dim in
    for i = 0 to Tensor.size explained 0 - 1 do
      if Tensor.item (Tensor.select explained 0 i) >= 0.95 then
        new_dim := i + 1
    done;
    let new_proj = Tensor.narrow eigenvectors 1 0 !new_dim in
    {
      dim = !new_dim;
      projection = new_proj;
      reconstruction = Tensor.transpose new_proj 0 1;
      explained_variance = Tensor.item (Tensor.select explained 0 (!new_dim - 1));
    }

  let should_expand t dataset =
    t.explained_variance < 0.95 && 
    t.dim < Dataset.size dataset

  let expand t =
    if t.dim >= Tensor.size t.reconstruction 0 then t
    else
      let new_dim = min (t.dim * 2) (Tensor.size t.reconstruction 0) in
      let new_proj = Tensor.narrow t.projection 0 0 new_dim in
      {t with 
       dim = new_dim;
       projection = new_proj;
       reconstruction = Tensor.transpose new_proj 0 1}
end

module MAB = struct
  type arm = {
    id: int;
    rewards: float list;
    pulls: int;
  }

  type t = {
    arms: arm array;
    exploration_factor: float;
  }

  let create n_arms exploration_factor =
    let arms = Array.init n_arms (fun id -> 
      {id; rewards = []; pulls = 0}
    ) in
    {arms; exploration_factor}

  let mean_reward arm =
    match arm.rewards with
    | [] -> 0.0
    | rs -> List.fold_left (+.) 0.0 rs /. float_of_int (List.length rs)

  let ucb t arm time =
    if arm.pulls = 0 then infinity
    else
      let exploit = mean_reward arm in
      let explore = t.exploration_factor *. 
        sqrt (log (float_of_int time) /. float_of_int arm.pulls) in
      exploit +. explore

  let select t =
    let time = Array.fold_left (fun acc arm -> acc + arm.pulls) 0 t.arms in
    let scores = Array.map (fun arm -> ucb t arm time) t.arms in
    let best_idx = ref 0 in
    let best_score = ref scores.(0) in
    Array.iteri (fun i score ->
      if score > !best_score then begin
        best_score := score;
        best_idx := i
      end
    ) scores;
    !best_idx

  let update t arm_idx reward =
    let arms = Array.copy t.arms in
    let arm = arms.(arm_idx) in
    arms.(arm_idx) <- {
      arm with
      rewards = reward :: arm.rewards;
      pulls = arm.pulls + 1
    };
    {t with arms}
end

module Monitor = struct
  type metric = {
    iteration: int;
    best_value: float;
    gp_likelihood: float;
    subspace_dim: int;
    runtime: float;
  }

  type history = metric list

  let create_metric iteration best_value gp_likelihood subspace_dim runtime =
    {iteration; best_value; gp_likelihood; subspace_dim; runtime}

  let update_history history metric =
    metric :: history

  let check_convergence history config =
    if List.length history < config.Config.optimization.patience then
      false
    else
      let recent = List.take config.Config.optimization.patience history in
      let values = List.map (fun m -> m.best_value) recent in
      let min_val = List.fold_left min infinity values in
      let max_val = List.fold_left max neg_infinity values in
      abs_float (max_val -. min_val) < config.Config.optimization.tolerance

  let save filename history =
    let oc = open_out filename in
    List.iter (fun metric ->
      Printf.fprintf oc "%d,%f,%f,%d,%f\n"
        metric.iteration
        metric.best_value
        metric.gp_likelihood
        metric.subspace_dim
        metric.runtime
    ) (List.rev history);
    close_out oc

  let load filename =
    let ic = open_in filename in
    let rec read_lines acc =
      try
        let line = input_line ic in
        match String.split_on_char ',' line with
        | [iter; best; ll; dim; time] ->
            let metric = {
              iteration = int_of_string iter;
              best_value = float_of_string best;
              gp_likelihood = float_of_string ll;
              subspace_dim = int_of_string dim;
              runtime = float_of_string time;
            } in
            read_lines (metric :: acc)
        | _ -> read_lines acc
      with End_of_file ->
        close_in ic;
        List.rev acc
    in
    read_lines []
end

module BOIDS = struct
  type result = {
    dataset: Dataset.t;
    history: Monitor.history;
    final_model: GP.t;
    runtime: float;
  }

  let initialize_dataset objective config =
    let points = Array.init config.Config.n_initial_points (fun _ ->
      Tensor.rand ~low:(-1.0) ~high:1.0 [config.input_dim]
    ) in
    let values = Array.map objective points in
    Dataset.create points values

  let create_acquisition_function gp dataset =
    let best_y = Dataset.get_best dataset |> snd in
    fun x ->
      let mu, sigma = GP.predict gp x dataset in
      let z = (mu -. best_y) /. (sqrt sigma) in
      let cdf = 0.5 *. (1.0 +. erf (z /. sqrt 2.0)) in
      let pdf = exp (-0.5 *. z *. z) /. sqrt (2.0 *. Float.pi) in
      sigma *. (z *. cdf +. pdf)

  let optimize objective config =
    let start_time = Unix.gettimeofday () in
    let dataset = ref (initialize_dataset objective config) in
    let subspace = ref (Subspace.create config.subspace) in
    let history = ref [] in
    
    let rec optimization_loop iteration =
      if iteration >= config.optimization.max_iterations then ()
      else begin
        (* Create and optimize GP model *)
        let gp = GP.create config.gp !dataset in
        let gp = GP.optimize_hyperparams gp !dataset in
        
        (* Generate candidates using line optimization *)
        let acquisition = create_acquisition_function gp !dataset in
        let lines = LineOpt.create_batch !dataset !subspace in
        let mab = MAB.create (Array.length lines) 0.5 in
        let best_line_idx = MAB.select mab in
        let best_line = lines.(best_line_idx) in
        
        (* Optimize along selected line *)
        let new_point = LineOpt.optimize best_line [|acquisition|] 
          config.optimization in
        
        (* Evaluate new point *)
        let orig_point = Subspace.reconstruct !subspace new_point in
        let value = objective orig_point in
        dataset := Dataset.add !dataset orig_point value;
        
        (* Update subspace if needed *)
        if Subspace.should_expand !subspace !dataset then
          subspace := Subspace.expand !subspace;
        
        (* Update monitoring *)
        let metric = Monitor.create_metric
          iteration
          (Dataset.get_best !dataset |> snd)
          (GP.log_likelihood gp !dataset)
          (!subspace).dim
          (Unix.gettimeofday () -. start_time) in
        history := Monitor.update_history !history metric;
        
        (* Check convergence *)
        if not (Monitor.check_convergence !history config.optimization) then
          optimization_loop (iteration + 1)
      end
    in
    
    optimization_loop 0;
    
    {
      dataset = !dataset;
      history = !history;
      final_model = GP.create config.gp !dataset;
      runtime = Unix.gettimeofday () -. start_time;
    }

  let continue_optimization prev_result config =
    let new_config = {config with
      n_initial_points = Dataset.size prev_result.dataset
    } in
    optimize (fun x -> 
      let dataset = prev_result.dataset in
      let gp = prev_result.final_model in
      fst (GP.predict gp x dataset)
    ) new_config
end

module Benchmarks = struct
  type problem = {
    name: string;
    dim: int;
    bounds: float * float;
    evaluate: Dataset.point -> float;
    optimum: float option;
  }

  let rosenbrock x =
    let x_array = Tensor.to_float1 x in
    let n = Array.length x_array in
    let rec sum i acc =
      if i >= n - 1 then acc
      else
        let xi = x_array.(i) in
        let xip1 = x_array.(i + 1) in
        let term1 = 100.0 *. (xip1 -. xi *. xi) ** 2.0 in
        let term2 = (1.0 -. xi) ** 2.0 in
        sum (i + 1) (acc +. term1 +. term2)
    in
    sum 0 0.0

  let create_suite dim =
    [|{
      name = "Rosenbrock";
      dim;
      bounds = (-5.0, 10.0);
      evaluate = rosenbrock;
      optimum = Some 0.0;
    }|]

  let load_real_world name =
    match name with
    | "nn_tuning" -> {
        name = "Neural Network Tuning";
        dim = 6;
        bounds = (0.0, 1.0);
        evaluate = (fun _ -> 0.0); 
        optimum = None;
      }
    | _ -> Error.raise_error (Error.ConfigError "Unknown benchmark problem")

  let evaluate_optimizer problem config n_runs =
    let results = Array.init n_runs (fun _ ->
      let start_time = Unix.gettimeofday () in
      let result = BOIDS.optimize problem.evaluate config in
      let runtime = Unix.gettimeofday () -. start_time in
      let best_value = Dataset.get_best result.dataset |> snd in
      (best_value, runtime)
    ) in
    let values, times = Array.split results in
    let mean_value = Array.fold_left (+.) 0.0 values /. float_of_int n_runs in
    let mean_time = Array.fold_left (+.) 0.0 times /. float_of_int n_runs in
    let std_value = sqrt (
      Array.fold_left (fun acc v -> 
        acc +. (v -. mean_value) ** 2.0
      ) 0.0 values /. float_of_int n_runs
    ) in
    (mean_value, std_value, mean_time)
end
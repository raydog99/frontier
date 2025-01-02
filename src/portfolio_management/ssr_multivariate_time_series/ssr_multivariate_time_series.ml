open Torch

module Parameters = struct
  type system_params = {
    coupling_strength: float;
    noise_level: float option;
    initial_conditions: float array option;
  }
  
  type simulation_params = {
    max_dim: int;
    r: float;
    k_neighbors: int;
  }
  
  type optimization_params = {
    n_particles: int;
    max_iterations: int;
    w: float;
    c1: float;
    c2: float;
  }
  
  type coupling_params = {
    bins: int;
    k_neighbors: int;
  }
end

type time_series = Tensor.t

type embedding_vector = {
  dimensions: int array;
  delays: int array;
}

type prediction_error = {
  nrmse: float;
  component_errors: float array;
}

let validate_preprocessing data =
  validate_data data 2
  
let scale_to_unit_interval tensor =
  let min_val = Tensor.min tensor in
  let max_val = Tensor.max tensor in
  let range = Tensor.sub max_val min_val in
  Tensor.(div (sub tensor min_val) range)
  
let standardize tensor =
  let mean = Tensor.mean tensor ~dim:[0] ~keepdim:true in
  let std = Tensor.std tensor ~dim:[0] ~keepdim:true ~unbiased:true in
  Tensor.(div (sub tensor mean) std)
  
let scale_multivariate data ~method_ =
  validate_preprocessing data;
  Array.map (fun series ->
    match method_ with
    | `UnitInterval -> scale_to_unit_interval series
    | `Standardize -> standardize series) data
    
let train_test_split data ~train_ratio =
  validate_preprocessing data;
  let n = Tensor.size data.(0) 0 in
  let train_size = int_of_float (float_of_int n *. train_ratio) in
  let test_size = n - train_size in
  
  let train_data = Array.map 
    (fun series -> Tensor.narrow series 0 0 train_size) data in
  let test_data = Array.map 
    (fun series -> Tensor.narrow series 0 train_size test_size) data in
  
  train_data, test_data

let validate_embedding_params ev total_vars =
  if Array.length ev.dimensions <> total_vars then
    raise (InvalidParameters "Embedding dimensions don't match variables");
  if Array.length ev.delays <> total_vars then
    raise (InvalidParameters "Embedding delays don't match variables")
    
let validate_embedded_data data =
  if Tensor.size data 0 < 1 then
    raise (InsufficientData "Empty embedded data")
    
let embed_univariate data ~dim ~delay =
  validate_data [|data|] (dim * delay);
  let n = Tensor.size data 0 in
  let embedded_size = n - (dim - 1) * delay in
  let result = Tensor.zeros [embedded_size; dim] in
  
  for i = 0 to embedded_size - 1 do
    for j = 0 to dim - 1 do
      let idx = i + j * delay in
      Tensor.set result [|i; j|] (Tensor.get data [|idx|])
    done
  done;
  result
  
let embed_multivariate data ev =
  validate_embedding_params ev (Array.length data);
  let n = Tensor.size data.(0) 0 in
  let total_dim = Array.fold_left (+) 0 ev.dimensions in
  let min_size = ref n in
  
  Array.iteri (fun i dim ->
    let size = n - (dim - 1) * ev.delays.(i) in
    min_size := min !min_size size
  ) ev.dimensions;
  
  let result = Tensor.zeros [!min_size; total_dim] in
  let col_offset = ref 0 in
  
  Array.iteri (fun i series ->
    let embedded = embed_univariate series 
                    ~dim:ev.dimensions.(i) 
                    ~delay:ev.delays.(i) in
                    
    for row = 0 to !min_size - 1 do
      for col = 0 to ev.dimensions.(i) - 1 do
        let value = Tensor.get embedded [|row; col|] in
        Tensor.set result [|row; !col_offset + col|] value
      done
    done;
    
    col_offset := !col_offset + ev.dimensions.(i)
  ) data;
  
  result


let validate_fnn_params max_dim r =
  if max_dim <= 0 then
    raise (InvalidParameters "max_dim must be positive");
  if r <= 0. then
    raise (InvalidParameters "r must be positive")
    
let calc_fnn data r =
  let n = Tensor.size data 0 in
  let dim = Tensor.size data 1 in
  let count = ref 0 in
  
  for i = 0 to n - 1 do
    let point = Tensor.select data 0 i in
    let neighbors = Models.find_neighbors_knn data point 2 in
    
    let d1 = Tensor.norm (Tensor.(-) point 
      (Tensor.select data 0 (Tensor.get neighbors [|1|]))) in
    let d2 = Tensor.norm (Tensor.(-) 
      (Tensor.cat [point; Tensor.zeros [1]] 0)
      (Tensor.cat [
        (Tensor.select data 0 (Tensor.get neighbors [|1|]));
        Tensor.zeros [1]
      ] 0)) in
    
    if d2 /. d1 > r then
      incr count
  done;
  
  float_of_int !count /. float_of_int n
  
let find_optimal data ~max_dim r =
  validate_fnn_params max_dim r;
  let best_dim = ref 1 in
  let best_score = ref Float.infinity in
  
  for dim = 1 to max_dim do
    let ev = {
      dimensions = Array.make (Array.length data) dim;
      delays = Array.make (Array.length data) 1;
    } in
    let embedded = embed_multivariate data ev in
    let score = calc_fnn embedded r in
    
    if score < !best_score then begin
      best_score := score;
      best_dim := dim
    end;
    
    if score < 0.01 then  (* Early stopping if good enough *)
      break
  done;
  
  let final_ev = {
    dimensions = Array.make (Array.length data) !best_dim;
    delays = Array.make (Array.length data) 1;
  } in
  final_ev, !best_score

let generate_embeddings nvars total_dim =
  let rec partition_number total parts accum =
    if parts = 1 then [[total]]
    else if total = 0 then [List.init parts (fun _ -> 0)]
    else
      List.concat (List.init (total + 1) (fun i ->
        List.map (fun rest -> i :: rest)
          (partition_number (total - i) (parts - 1) accum)))
  in
  
  let partitions = partition_number total_dim nvars [] in
  List.map (fun dims -> 
    { dimensions = Array.of_list dims;
      delays = Array.make nvars 1 }) partitions
      
let calc_fnn_increase data ev dim_to_increase r =
  let current = embed_multivariate data ev in
  let new_dims = Array.copy ev.dimensions in
  new_dims.(dim_to_increase) <- new_dims.(dim_to_increase) + 1;
  let new_ev = { ev with dimensions = new_dims } in
  let increased = embed_multivariate data new_ev in
  
  calc_fnn increased r
  
let find_optimal data ~max_dim r =
  let nvars = Array.length data in
  let best_ev = ref {
    dimensions = Array.make nvars 1;
    delays = Array.make nvars 1;
  } in
  let best_score = ref Float.infinity in
  
  for total_dim = nvars to max_dim do
    let candidates = generate_embeddings nvars total_dim in
    List.iter (fun ev ->
      let total_fnn = ref 0. in
      
      for i = 0 to nvars - 1 do
        total_fnn := !total_fnn +. calc_fnn_increase data ev i r
      done;
      
      if !total_fnn < !best_score then begin
        best_score := !total_fnn;
        best_ev := ev
      end
    ) candidates;
    
    if !best_score < 0.01 then break  (* Early stopping *)
  done;
  
  !best_ev, !best_score

let calculate_nrmse predictions actuals =
  let n = Tensor.size predictions 0 in
  let p = Tensor.size predictions 1 in
  
  let mean_actuals = Tensor.mean actuals ~dim:[0] ~keepdim:true in
  let errors = Tensor.(-) predictions actuals in
  let squared_errors = Tensor.mul errors errors in
  let sum_squared_errors = Tensor.sum squared_errors in
  
  let deviations = Tensor.(-) actuals mean_actuals in
  let squared_deviations = Tensor.mul deviations deviations in
  let sum_squared_deviations = Tensor.sum squared_deviations in
  
  let nrmse = sqrt (Tensor.float_value sum_squared_errors /. 
                   Tensor.float_value sum_squared_deviations) in
                   
  let component_errors = Array.init p (fun i ->
    let comp_errors = Tensor.select errors 1 i in
    let comp_devs = Tensor.select deviations 1 i in
    sqrt (Tensor.float_value (Tensor.sum (Tensor.mul comp_errors comp_errors)) /.
          Tensor.float_value (Tensor.sum (Tensor.mul comp_devs comp_devs)))
  ) in
  
  { nrmse; component_errors }
  
let optimize_embedding data ~max_dim ~k_neighbors =
  let best_ev = ref {
    dimensions = Array.make (Array.length data) 1;
    delays = Array.make (Array.length data) 1;
  } in
  let best_error = ref Float.infinity in
  
let try_embedding ev =
  let embedded = embed_multivariate data ev in
  let n = Tensor.size embedded 0 in
  let split = (3 * n) / 4 in
  
  let train = Tensor.narrow embedded 0 0 split in
  let test = Tensor.narrow embedded 0 split (n - split) in
  
  let predictions = Tensor.zeros [n - split; Array.length data] in
  for i = 0 to n - split - 1 do
    let point = Tensor.select test 0 i in
    for j = 0 to Array.length data - 1 do
      let pred = Models.predict train point 
        (Tensor.narrow embedded 0 0 split)
        ~k_neighbors in
      Tensor.set predictions [|i; j|] (Tensor.get pred [|0; 0|])
    done
  done;
  
  let error = calculate_nrmse predictions 
    (Tensor.narrow embedded 0 (split + 1) (n - split - 1)) in
  error.nrmse
in

(* Try all possible combinations up to max_dim *)
let candidates = generate_embeddings 
  (Array.length data) max_dim in

List.iter (fun ev ->
  let error = try_embedding ev in
  if error < !best_error then begin
    best_error := error;
    best_ev := ev
  end
) candidates;

!best_ev, !best_error


let correlation_matrix data =
  let n_vars = Array.length data in
  let result = Tensor.zeros [n_vars; n_vars] in
  
  for i = 0 to n_vars - 1 do
    for j = 0 to n_vars - 1 do
      let xi = standardize data.(i) in
      let xj = standardize data.(j) in
      let corr = Tensor.mean (Tensor.mul xi xj) in
      Tensor.set result [|i; j|] (Tensor.float_value corr)
    done
  done;
  result
  
let detect_redundancy data threshold =
  let corr_mat = correlation_matrix data in
  let n = Array.length data in
  let redundant = Array.make n false in
  
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      let corr = abs_float (Tensor.get corr_mat [|i; j|]) in
      if corr > threshold then
        redundant.(j) <- true
    done
  done;
  redundant
  
let filter_redundant data threshold =
  let redundant = detect_redundancy data threshold in
  Array.mapi (fun i var -> 
    if redundant.(i) then None else Some var)
    data
  |> Array.to_list
  |> List.filter_map (fun x -> x)
  |> Array.of_list

module Models = struct
  type model_type = ZeroOrder | LinearModel | WeightedLinear
  
  let validate_model_params model_type radius k_neighbors =
    match radius, k_neighbors with
    | Some r, _ when r <= 0. ->
        raise (ErrorHandling.InvalidParameters "Radius must be positive")
    | _, Some k when k <= 0 ->
        raise (ErrorHandling.InvalidParameters "k_neighbors must be positive")
    | None, None ->
        raise (ErrorHandling.InvalidParameters "Must specify either radius or k_neighbors")
    | _ -> ()
    
  let find_neighbors_radius embedded_data point radius =
    let n = Tensor.size embedded_data 0 in
    let distances = Tensor.zeros [n] in
    
    for i = 0 to n - 1 do
      let diff = Tensor.(-) 
        (Tensor.select embedded_data 0 i)
        point in
      let dist = Tensor.norm diff in
      Tensor.set distances [|i|] dist
    done;
    
    Tensor.nonzero (Tensor.lt distances (Tensor.of_float1 radius))
    
  let find_neighbors_knn embedded_data point k =
    let n = Tensor.size embedded_data 0 in
    let distances = Tensor.zeros [n] in
    
    for i = 0 to n - 1 do
      let diff = Tensor.(-) 
        (Tensor.select embedded_data 0 i)
        point in
      let dist = Tensor.norm diff in
      Tensor.set distances [|i|] dist
    done;
    
    let _, indices = Tensor.topk distances k ~largest:false in
    indices
    
  let predict embedded_data point targets ?model_type ?radius ?k_neighbors =
    validate_model_params 
      (Option.value ~default:LinearModel model_type)
      radius k_neighbors;
      
    let neighbors_idx = match radius, k_neighbors with
      | Some r, _ -> find_neighbors_radius embedded_data point r
      | _, Some k -> find_neighbors_knn embedded_data point k
      | _ -> assert false in
      
    let neighbors = Tensor.index_select embedded_data 0 neighbors_idx in
    let neighbor_targets = Tensor.index_select targets 0 neighbors_idx in
    
    match Option.value ~default:LinearModel model_type with
    | ZeroOrder -> 
        Tensor.mean neighbor_targets ~dim:[0] ~keepdim:true
    | LinearModel ->
        let x = Tensor.cat [neighbors; Tensor.ones [Tensor.size neighbors 0; 1]] 1 in
        let xt = Tensor.transpose x 0 1 in
        let xtx = Tensor.matmul xt x in
        let xty = Tensor.matmul xt neighbor_targets in
        Tensor.matmul 
          (Tensor.cat [point; Tensor.ones [1; 1]] 1)
          (Tensor.solve xtx xty)
    | WeightedLinear ->
        let distances = Tensor.cdist point neighbors in
        let weights = Tensor.exp (Tensor.mul distances (Tensor.of_float (-1.0))) in
        let w_sqrt = Tensor.sqrt weights in
        let wx = Tensor.mul neighbors (Tensor.expand_as w_sqrt neighbors) in
        let wy = Tensor.mul neighbor_targets (Tensor.expand_as w_sqrt neighbor_targets) in
        let xtx = Tensor.matmul (Tensor.transpose wx 0 1) wx in
        let xty = Tensor.matmul (Tensor.transpose wx 0 1) wy in
        Tensor.matmul point (Tensor.solve xtx xty)
end

module MonteCarlo = struct
  type simulation_result = {
    embedding: embedding_vector;
    nrmse: float;
    frequency: int;
  }
  
  let check_termination_criteria fnn_percentages threshold iterations =
    let stable_count = ref 0 in
    let prev_min = ref Float.infinity in
    
    List.iter (fun percentage ->
      if abs_float (percentage -. !prev_min) < threshold then
        incr stable_count
      else
        stable_count := 0;
      prev_min := min !prev_min percentage
    ) fnn_percentages;
    
    !stable_count >= iterations
    
  let run_trial system_gen params methods =
    let data = system_gen () in
    let scaled_data = scale_multivariate 
      data ~method_:`UnitInterval in
    let train_data, test_data = train_test_split 
      scaled_data ~train_ratio:0.75 in
    
    List.map (fun method_ ->
      match method_ with
      | `FNN1 -> 
          let ev, _ = find_optimal 
            train_data params.Parameters.max_dim params.Parameters.r in
          let predictions = embed_multivariate test_data ev in
          let error = calculate_nrmse predictions test_data in
          (`FNN1, ev, error.nrmse)
      | `FNN2 ->
          let ev, _ = find_optimal 
            train_data params.Parameters.max_dim params.Parameters.r in
          let predictions = embed_multivariate test_data ev in
          let error = calculate_nrmse predictions test_data in
          (`FNN2, ev, error.nrmse)
      | `PEM ->
          let ev, error = optimize_embedding 
            train_data 
            ~max_dim:params.Parameters.max_dim 
            ~k_neighbors:params.Parameters.k_neighbors in
          (`PEM, ev, error)
    ) methods
    
  let run_simulation ~system_gen ~n_trials ~methods ~params =
    let results = List.init n_trials (fun _ ->
      run_trial system_gen params methods) in
      
    (* Aggregate results *)
    let method_results = Hashtbl.create 3 in
    List.iter (fun trial_results ->
      List.iter (fun (method_, ev, nrmse) ->
        let current = 
          try Hashtbl.find method_results (method_, ev)
          with Not_found -> (0, 0.) in
        Hashtbl.replace method_results (method_, ev)
          (fst current + 1, snd current +. nrmse)
      ) trial_results
    ) results;
    
    (* Convert to final list *)
    Hashtbl.fold (fun (method_, ev) (freq, total_nrmse) acc ->
      let avg_nrmse = total_nrmse /. float_of_int freq in
      (method_, { embedding = ev; 
                 nrmse = avg_nrmse; 
                 frequency = freq }) :: acc
    ) method_results []
end

module CouplingAnalysis = struct
  type coupling_metric = {
    mutual_info: float;
    correlation: float;
    prediction_error: float;
  }
  
  let mutual_information x y bins =
    let n = Tensor.size x 0 in
    let joint_hist = Tensor.zeros [bins; bins] in
    let x_hist = Tensor.zeros [bins] in
    let y_hist = Tensor.zeros [bins] in
    
    (* Normalize data *)
    let x_norm = scale_to_unit_interval x in
    let y_norm = scale_to_unit_interval y in
    
    (* Fill histograms *)
    for i = 0 to n - 1 do
      let x_val = Tensor.get x_norm [|i|] in
      let y_val = Tensor.get y_norm [|i|] in
      let x_bin = min (bins - 1) (int_of_float (x_val *. float_of_int bins)) in
      let y_bin = min (bins - 1) (int_of_float (y_val *. float_of_int bins)) in
      
      let current = Tensor.get joint_hist [|x_bin; y_bin|] in
      Tensor.set joint_hist [|x_bin; y_bin|] (current +. 1.);
      
      let x_current = Tensor.get x_hist [|x_bin|] in
      Tensor.set x_hist [|x_bin|] (x_current +. 1.);
      
      let y_current = Tensor.get y_hist [|y_bin|] in
      Tensor.set y_hist [|y_bin|] (y_current +. 1.)
    done;
    
    (* Normalize and calculate MI *)
    let n_float = float_of_int n in
    let joint_prob = Tensor.div joint_hist (Tensor.of_float n_float) in
    let x_prob = Tensor.div x_hist (Tensor.of_float n_float) in
    let y_prob = Tensor.div y_hist (Tensor.of_float n_float) in
    
    let mi = ref 0. in
    for i = 0 to bins - 1 do
      for j = 0 to bins - 1 do
        let pxy = Tensor.get joint_prob [|i; j|] in
        if pxy > 0. then begin
          let px = Tensor.get x_prob [|i|] in
          let py = Tensor.get y_prob [|j|] in
          mi := !mi +. pxy *. log (pxy /. (px *. py))
        end
      done
    done;
    !mi

module Statistics = struct
  type summary_stats = {
    mean: float;
    std_dev: float;
    confidence_interval: float * float;
  }
  
  let confidence_interval values alpha =
    let n = Array.length values in 
    let mean = Array.fold_left (+.) 0. values /. float_of_int n in
    
    let variance = Array.fold_left (fun acc x ->
      acc +. (x -. mean) ** 2.0) 0. values /. float_of_int (n - 1) in
    let std_dev = sqrt variance in
    let std_error = std_dev /. sqrt (float_of_int n) in
    
    (* Using t-distribution for small samples *)
    let t_value = match n with
      | n when n < 30 -> 2.0 (* Approximation for small samples *)
      | _ -> 1.96 in (* Normal approximation for large samples *)
    
    let margin = t_value *. std_error in
    mean, (mean -. margin, mean +. margin)
    
  let summarize_results results =
    let nrmse_values = Array.map (fun r -> r.nrmse) results in
    let mean, (ci_low, ci_high) = confidence_interval nrmse_values 0.05 in
    let std_dev = sqrt (Array.fold_left (fun acc x ->
      acc +. (x -. mean) ** 2.0) 0. nrmse_values /. 
      float_of_int (Array.length nrmse_values - 1)) in
    
    { mean; std_dev; confidence_interval = (ci_low, ci_high) }
    
  let friedman_test results =
    let n_methods = List.length (List.hd results) in
    let n_trials = List.length results in
    
    (* Convert results to ranks *)
    let ranks = List.map (fun trial ->
      let sorted = List.sort (fun a b -> 
        compare (snd a) (snd b)) trial in
      List.mapi (fun i (method_, _) -> (method_, float_of_int (i + 1))) sorted
    ) results in
    
    (* Calculate rank sums *)
    let rank_sums = List.fold_left (fun acc trial ->
      List.fold_left (fun acc' (method_, rank) ->
        let current = try Hashtbl.find acc' method_ with Not_found -> 0. in
        Hashtbl.replace acc' method_ (current +. rank);
        acc'
      ) acc trial
    ) (Hashtbl.create n_methods) ranks in
    
    (* Calculate test statistic *)
    let chi_sq = 12. *. float_of_int n_trials /. 
                 (float_of_int (n_methods * (n_methods + 1))) *.
                 Hashtbl.fold (fun _ rank_sum acc ->
                   acc +. (rank_sum ** 2.0)
                 ) rank_sums 0. -.
                 3. *. float_of_int (n_trials * (n_methods + 1)) in
    
    chi_sq
end

module Optimization = struct
  type particle = {
    position: embedding_vector;
    velocity: float array;
    best_position: embedding_vector;
    best_score: float;
  }
  
  let init_particle nvars max_dim =
    let init_random_array size = 
      Array.init size (fun _ -> Random.float 2.0 -. 1.0) in
    
    let dims = Array.init nvars (fun _ -> 1 + Random.int max_dim) in
    let delays = Array.make nvars 1 in
    let ev = { dimensions = dims; delays } in
    
    { position = ev;
      velocity = init_random_array nvars;
      best_position = ev;
      best_score = Float.infinity }
      
  let update_particle particle global_best params =
    let new_velocity = Array.mapi (fun i v ->
      params.w *. v +.
      params.c1 *. Random.float 1.0 *. 
        float_of_int (particle.best_position.dimensions.(i) - 
                     particle.position.dimensions.(i)) +.
      params.c2 *. Random.float 1.0 *. 
        float_of_int (global_best.dimensions.(i) - 
                     particle.position.dimensions.(i))
    ) particle.velocity in
    
    let new_dims = Array.mapi (fun i dim ->
      max 1 (min dim (int_of_float 
        (float_of_int dim +. new_velocity.(i))))
    ) particle.position.dimensions in
    
    { particle with
      position = { dimensions = new_dims; 
                  delays = particle.position.delays };
      velocity = new_velocity }
      
  let optimize_pso data ~params =
    let particles = Array.init params.n_particles (fun _ ->
      init_particle (Array.length data) params.max_iterations) in
    
    let global_best = ref particles.(0).position in
    let global_best_score = ref Float.infinity in
    
    for _ = 1 to params.max_iterations do
      (* Evaluate particles *)
      Array.iter (fun particle ->
        let embedded = embed_multivariate data particle.position in
        let error = calculate_nrmse embedded data in
        
        if error.nrmse < particle.best_score then begin
          particle.best_position <- particle.position;
          particle.best_score <- error.nrmse;
          
          if error.nrmse < !global_best_score then begin
            global_best := particle.position;
            global_best_score := error.nrmse
          end
        end
      ) particles;
      
      (* Update particles *)
      Array.iter (fun particle ->
        let updated = update_particle particle !global_best params in
        particle.position <- updated.position;
        particle.velocity <- updated.velocity
      ) particles
    done;
    
    !global_best, !global_best_score
end

module Systems = struct
  let default_params = {
    coupling_strength = 0.0;
    noise_level = None;
    initial_conditions = None;
  }
  
  let add_noise data noise_level =
    match noise_level with
    | None -> data
    | Some level ->
        Array.map (fun series ->
          let noise = Tensor.randn (Tensor.size series) in
          Tensor.(add series (mul noise (float_value level)))
        ) data
  
  let generate_ikeda n ?params () =
    let params = Option.value ~default:default_params params in
    let rec ikeda_map z =
      let open Complex in
      let c = mul z (add one (div one (mul z (conj z)))) in
      mul (mul c (euler (mul I (neg (div (Float 6.) 
        (add one (mul z (conj z)))))))) (Float 0.9)
    in
    
    let init_z = match params.initial_conditions with
      | Some [|x; y|] -> Complex.{ re = x; im = y }
      | _ -> Complex.one in
      
    let series = Array.make n init_z in
    for i = 1 to n - 1 do
      series.(i) <- ikeda_map series.(i-1)
    done;
    
    let result = Array.init 2 (fun i ->
      let values = Array.map (fun z ->
        if i = 0 then Complex.re z else Complex.im z) series in
      Tensor.of_float1 values) in
      
    add_noise result params.noise_level
    
  let generate_henon n ?params () =
    let params = Option.value ~default:default_params params in
    let init_x, init_y = match params.initial_conditions with
      | Some [|x; y|] -> x, y
      | _ -> 0.1, 0.1 in
      
    let x = Tensor.zeros [n] in
    let y = Tensor.zeros [n] in
    
    Tensor.set x [|0|] init_x;
    Tensor.set y [|0|] init_y;
    
    for i = 1 to n - 1 do
      let xprev = Tensor.get x [|i-1|] in
      let yprev = Tensor.get y [|i-1|] in
      
      Tensor.set x [|i|] (1.4 -. xprev *. xprev +. yprev);
      Tensor.set y [|i|] (0.3 *. xprev)
    done;
    
    add_noise [|x; y|] params.noise_level

  let generate_kdr n ?params () =
    let params = Option.value ~default:default_params params in
    let m = [|[|0.49; 0.21|]; [|0.21; 0.70|]|] in
    let l = [|[|0.24; 0.27|]; [|0.27; 0.51|]|] in
    let w = [|6.36; 9.0|] in
    
    let init_state = match params.initial_conditions with
      | Some state when Array.length state = 4 -> state
      | _ -> [|0.1; 0.1; 0.1; 0.1|] in
      
    let x1 = Tensor.zeros [n] in
    let x2 = Tensor.zeros [n] in
    let y1 = Tensor.zeros [n] in
    let y2 = Tensor.zeros [n] in
    
    let update_state (x1, x2, y1, y2) =
      let x1' = x1 +. m.(0).(0) *. y1 +. m.(0).(1) *. y2 in
      let x2' = x2 +. m.(1).(0) *. y1 +. m.(1).(1) *. y2 in
      let y1' = l.(0).(0) *. y1 +. l.(0).(1) *. y2 +. w.(0) *. sin x1 in
      let y2' = l.(1).(0) *. y1 +. l.(1).(1) *. y2 +. w.(1) *. sin x2 in
      (x1', x2', y1', y2') in
    
    let state = ref (init_state.(0), init_state.(1), 
                    init_state.(2), init_state.(3)) in
                    
    for i = 0 to n - 1 do
      let (x1v, x2v, y1v, y2v) = !state in
      Tensor.set x1 [|i|] x1v;
      Tensor.set x2 [|i|] x2v;
      Tensor.set y1 [|i|] y1v;
      Tensor.set y2 [|i|] y2v;
      state := update_state !state
    done;
    
    add_noise [|x1; x2; y1; y2|] params.noise_level
    
  let generate_coupled_henon n ~params ~lattice_size 
      ~lattice_topology =
    let init_state = match params.initial_conditions with
      | Some state when Array.length state = lattice_size * 2 -> state
      | _ -> Array.make (lattice_size * 2) 0.1 in
      
    let series = Array.init lattice_size (fun _ -> 
      Tensor.zeros [n]) in
      
    let state = ref (Array.copy init_state) in
    
    for t = 0 to n - 1 do
      (* Store current state *)
      for i = 0 to lattice_size - 1 do
        Tensor.set series.(i) [|t|] !state.(i * 2)
      done;
      
      (* Update state *)
      let new_state = Array.make (lattice_size * 2) 0. in
      for i = 0 to lattice_size - 1 do
        let neighbors = get_neighbors lattice_topology i lattice_size in
        let coupling_sum = List.fold_left (fun acc j ->
          acc +. !state.(j * 2)) 0. neighbors in
        let num_neighbors = float_of_int (List.length neighbors) in
        
        let x = !state.(i * 2) in
        let y = !state.(i * 2 + 1) in
        
        let coupling_term = coupling_sum /. num_neighbors in
        let coupled_x = (1. -. params.coupling_strength) *. x +. 
                       params.coupling_strength *. coupling_term in
        
        new_state.(i * 2) <- 1.4 -. coupled_x *. coupled_x +. y;
        new_state.(i * 2 + 1) <- 0.3 *. x
      done;
      
      state := new_state
    done;
    
    add_noise series params.noise_level
end
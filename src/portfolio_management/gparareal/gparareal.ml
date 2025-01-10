open Torch

module StateTypes = struct
  type system_state = {
    values: Tensor.t;      (* State vector *)
    dimension: int;        (* System dimension *)
    time: float;          (* Current time *)
    derivatives: Tensor.t option;  (* Cached derivatives *)
  }

  type time_mesh = {
    points: float array;   (* Mesh points *)
    fine_dt: float;       (* Fine timestep *)
    coarse_dt: float;     (* Coarse timestep *)
    slice_indices: int array;  (* Start indices of slices *)
  }

  type solver_config = {
    order: int;           (* Method order *)
    method_name: string;  (* Method identifier *)
    adaptive: bool;       (* Whether solver uses adaptive stepping *)
    tol: float;          (* Local error tolerance *)
  }

  type solution_history = {
    states: system_state array;
    times: float array;
    iteration: int;
    from_fine: bool;
  }
end

module TimeMesh = struct
  type slice = {
    t_start: float;
    t_end: float;
    fine_points: float array;
    coarse_points: float array;
  }

  let create_mesh ~t0 ~tf ~num_slices ~fine_points_per_slice ~coarse_points_per_slice =
    let delta_t = (tf -. t0) /. float_of_int num_slices in
    let fine_dt = delta_t /. float_of_int fine_points_per_slice in
    let coarse_dt = delta_t /. float_of_int coarse_points_per_slice in
    
    let total_fine = num_slices * fine_points_per_slice + 1 in
    let total_coarse = num_slices * coarse_points_per_slice + 1 in
    
    let points = Array.init total_fine (fun i -> 
      t0 +. float_of_int i *. fine_dt) in
    
    let slice_indices = Array.init (num_slices + 1) (fun i ->
      i * fine_points_per_slice) in
    
    {points; fine_dt; coarse_dt; slice_indices}

  let get_slice mesh idx =
    let start_idx = mesh.slice_indices.(idx) in
    let end_idx = mesh.slice_indices.(idx + 1) in
    let t_start = mesh.points.(start_idx) in
    let t_end = mesh.points.(end_idx) in
    
    let fine_points = Array.sub mesh.points start_idx (end_idx - start_idx + 1) in
    let n_coarse = int_of_float ((t_end -. t_start) /. mesh.coarse_dt) + 1 in
    let coarse_points = Array.init n_coarse (fun i ->
      t_start +. float_of_int i *. mesh.coarse_dt) in
    
    {t_start; t_end; fine_points; coarse_points}

  let interpolate values times target_times =
    let n_target = Array.length target_times in
    let result = Array.make n_target (Tensor.zeros_like values.(0)) in
    
    for i = 0 to n_target - 1 do
      let t = target_times.(i) in
      let idx = ref 0 in
      while !idx < Array.length times - 1 && times.(!idx + 1) <= t do incr idx done;
      
      if !idx = Array.length times - 1 then
        result.(i) <- values.(!idx)
      else begin
        let alpha = (t -. times.(!idx)) /. (times.(!idx + 1) -. times.(!idx)) in
        result.(i) <- Tensor.(values.(!idx) + Scalar alpha * (values.(!idx + 1) - values.(!idx)))
      end
    done;
    result
end

module SolutionStorage = struct
  type stored_solution = {
    state: system_state;
    iteration: int;
    error_est: float option;
    method_used: string;
  }

  type history = {
    solutions: stored_solution array array;  (* [iteration][slice] *)
    corrections: Tensor.t array array;      (* [iteration][slice] *)
    max_hist: int;
    current_size: int;
  }

  let create_history max_hist num_slices =
    {
      solutions = Array.make_matrix max_hist num_slices
        {state = {
           values = Tensor.zeros [|1|];
           dimension = 1;
           time = 0.0;
           derivatives = None
         };
         iteration = -1;
         error_est = None;
         method_used = ""
        };
      corrections = Array.make_matrix max_hist (num_slices - 1)
        (Tensor.zeros [|1|]);
      max_hist;
      current_size = 0;
    }

  let add_solution history ~iteration ~slice ~solution =
    if iteration < history.max_hist then begin
      history.solutions.(iteration).(slice) <- solution;
      if iteration >= history.current_size then
        history.current_size <- iteration + 1
    end

  let add_correction history ~iteration ~slice ~correction =
    if iteration < history.max_hist then
      history.corrections.(iteration).(slice) <- correction

  let get_solution history ~iteration ~slice =
    if iteration < history.current_size then
      Some history.solutions.(iteration).(slice)
    else None

  let get_nearby_solutions history ~time ~window =
    let solutions = ref [] in
    for i = 0 to history.current_size - 1 do
      for j = 0 to Array.length history.solutions.(i) - 1 do
        let sol = history.solutions.(i).(j) in
        if abs_float (sol.state.time -. time) <= window then
          solutions := sol :: !solutions
      done
    done;
    Array.of_list !solutions
end

module Solver = struct
  type t = {
    config: solver_config;
    step: float -> float -> Tensor.t -> Tensor.t;  (* (t, dt, y) -> y_next *)
    error_est: (float -> float -> Tensor.t -> float) option;
  }

  (* RK4 coefficients *)
  let rk4_a = [|[|0.; 0.; 0.; 0.|];
                [|0.5; 0.; 0.; 0.|];
                [|0.; 0.5; 0.; 0.|];
                [|0.; 0.; 1.; 0.|]|]
  let rk4_b = [|1./.6.; 1./.3.; 1./.3.; 1./.6.|]
  let rk4_c = [|0.; 0.5; 0.5; 1.0|]

  (* Create RK4 solver *)
  let make_rk4 ~f ~tol =
    let step t dt y =
      let open Tensor in
      let k1 = f t y in
      let k2 = f (t +. dt *. 0.5) (y + dt/.(Scalar 2.) * k1) in
      let k3 = f (t +. dt *. 0.5) (y + dt/.(Scalar 2.) * k2) in
      let k4 = f (t +. dt) (y + dt * k3) in
      y + dt/.(Scalar 6.) * (k1 + Scalar 2. * k2 + Scalar 2. * k3 + k4)
    in
    
    (* Embedded method *)
    let error_est t dt y =
      let y1 = step t dt y in
      let y2 = step t (dt/.2.) y in
      let y2 = step (t +. dt/.2.) (dt/.2.) y2 in
      Tensor.(abs (y1 - y2))
      |> Tensor.float_value
      |> sqrt
    in
    
    {
      config = {
        order = 4;
        method_name = "RK4";
        adaptive = true;
        tol;
      };
      step;
      error_est = Some error_est;
    }

  (* Create Forward Euler solver *)
  let make_euler ~f ~tol =
    let step t dt y =
      let open Tensor in
      y + dt * (f t y)
    in
    {
      config = {
        order = 1;
        method_name = "Euler";
        adaptive = false;
        tol;
      };
      step;
      error_est = None;
    }

  (* Solve over interval with optional adaptation *)
  let solve solver ~t0 ~tf ~init_state =
    let dt_init = (tf -. t0) /. 100. in  (* Initial step size *)
    let rec solve_adaptive t y solutions times dt =
      if t >= tf then
        (Array.of_list (List.rev solutions),
         Array.of_list (List.rev times))
      else begin
        let dt = min dt (tf -. t) in
        let y_next = solver.step t dt y in
        
        match solver.error_est with
        | Some est_fn ->
            let err = est_fn t dt y in
            if err <= solver.config.tol then
              solve_adaptive (t +. dt) y_next (y_next :: solutions) 
                ((t +. dt) :: times) dt
            else
              let dt_new = dt *. 0.9 *. (solver.config.tol /. err) ** 0.2 in
              solve_adaptive t y solutions times dt_new
        | None ->
            solve_adaptive (t +. dt) y_next (y_next :: solutions)
              ((t +. dt) :: times) dt
      end
    in
    solve_adaptive t0 init_state [init_state] [t0] dt_init
end

(* Gaussian Process for correction prediction *)
module GP = struct
  type training_point = {
    input: Tensor.t;
    output: Tensor.t;
    time: float;
  }

  type t = {
    points: training_point array;
    hyperparams: Tensor.t;  (* [sigma; length_scale; time_scale] *)
    dimension: int;
  }

  let create ~dimension =
    {
      points = [||];
      hyperparams = Tensor.of_float1 [|1.0; 1.0; 1.0|];
      dimension;
    }

  (* Kernel function with ARD (Automatic Relevance Determination) *)
  let kernel gp x1 x2 t1 t2 =
    let sigma = Tensor.get gp.hyperparams 0 in
    let length_scale = Tensor.get gp.hyperparams 1 in
    let time_scale = Tensor.get gp.hyperparams 2 in
    
    let spatial_diff = Tensor.(x1 - x2) in
    let spatial_dist = Tensor.(sum (spatial_diff * spatial_diff)) 
      |> Tensor.float_value in
    let time_dist = ((t1 -. t2) /. time_scale) ** 2. in
    
    sigma *. exp (-0.5 *. (spatial_dist /. (length_scale ** 2.) +. time_dist))

  let build_matrices gp x t =
    let n = Array.length gp.points in
    let k = Tensor.zeros [|1; n|] in
    let k_star = Tensor.zeros [|n; n|] in
    
    (* Build K matrix *)
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let kij = kernel gp 
          gp.points.(i).input gp.points.(j).input
          gp.points.(i).time gp.points.(j).time in
        Tensor.set k_star [|i; j|] kij
      done
    done;
    
    (* Build k vector *)
    for i = 0 to n-1 do
      let ki = kernel gp x gp.points.(i).input t gp.points.(i).time in
      Tensor.set k [|0; i|] ki
    done;
    
    k, k_star

  let add_point gp ~input ~output ~time =
    let point = {input; output; time} in
    {gp with points = Array.append gp.points [|point|]}

  let predict gp ~x ~t =
    if Array.length gp.points = 0 then
      Tensor.zeros [|gp.dimension|],
      Tensor.zeros [|gp.dimension; gp.dimension|]
    else begin
      let k, k_star = build_matrices gp x t in
      let k_star_inv = Tensor.inverse k_star in
      
      let y = Tensor.stack 
        (Array.map (fun p -> p.output) gp.points |> Array.to_list) 0 in
      
      let mean = Tensor.(matmul k (matmul k_star_inv y)) in
      let var = Tensor.(
        Scalar 1.0 - matmul k (matmul k_star_inv (transpose k 0 1))
      ) in
      
      mean, var
    end

  (* Optimize hyperparameters using log marginal likelihood *)
  let optimize gp =
    let n = Array.length gp.points in
    if n = 0 then gp
    else begin
      let y = Tensor.stack 
        (Array.map (fun p -> p.output) gp.points |> Array.to_list) 0 in
      
      let objective params =
        let gp = {gp with hyperparams = params} in
        let _, k = build_matrices gp 
          gp.points.(0).input gp.points.(0).time in
        
        let k_inv = Tensor.inverse k in
        let term1 = Tensor.(
          -0.5 * matmul (matmul (transpose y 0 1) k_inv) y
        ) in
        let term2 = -0.5 *. log (Tensor.det k) in
        let term3 = -0.5 *. float_of_int n *. log (2. *. Float.pi) in
        
        Tensor.float_value term1 +. term2 +. term3
      in
      
      (* Simple gradient descent *)
      let params = ref gp.hyperparams in
      let learning_rate = 0.01 in
      
      for _ = 1 to 100 do
        let p = (!params).requires_grad () in
        let loss = objective p in
        backward loss;
        params := Tensor.(!params + Scalar learning_rate * grad p)
      done;
      
      {gp with hyperparams = !params}
    end
end

module Correction = struct
  type t = {
    value: Tensor.t;
    input_state: Tensor.t;
    time: float;
    iteration: int;
    from_gp: bool;
    quality_score: float option;
  }

  let create ~value ~input_state ~time ~iteration ~from_gp ~quality_score =
    {value; input_state; time; iteration; from_gp; quality_score}

  (* Correction database for efficient storage and retrieval *)
  module Database = struct
    type t = {
      corrections: t array;
      times: float array;
      max_size: int;
      current_size: int;
      dimension: int;
    }

    let create ~max_size ~dimension = {
      corrections = Array.make max_size 
        {value = Tensor.zeros [|dimension|];
         input_state = Tensor.zeros [|dimension|];
         time = 0.0;
         iteration = -1;
         from_gp = false;
         quality_score = None};
      times = Array.make max_size 0.0;
      max_size;
      current_size = 0;
      dimension;
    }

    let add db correction =
      if db.current_size < db.max_size then begin
        db.corrections.(db.current_size) <- correction;
        db.times.(db.current_size) <- correction.time;
        {db with current_size = db.current_size + 1}
      end else db

    let get_nearby db ~time ~window =
      let nearby = ref [] in
      for i = 0 to db.current_size - 1 do
        if abs_float (db.times.(i) -. time) <= window then
          nearby := db.corrections.(i) :: !nearby
      done;
      Array.of_list !nearby

    let filter_by_quality db ~min_quality =
      let filtered = ref [] in
      for i = 0 to db.current_size - 1 do
        match db.corrections.(i).quality_score with
        | Some q when q >= min_quality ->
            filtered := db.corrections.(i) :: !filtered
        | _ -> ()
      done;
      Array.of_list !filtered
  end

  (* Quality assessment of corrections *)
  let assess_quality ~correction ~true_value =
    let error = Tensor.(abs (correction.value - true_value)) in
    let error_norm = Tensor.float_value (Tensor.mean error) in
    let quality = exp (-. error_norm) in  (* Quality score between 0 and 1 *)
    {correction with quality_score = Some quality}
end

(* Vector-valued GP implementation *)
module VectorGP = struct
  type t = {
    gps: GP.t array;  (* Array of GPs, one per dimension *)
    dimension: int;
  }

  let create ~dimension =
    {
      gps = Array.init dimension (fun _ -> GP.create ~dimension:1);
      dimension;
    }

  let add_point t ~input ~output ~time =
    let new_gps = Array.mapi (fun i gp ->
      let output_i = Tensor.select output ~dim:0 ~index:i in
      GP.add_point gp ~input ~output:output_i ~time
    ) t.gps in
    {t with gps = new_gps}

  let predict t ~x ~t =
    let predictions = Array.map (fun gp ->
      GP.predict gp ~x ~t
    ) t.gps in
    let means = Array.map fst predictions in
    let vars = Array.map snd predictions in
    Tensor.cat (Array.to_list means) 0,
    Tensor.cat (Array.to_list vars) 0

  let optimize t =
    let new_gps = Array.map GP.optimize t.gps in
    {t with gps = new_gps}
end

(* Legacy data handling *)
module LegacyData = struct
  type run_data = {
    states: StateTypes.system_state array;
    corrections: Correction.t array;
    times: float array;
    run_id: int;
  }

  type t = {
    runs: run_data array;
    dimension: int;
  }

  let create ~dimension = {
    runs = [||];
    dimension;
  }

  let add_run t run =
    {t with runs = Array.append t.runs [|run|]}

  let get_nearby_corrections t ~time ~window =
    let corrections = ref [] in
    Array.iter (fun run ->
      Array.iter (fun corr ->
        if abs_float (corr.time -. time) <= window then
          corrections := corr :: !corrections
      ) run.corrections
    ) t.runs;
    Array.of_list !corrections
end

module GParareal = struct
  type config = {
    t0: float;
    tf: float;
    num_slices: int;
    dimension: int;
    tol: float;
    max_iterations: int;
    fine_points_per_slice: int;
    coarse_points_per_slice: int;
  }

  type t = {
    config: config;
    mesh: TimeMesh.time_mesh;
    fine_solver: Solver.t;
    coarse_solver: Solver.t;
    vector_gp: VectorGP.t;
    correction_db: Correction.Database.t;
    solution_history: SolutionStorage.history;
    legacy_data: LegacyData.t option;
  }

  let create ~f ~dimension ~t0 ~tf ~num_slices ~tol ?legacy_data () =
    let config = {
      t0; tf; num_slices; dimension; tol;
      max_iterations = num_slices * 2;
      fine_points_per_slice = 20;
      coarse_points_per_slice = 4;
    } in
    
    let mesh = TimeMesh.create_mesh
      ~t0 ~tf ~num_slices
      ~fine_points_per_slice:config.fine_points_per_slice
      ~coarse_points_per_slice:config.coarse_points_per_slice in
      
    {
      config;
      mesh;
      fine_solver = Solver.make_rk4 ~f ~tol;
      coarse_solver = Solver.make_euler ~f ~tol;
      vector_gp = VectorGP.create ~dimension;
      correction_db = Correction.Database.create 
        ~max_size:(num_slices * 10) ~dimension;
      solution_history = SolutionStorage.create_history 
        (num_slices * 2) num_slices;
      legacy_data;
    }

  let solve_slice t solver state slice_idx =
    let slice = TimeMesh.get_slice t.mesh slice_idx in
    let dt = if solver.Solver.config.order > 2 then
        t.mesh.fine_dt else t.mesh.coarse_dt in
    
    let values, times = Solver.solve solver
      ~t0:slice.t_start
      ~tf:slice.t_end
      ~init_state:state.StateTypes.values in
    
    {StateTypes.
      values = Array.get values (Array.length values - 1);
      dimension = state.dimension;
      time = slice.t_end;
      derivatives = None;
    }

  let initial_iteration t init_state =
    let state = {StateTypes.
      values = init_state;
      dimension = t.config.dimension;
      time = t.config.t0;
      derivatives = None;
    } in
    
    let solutions = Array.make t.config.num_slices state in
    solutions.(0) <- state;
    
    (* Run coarse solver sequentially *)
    for i = 1 to t.config.num_slices - 1 do
      solutions.(i) <- solve_slice t t.coarse_solver solutions.(i-1) (i-1)
    done;
    
    (* Store initial solutions *)
    for i = 0 to t.config.num_slices - 1 do
      SolutionStorage.add_solution t.solution_history
        ~iteration:0
        ~slice:i
        ~solution:{
          state = solutions.(i);
          iteration = 0;
          error_est = None;
          method_used = t.coarse_solver.config.method_name;
        }
    done;
    
    solutions

  let compute_correction t ~fine_sol ~coarse_sol ~input_state ~time ~iteration =
    let correction = Correction.create
      ~value:Tensor.(fine_sol.StateTypes.values - coarse_sol.StateTypes.values)
      ~input_state:input_state.StateTypes.values
      ~time
      ~iteration
      ~from_gp:false
      ~quality_score:None in
    
    (* Update correction database *)
    let db = Correction.Database.add t.correction_db correction in
    {t with correction_db = db}, correction

  let predict_correction t state time =
    (* Get nearby corrections from legacy data *)
    let legacy_corrections = match t.legacy_data with
      | Some legacy -> LegacyData.get_nearby_corrections legacy ~time ~window:0.1
      | None -> [||]
    in
    
    (* Get recent corrections from database *)
    let recent_corrections = 
      Correction.Database.get_nearby t.correction_db ~time ~window:0.1 in
    
    (* Update GP with all available corrections *)
    let vector_gp = ref t.vector_gp in
    Array.iter (fun corr ->
      vector_gp := VectorGP.add_point !vector_gp
        ~input:corr.input_state
        ~output:corr.value
        ~time:corr.time
    ) (Array.append legacy_corrections recent_corrections);
    
    (* Predict correction *)
    let mean, var = VectorGP.predict !vector_gp 
      ~x:state.StateTypes.values 
      ~t:time in
    
    let correction = Correction.create
      ~value:mean
      ~input_state:state.StateTypes.values
      ~time
      ~iteration:0
      ~from_gp:true
      ~quality_score:None in
    
    {t with vector_gp = !vector_gp}, correction

  let iterate t ~k ~prev_solutions =
    let n = Array.length prev_solutions in
    let new_solutions = Array.copy prev_solutions in
    let t = ref t in
    
    (* Run fine solutions and collect corrections in parallel *)
    for i = 0 to n-2 do
      (* Get fine solution *)
      let fine_sol = solve_slice !t !t.fine_solver prev_solutions.(i) i in
      
      (* Compute and store correction *)
      let new_t, correction = compute_correction !t
        ~fine_sol
        ~coarse_sol:prev_solutions.(i+1)
        ~input_state:prev_solutions.(i)
        ~time:prev_solutions.(i).time
        ~iteration:k in
      t := new_t;
      
      (* Store fine solution *)
      SolutionStorage.add_solution !t.solution_history
        ~iteration:k
        ~slice:i
        ~solution:{
          state = fine_sol;
          iteration = k;
          error_est = None;
          method_used = !t.fine_solver.config.method_name;
        }
    done;
    
    (* Update solutions using GP-corrected predictor *)
    for j = 1 to n-1 do
      (* Get coarse prediction *)
      let coarse_next = solve_slice !t !t.coarse_solver new_solutions.(j-1) (j-1) in
      
      (* Get GP correction *)
      let new_t, gp_correction = predict_correction !t new_solutions.(j-1) coarse_next.time in
      t := new_t;
      
      (* Apply correction *)
      new_solutions.(j) <- {coarse_next with
        values = Tensor.(coarse_next.values + gp_correction.value)
      };
      
      (* Store corrected solution *)
      SolutionStorage.add_solution !t.solution_history
        ~iteration:k
        ~slice:j
        ~solution:{
          state = new_solutions.(j);
          iteration = k;
          error_est = None;
          method_used = "GParareal";
        }
    done;
    
    (* Check convergence *)
    let converged = ref true in
    for j = 1 to n-1 do
      let diff = Tensor.(
        abs (new_solutions.(j).values - prev_solutions.(j).values)
      ) in
      if Tensor.float_value (Tensor.mean diff) > !t.config.tol then
        converged := false
    done;
    
    new_solutions, !converged, !t

  let solve t init_state =
    (* Initial iteration *)
    let solutions = initial_iteration t init_state in
    let k = ref 1 in
    let converged = ref false in
    let t = ref t in
    
    (* Main iteration loop *)
    while not !converged && !k < !t.config.max_iterations do
      let new_sols, conv, new_t = iterate !t ~k:!k ~prev_solutions:solutions in
      Array.blit new_sols 0 solutions 0 (Array.length solutions);
      converged := conv;
      t := new_t;
      incr k
    done;
    
    !t, solutions, !k

  (* Extract full solution history *)
  let get_solution_history t =
    let history = Array.init t.config.max_iterations (fun i ->
      Array.init t.config.num_slices (fun j ->
        match SolutionStorage.get_solution t.solution_history ~iteration:i ~slice:j with
        | Some sol -> sol
        | None -> failwith "Missing solution in history"
      )
    ) in
    history
end
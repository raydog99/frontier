open Torch

module Solver = struct
  type t = [
    | `Euler
    | `RK4
    | `AdaptiveHeun of {rtol: float; atol: float}
    | `DormandPrince of {rtol: float; atol: float}
  ]
end

type parameters = float array

type state = {
  u: Tensor.t;
  v: Tensor.t option;
  spatial_dims: int * int;
}

type scattering_config = {
  orders: int;
  orientations: int;
  scales: int;
}

type node_config = {
  hidden_dims: int array;
  activation: Tensor.t -> Tensor.t;
  dt: float;
  solver: Solver.t;
  dropout_rate: float;
}

let make_gaussian_filter ~sigma ~size =
  let x = Tensor.linspace ~start:(-. float_of_int size /. 2.)
                               ~end_:(float_of_int size /. 2.)
                               size in
  let y = Tensor.linspace ~start:(-. float_of_int size /. 2.)
                               ~end_:(float_of_int size /. 2.)
                               size in
  let grid = Tensor.meshgrid [x; y] in
  let x_grid = Tensor.get grid 0 in
  let y_grid = Tensor.get grid 1 in
  let r2 = Tensor.add
    (Tensor.mul x_grid x_grid)
    (Tensor.mul y_grid y_grid) in
  Tensor.exp (Tensor.div_scalar r2 (-2. *. sigma *. sigma))

let make_morlet_filter ~scale ~orientation ~size =
  let omega = 2. *. Float.pi in
  let x = Tensor.linspace ~start:(-. size /. 2.) ~end_:(size /. 2.) size in
  let y = Tensor.linspace ~start:(-. size /. 2.) ~end_:(size /. 2.) size in
  let grid = Tensor.meshgrid [x; y] in
  let rot_x = (Tensor.get grid 0 |> fun x ->
               Tensor.mul_scalar x (cos orientation)) |>
              Tensor.add
              (Tensor.get grid 1 |> fun y ->
               Tensor.mul_scalar y (sin orientation)) in
  let rot_y = (Tensor.get grid 0 |> fun x ->
               Tensor.mul_scalar x (-. sin orientation)) |>
              Tensor.add
              (Tensor.get grid 1 |> fun y ->
               Tensor.mul_scalar y (cos orientation)) in
  let gauss = Tensor.exp
    (Tensor.div
      (Tensor.add
        (Tensor.mul rot_x rot_x)
        (Tensor.mul rot_y rot_y))
      (-2. *. scale *. scale)) in
  let wave = Tensor.cos
    (Tensor.div_scalar rot_x (scale /. omega)) in
  Tensor.mul gauss wave

let laplacian x =
  let padded = Tensor.pad x ~pad:[|1;1;1;1|] ~mode:"replicate" in
  let center = Tensor.slice padded ~dim:0 ~start:1 ~end_:(-1) |>
              Tensor.slice ~dim:1 ~start:1 ~end_:(-1) in
  let top = Tensor.slice padded ~dim:0 ~start:0 ~end_:(-2) |>
            Tensor.slice ~dim:1 ~start:1 ~end_:(-1) in
  let bottom = Tensor.slice padded ~dim:0 ~start:2 ~end_:None |>
              Tensor.slice ~dim:1 ~start:1 ~end_:(-1) in
  let left = Tensor.slice padded ~dim:0 ~start:1 ~end_:(-1) |>
             Tensor.slice ~dim:1 ~start:0 ~end_:(-2) in
  let right = Tensor.slice padded ~dim:0 ~start:1 ~end_:(-1) |>
              Tensor.slice ~dim:1 ~start:2 ~end_:None in
  let sum_neighbors = Tensor.add (Tensor.add top bottom)
                                     (Tensor.add left right) in
  Tensor.sub sum_neighbors (Tensor.mul_scalar center 4.)

module ScatteringTransform = struct
  type t = {
    filter_bank: {
      morlet_filters: Tensor.t array array;
      gaussian: Tensor.t;
      size: int;
    };
    orders: int;
    orientations: int;
    scales: int;
  }

  type selection_strategy = {
    energy_threshold: float;
    correlation_threshold: float;
    max_coefficients: int;
  }

  let init_filters ~size ~orientations ~scales =
    let morlet_filters = Array.make_matrix scales orientations (Tensor.zeros [|1|]) in
    for s = 0 to scales - 1 do
      for o = 0 to orientations - 1 do
        let scale = 2. ** float_of_int s in
        let orientation = Float.pi *. float_of_int o /. float_of_int orientations in
        morlet_filters.(s).(o) <- make_morlet_filter ~scale ~orientation ~size
      done
    done;
    let gaussian = make_gaussian_filter ~sigma:1.0 ~size in
    {
      filter_bank = {morlet_filters; gaussian; size};
      orders;
      orientations;
      scales;
    }

  let rec scatter input filter_bank depth max_depth =
    if depth >= max_depth then []
    else begin
      let s0 = if depth = 0 then
        [Tensor.conv2d input filter_bank.gaussian ~padding:1 
         |> fun x -> Tensor.mean x ~dim:[0; 1]]
      else [] in

      let s1 = Array.to_list (Array.mapi (fun s scale_filters ->
        Array.to_list (Array.mapi (fun o filter ->
          let filtered = Tensor.conv2d input filter ~padding:1 in
          let modulus = Tensor.abs filtered in
          Tensor.mean modulus ~dim:[0; 1]
        ) scale_filters)
      ) filter_bank.morlet_filters) |> List.concat in

      let s2 = if depth < max_depth - 1 then
        let modulus_field = Array.fold_left (fun acc scale_filters ->
          Array.fold_left (fun acc filter ->
            let filtered = Tensor.conv2d input filter ~padding:1 in
            Tensor.abs filtered :: acc
          ) acc scale_filters
        ) [] filter_bank.morlet_filters in
        List.map (fun x -> scatter x filter_bank (depth + 1) max_depth |> List.concat)
          modulus_field |> List.concat
      else [] in

      s0 @ s1 @ s2
    end

  let forward t input =
    let coeffs = scatter input t.filter_bank 0 t.orders in
    Tensor.cat (Array.of_list coeffs) ~dim:0

  let select_coefficients coeffs strategy =
    let compute_energy coeff =
      Tensor.pow coeff 2. |> 
      Tensor.sum |> 
      Tensor.to_float0_exn in

    let compute_correlation a b =
      let a_norm = Tensor.norm a in
      let b_norm = Tensor.norm b in
      let dot = Tensor.dot a b in
      Tensor.div dot (Tensor.mul a_norm b_norm)
      |> Tensor.to_float0_exn in
    
    let total_energy = List.fold_left (fun acc c -> acc +. compute_energy c) 0. coeffs in
    let thresh_energy = total_energy *. strategy.energy_threshold in

    (* Sort by energy *)
    let sorted_coeffs = List.sort (fun a b ->
      compare (compute_energy b) (compute_energy a)
    ) coeffs in

    (* Select coefficients *)
    let rec select acc_energy selected = function
      | [] -> selected
      | c :: rest ->
          if List.length selected >= strategy.max_coefficients then selected
          else if acc_energy >= thresh_energy then selected
          else
            let is_redundant = List.exists (fun s ->
              abs_float (compute_correlation c s) > strategy.correlation_threshold
            ) selected in
            if is_redundant then
              select acc_energy selected rest
            else
              let e = compute_energy c in
              select (acc_energy +. e) (c :: selected) rest in

    select 0. [] sorted_coeffs
end

module NeuralODE = struct
  type t = {
    layers: (Tensor.t -> Tensor.t) array;
    params: Tensor.t array;
    augment_dim: int option;
    regularization: float;
    solver: Solver.t;
  }

  let make ~config ~augment_dim ~regularization =
    let input_dim = match augment_dim with
      | None -> config.hidden_dims.(0)
      | Some d -> config.hidden_dims.(0) + d in
    
    let dims = Array.copy config.hidden_dims in
    dims.(0) <- input_dim;
    
    let layers = Array.make (Array.length dims - 1) (fun x -> x) in
    let params = Array.make (Array.length dims - 1) Tensor.zeros in
    
    for i = 0 to Array.length layers - 1 do
      let fan_in = dims.(i) in
      let fan_out = dims.(i+1) in
      let bound = sqrt (6. /. float_of_int (fan_in + fan_out)) in
      
      let w = Tensor.uniform2 dims.(i) dims.(i+1) 
        ~low:(-.bound) ~high:bound in
      let b = Tensor.zeros [|dims.(i+1)|] in
      
      params.(i) <- Tensor.cat [|w; b|] ~dim:0;
      layers.(i) <- (fun x ->
        let wx = Tensor.mm x w in
        let wxb = Tensor.add wx b in
        if i < Array.length layers - 1 then
          config.activation wxb
        else wxb)
    done;
    
    {layers; params; augment_dim; regularization; solver = config.solver}

  let forward node x t =
    (* Augment state with parameters if configured *)
    let x = match node.augment_dim with
      | None -> x
      | Some d -> 
          let aug = Tensor.zeros [|Tensor.shape x |> Array.get_unsafe 0; d|] in
          Tensor.cat [x; aug] ~dim:1 in
    
    Array.fold_left (fun acc f -> f acc) x node.layers

  (* Adaptive step size control *)
  let adaptive_step ~rtol ~atol ~dt y1 y2 =
    let err = Tensor.sub y1 y2 in
    let tol = Tensor.add 
      (Tensor.mul_scalar (Tensor.abs y1) rtol)
      (Tensor.scalar_tensor atol) in
    let err_ratio = Tensor.div err tol |> 
                   Tensor.max |> 
                   Tensor.to_float0_exn in
    
    let safety = 0.9 in
    let new_dt = dt *. safety *. (1. /. err_ratio) ** 0.2 in
    
    if err_ratio <= 1. then Some (new_dt, y1)
    else None

  (* Integration methods *)
  let euler_step node x t dt =
    let dx = forward node x t in
    Tensor.add x (Tensor.mul_scalar dx dt)

  let rk4_step node x t dt =
    let k1 = forward node x t in
    let k2 = forward node (Tensor.add x (Tensor.mul_scalar k1 (dt/.2.))) 
                         (t +. dt/.2.) in
    let k3 = forward node (Tensor.add x (Tensor.mul_scalar k2 (dt/.2.))) 
                         (t +. dt/.2.) in
    let k4 = forward node (Tensor.add x (Tensor.mul_scalar k3 dt)) 
                         (t +. dt) in
    
    let weighted_sum = Tensor.add_list [
      k1;
      Tensor.mul_scalar k2 2.;
      Tensor.mul_scalar k3 2.;
      k4
    ] in
    Tensor.add x (Tensor.mul_scalar weighted_sum (dt/.6.))

  let dopri5_step node x t dt =
    (* Coefficients *)
    let a = [|
      [||];
      [|1./.5.|];
      [|3./.40.; 9./.40.|];
      [|44./.45.; -56./.15.; 32./.9.|];
      [|19372./.6561.; -25360./.2187.; 64448./.6561.; -212./.729.|];
      [|9017./.3168.; -355./.33.; 46732./.5247.; 49./.176.; -5103./.18656.|];
      [|35./.384.; 0.; 500./.1113.; 125./.192.; -2187./.6784.; 11./.84.|]
    |] in
    let b = [|35./.384.; 0.; 500./.1113.; 125./.192.; -2187./.6784.; 11./.84.; 0.|] in
    let b_star = [|5179./.57600.; 0.; 7571./.16695.; 393./.640.;
                   -92097./.339200.; 187./.2100.; 1./.40.|] in
    
    let k = Array.make 7 (Tensor.zeros_like x) in
    k.(0) <- forward node x t;
    
    for i = 1 to 6 do
      let sum = ref (Tensor.zeros_like x) in
      for j = 0 to i-1 do
        sum := Tensor.add !sum 
          (Tensor.mul_scalar k.(j) (a.(i).(j) *. dt))
      done;
      let x_new = Tensor.add x !sum in
      k.(i) <- forward node x_new (t +. dt)
    done;
    
    let y1 = ref x in
    let y2 = ref x in
    for i = 0 to 6 do
      y1 := Tensor.add !y1 
        (Tensor.mul_scalar k.(i) (b.(i) *. dt));
      y2 := Tensor.add !y2 
        (Tensor.mul_scalar k.(i) (b_star.(i) *. dt))
    done;
    !y1, !y2

  let solve node x t0 t1 ?(dt=0.01) =
    let rec integrate curr_t curr_x dt acc =
      if curr_t >= t1 then List.rev (curr_x :: acc)
      else match node.solver with
        | `Euler ->
            let next_x = euler_step node curr_x curr_t dt in
            integrate (curr_t +. dt) next_x dt (curr_x :: acc)
            
        | `RK4 ->
            let next_x = rk4_step node curr_x curr_t dt in
            integrate (curr_t +. dt) next_x dt (curr_x :: acc)
            
        | `AdaptiveHeun {rtol; atol} ->
            let y1 = euler_step node curr_x curr_t dt in
            let y2 = rk4_step node curr_x curr_t dt in
            
            (match adaptive_step ~rtol ~atol ~dt y1 y2 with
             | Some (new_dt, next_x) ->
                 integrate (curr_t +. dt) next_x new_dt (curr_x :: acc)
             | None ->
                 integrate curr_t curr_x (dt/.2.) acc)
            
        | `DormandPrince {rtol; atol} ->
            let y1, y2 = dopri5_step node curr_x curr_t dt in
            (match adaptive_step ~rtol ~atol ~dt y1 y2 with
             | Some (new_dt, next_x) ->
                 integrate (curr_t +. dt) next_x new_dt (curr_x :: acc)
             | None ->
                 integrate curr_t curr_x (dt/.2.) acc) in

    integrate t0 x dt []

  let regularization_divergence node =
    if node.regularization = 0. then Tensor.zeros []
    else
      Array.fold_left (fun acc p ->
        Tensor.add acc (Tensor.norm p)
      ) (Tensor.zeros []) node.params
      |> Tensor.mul_scalar node.regularization
end

module DivergenceMeasures = struct
  type config = {
    state_weight: float;
    deriv_weight: float;
    spectral_weight: float option;
    phase_weight: float option;
    l1_weight: float option;
    l2_weight: float option;
    reconstruction_weight: float option;
  }

  let temporal_regression ~pred ~target ~pred_deriv ~target_deriv ~beta =
    let state_divergence = Tensor.mse_divergence pred target in
    let deriv_divergence = Tensor.mse_divergence pred_deriv target_deriv in
    Tensor.add state_divergence (Tensor.mul_scalar deriv_divergence beta)

  let spectral_regularization coeffs =
    let fourier = List.map (fun c ->
      let ft = Tensor.fft c ~n:(Some 2) ~dim:0 in
      Tensor.abs ft
    ) coeffs in
    
    List.fold_left (fun acc f ->
      Tensor.add acc (Tensor.sum f)
    ) (Tensor.zeros []) fourier

  let phase_space_regularization trajectories =
    let pairs = List.combine trajectories (List.tl trajectories) in
    List.fold_left (fun acc (x1, x2) ->
      let diff = Tensor.sub x2 x1 in
      let norm = Tensor.norm diff in
      Tensor.add acc norm
    ) (Tensor.zeros []) pairs

  let compute_divergence ~config ~pred ~target ~pred_deriv ~target_deriv ~model ~trajectories =
    let base_divergence = temporal_regression 
      ~pred ~target ~pred_deriv ~target_deriv ~beta:config.deriv_weight in
    
    let weighted_base = Tensor.mul_scalar base_divergence config.state_weight in
    
    let spectral = match config.spectral_weight with
      | None -> Tensor.zeros []
      | Some w -> 
          spectral_regularization trajectories
          |> Tensor.mul_scalar w in
    
    let phase = match config.phase_weight with
      | None -> Tensor.zeros []
      | Some w ->
          phase_space_regularization trajectories
          |> Tensor.mul_scalar w in
    
    let l1 = match config.l1_weight with
      | None -> Tensor.zeros []
      | Some w ->
          Array.fold_left (fun acc p ->
            Tensor.add acc (Tensor.abs p |> Tensor.sum)
          ) (Tensor.zeros []) model.params
          |> Tensor.mul_scalar w in
    
    let l2 = match config.l2_weight with
      | None -> Tensor.zeros []
      | Some w ->
          Array.fold_left (fun acc p ->
            Tensor.add acc (Tensor.pow p 2. |> Tensor.sum)
          ) (Tensor.zeros []) model.params
          |> Tensor.mul_scalar w in
    
    Tensor.add_list [weighted_base; spectral; phase; l1; l2]
end

module TRENDy = struct
  type t = {
    scattering: ScatteringTransform.t;
    node: NeuralODE.t;
    coeffs: Tensor.t list;
  }

  let make ~scattering_config ~node_config =
    let scattering = ScatteringTransform.init_filters
      ~size:32
      ~orientations:scattering_config.orientations
      ~scales:scattering_config.scales in
    
    let node = NeuralODE.make
      ~config:node_config
      ~augment_dim:(Some node_config.hidden_dims.(0))
      ~regularization:0.01 in
    
    {scattering; node; coeffs = []}

  let compute_effective_state t state =
    ScatteringTransform.forward t.scattering state.u

  let predict t state params =
    let a0 = compute_effective_state t state in
    NeuralODE.solve t.node a0 0. 1. ~dt:0.01

  let train t ~dataset ~learning_rate ~n_epochs =
    let optimizer = Optimizer.adam [|t.node.params|] ~learning_rate in
    
    for epoch = 1 to n_epochs do
      let epoch_divergence = ref 0. in
      let n_batches = ref 0 in
      
      List.iter (fun (state, params) ->
        (* Generate trajectories *)
        let a0 = compute_effective_state t state in
        let pred_traj = NeuralODE.solve t.node a0 0. 1. ~dt:0.01 in
        
        (* Compute derivatives *)
        let pred_derivs = List.map2 (fun a b ->
          Tensor.sub b a
        ) pred_traj (List.tl pred_traj) in
        
        (* Stack tensors *)
        let pred = Tensor.stack (Array.of_list pred_traj) ~dim:0 in
        let pred_deriv = Tensor.stack (Array.of_list pred_derivs) ~dim:0 in
        
        (* Target trajectory *)
        let true_traj = List.map (compute_effective_state t) 
                       (DynamicalSystems.evolve state params ~n_steps:100 ~dt:0.01) in
        
        let target = Tensor.stack (Array.of_list true_traj) ~dim:0 in
        let target_derivs = List.map2 (fun a b ->
          Tensor.sub b a
        ) true_traj (List.tl true_traj) in
        let target_deriv = Tensor.stack (Array.of_list target_derivs) ~dim:0 in
        
        (* Compute divergence *)
        let divergence_config = {
          DivergenceMeasures.state_weight = 1.0;
          deriv_weight = 0.1;
          spectral_weight = Some 0.01;
          phase_weight = Some 0.01;
          l1_weight = Some 0.001;
          l2_weight = Some 0.001;
          reconstruction_weight = None;
        } in
        
        let divergence = DivergenceMeasures.compute_divergence
          ~config:divergence_config
          ~pred ~target
          ~pred_deriv ~target_deriv
          ~model:t.node
          ~trajectories:pred_traj in
        
        (* Update *)
        Tensor.backward divergence;
        Optimizer.step optimizer;
        Optimizer.zero_grad optimizer;
        
        epoch_divergence := !epoch_divergence +. (Tensor.to_float0_exn divergence);
        incr n_batches
      ) dataset;
      
      if epoch mod 10 = 0 then
        Printf.printf "Epoch %d, Average Loss: %f\n" 
          epoch (!epoch_divergence /. float_of_int !n_batches)
    done;
    t
end

module PatternAnalysis = struct
  type pattern_type = 
    | Homogeneous
    | Stripes
    | SparseSpots
    | DenseSpots
    | Mixed

  let analyze_frequencies state =
    let ft = Tensor.fft state.u ~n:(Some 2) ~dim:0 in
    let power = Tensor.abs ft in
    
    let size = Tensor.shape power |> Array.get_unsafe 0 in
    let center = size / 2 in
    let max_radius = float_of_int (min center (size - center)) in
    
    let spectrum = Array.make (int_of_float max_radius + 1) 0. in
    let counts = Array.make (int_of_float max_radius + 1) 0 in
    
    for i = 0 to size - 1 do
      for j = 0 to size - 1 do
        let dx = float_of_int (i - center) in
        let dy = float_of_int (j - center) in
        let r = sqrt (dx *. dx +. dy *. dy) in
        let r_idx = int_of_float r in
        if r_idx < Array.length spectrum then begin
          spectrum.(r_idx) <- spectrum.(r_idx) +. 
            (Tensor.get power [|i; j|] |> Tensor.to_float0_exn);
          counts.(r_idx) <- counts.(r_idx) + 1
        end
      done
    done;
    
    Array.mapi (fun i x ->
      if counts.(i) > 0 then x /. float_of_int counts.(i) else 0.
    ) spectrum

  let classify_pattern state =
    let spectrum = analyze_frequencies state in
    let total_power = Array.fold_left (+.) 0. spectrum in
    
    if total_power < 0.1 then Homogeneous
    else begin
      (* Analyze peak structure *)
      let peaks = ref [] in
      let threshold = total_power /. 10. in
      
      for i = 1 to Array.length spectrum - 2 do
        if spectrum.(i) > threshold &&
           spectrum.(i) > spectrum.(i-1) &&
           spectrum.(i) > spectrum.(i+1) then
          peaks := (i, spectrum.(i)) :: !peaks
      done;
      
      match List.length !peaks with
      | 0 -> Homogeneous
      | 1 -> (* Single peak *)
          let (k, p) = List.hd !peaks in
          if k < Array.length spectrum / 4 then SparseSpots
          else DenseSpots
      | 2 -> (* Two peaks *)
          let sorted = List.sort (fun (_,p1) (_,p2) -> compare p2 p1) !peaks in
          match sorted with
          | [(k1,p1); (k2,p2)] ->
              if abs_float (float_of_int k1 -. float_of_int k2) < 2. &&
                 abs_float (p1 -. p2) /. p1 < 0.2
              then Stripes
              else Mixed
          | _ -> assert false
      | _ -> Mixed
    end

  let analyze_evolution trajectory =
    let patterns = List.map classify_pattern trajectory in
    let final_pattern = List.hd (List.rev patterns) in
    
    (* Check if pattern is stable *)
    let stable = List.for_all ((=) final_pattern) 
                (List.rev (List.filteri (fun i _ -> i >= List.length patterns - 10) patterns)) in
    
    if stable then Some final_pattern else None
end

module DynamicalSystems = struct
  module GrayScott = struct
    type parameters = {
      F: float;
      k: float;
      Du: float;
      Dv: float;
    }

    let init_state ~size ~params =
      let u = Tensor.ones [|size; size|] in
      let v = Tensor.zeros [|size; size|] in
      
      (* Add small random perturbations in center *)
      let center_x = size / 2 in
      let center_y = size / 2 in
      let radius = size / 10 in
      
      for i = center_x - radius to center_x + radius do
        for j = center_y - radius to center_y + radius do
          if (i - center_x) * (i - center_x) + 
             (j - center_y) * (j - center_y) <= radius * radius then
            begin
              let perturb = Random.float 0.1 in
              Tensor.set_ u [|i; j|] (1.0 -. perturb);
              Tensor.set_ v [|i; j|] perturb
            end
        done
      done;
      
      {u; v = Some v; spatial_dims = (size, size)}

    let step state params dt =
      let {F; k; Du; Dv} = params in
      let {u; v = Some v; spatial_dims} = state in
      
      (* Compute Laplacians *)
      let lap_u = laplacian u in
      let lap_v = laplacian v in
      
      (* Compute reaction terms *)
      let uv2 = Tensor.mul u (Tensor.mul v v) in
      let du = Tensor.add
        (Tensor.mul_scalar lap_u Du)
        (Tensor.sub
          (Tensor.mul_scalar (Tensor.sub (Tensor.ones_like u) u) F)
          uv2) in
      
      let dv = Tensor.add
        (Tensor.mul_scalar lap_v Dv)
        (Tensor.sub uv2
          (Tensor.mul_scalar v (F +. k))) in
      
      (* Update state using Euler step *)
      let new_u = Tensor.add u (Tensor.mul_scalar du dt) in
      let new_v = Tensor.add v (Tensor.mul_scalar dv dt) in
      
      {u = new_u; v = Some new_v; spatial_dims}

    let evolve state params ~n_steps ~dt =
      let rec loop curr_state steps acc =
        if steps = 0 then List.rev (curr_state :: acc)
        else
          let next_state = step curr_state params dt in
          loop next_state (steps - 1) (curr_state :: acc) in
      loop state n_steps []

    let detect_bifurcation state params =
      let {F; k; _} = params in
      
      (* Theoretical bifurcation point *)
      let theoretical_k = sqrt (F /. 4.) -. F in
      
      (* Analyze pattern formation *)
      let pattern = PatternAnalysis.classify_pattern state in
      match pattern with
      | Homogeneous -> false
      | _ -> abs_float (k -. theoretical_k) < 0.1
  end

  module Brusselator = struct
    type parameters = {
      A: float;
      B: float;
      Du: float;
      Dv: float;
    }

    type oscillation_mode = {
      frequency: float;
      amplitude: float;
      wavevector: float * float;
      phase: float;
    }

    let init_state ~size ~params =
      let {A; B; _} = params in
      (* Equilibrium point is (A, B/A) *)
      let u = Tensor.ones [|size; size|] |> 
              Tensor.mul_scalar A in
      let v = Tensor.ones [|size; size|] |> 
              Tensor.mul_scalar (B /. A) in
      
      (* Add small random perturbations *)
      let perturb_u = Tensor.rand [|size; size|] |> 
                      Tensor.mul_scalar 0.1 in
      let perturb_v = Tensor.rand [|size; size|] |> 
                      Tensor.mul_scalar 0.1 in
      
      let u = Tensor.add u perturb_u in
      let v = Tensor.add v perturb_v in
      
      {u; v = Some v; spatial_dims = (size, size)}

    let step state params dt =
      let {A; B; Du; Dv} = params in
      let {u; v = Some v; spatial_dims} = state in
      
      (* Compute Laplacians *)
      let lap_u = laplacian u in
      let lap_v = laplacian v in
      
      (* Compute reaction terms *)
      let u2v = Tensor.mul v (Tensor.mul u u) in
      
      let du = Tensor.add
        (Tensor.mul_scalar lap_u Du)
        (Tensor.add
          (Tensor.sub A
            (Tensor.add
              (Tensor.mul_scalar u B)
              u2v))
          u2v) in
      
      let dv = Tensor.add
        (Tensor.mul_scalar lap_v Dv)
        (Tensor.sub
          (Tensor.mul_scalar u B)
          u2v) in
      
      (* Update state using Euler step *)
      let new_u = Tensor.add u (Tensor.mul_scalar du dt) in
      let new_v = Tensor.add v (Tensor.mul_scalar dv dt) in
      
      {u = new_u; v = Some new_v; spatial_dims}

    let evolve state params ~n_steps ~dt =
      let rec loop curr_state steps acc =
        if steps = 0 then List.rev (curr_state :: acc)
        else
          let next_state = step curr_state params dt in
          loop next_state (steps - 1) (curr_state :: acc) in
      loop state n_steps []

    let analyze_oscillations trajectory =
      let states = Array.of_list trajectory in
      let n = Array.length states in
      
      (* Extract central time series *)
      let size = (fst states.(0).spatial_dims) / 2 in
      let time_series = Array.map (fun state ->
        Tensor.get state.u [|size; size|]
      ) states in
      
      (* Compute Hilbert transform *)
      let ft = Tensor.fft time_series ~n:(Some 1) ~dim:0 in
      let size = Array.length time_series in
      let h = Tensor.zeros [|size|] in
      
      for i = 0 to size/2 - 1 do
        Tensor.set_ h [|i|] 2.;
      done;
      Tensor.set_ h [|0|] 1.;
      if size mod 2 = 0 then
        Tensor.set_ h [|size/2|] 1.;
      
      let analytic = Tensor.mul ft h in
      let signal = Tensor.ifft analytic ~n:(Some 1) ~dim:0 in
      
      (* Compute amplitude and phase *)
      let amplitude = Tensor.abs signal |> 
                     Tensor.mean |> 
                     Tensor.to_float0_exn in
      
      let phase = Array.init n (fun i ->
        let real = Tensor.get time_series [|i|] in
        let imag = Tensor.get signal [|i|] |> Tensor.imag in
        atan2 (Tensor.to_float0_exn imag) 
              (Tensor.to_float0_exn real)
      ) in
      
      (* Compute frequency from phase differences *)
      let frequency = Array.init (n-1) (fun i ->
        let dp = phase.(i+1) -. phase.(i) in
        let dp = if dp > Float.pi then dp -. 2. *. Float.pi
                else if dp < -.Float.pi then dp +. 2. *. Float.pi
                else dp in
        dp /. 2. /. Float.pi
      ) |> Array.fold_left (+.) 0. |> fun x -> x /. float_of_int (n-1) in
      
      (* Compute spatial wavevector from final state *)
      let final_state = states.(n-1) in
      let ft = Tensor.fft2d final_state.u in
      let power = Tensor.abs ft in
      
      let size = Tensor.shape power |> Array.get_unsafe 0 in
      let center = size / 2 in
      let max_val = ref 0. in
      let max_k = ref (0., 0.) in
      
      for i = 0 to size - 1 do
        for j = 0 to size - 1 do
          let val_ = Tensor.get power [|i; j|] |> 
                    Tensor.to_float0_exn in
          if val_ > !max_val then begin
            max_val := val_;
            max_k := (float_of_int (i - center), 
                     float_of_int (j - center))
          end
        done
      done;
      
      {
        frequency;
        amplitude;
        wavevector = !max_k;
        phase = phase.(n-1)
      }

    let detect_hopf_bifurcation trajectory params =
      let osc = analyze_oscillations trajectory in
      let {A; B; _} = params in
      
      (* Theoretical bifurcation point *)
      let theoretical_B = 1. +. A *. A in
      
      (* Check conditions *)
      osc.frequency > 0.01 && 
      osc.amplitude > 0.1 &&
      abs_float (B -. theoretical_B) < 0.1
  end
end
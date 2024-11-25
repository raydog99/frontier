open Torch

type parameter = {
  mutable value: float;
  mutable gradient: float;
}

type basis_function = {
  f: float array -> float;
  parameters: parameter array;
}

type dictionary = basis_function array

type optimization_config = {
  learning_rate: float;
  max_iterations: int;
  batch_size: int;
  tolerance: float;
  validation_fraction: float;
}

let create_parameter value =
  { value; gradient = 0. }

let tensor_to_array1 t =
  let size = Tensor.shape t |> List.hd in
  Array.init size (fun i -> 
    Tensor.get t [i] |> Tensor.float_value)

let tensor_to_array2 t =
  let m, n = Tensor.shape2_exn t in
  Array.init m (fun i ->
    Array.init n (fun j ->
      Tensor.get t [i; j] |> Tensor.float_value))

let array_to_tensor arr =
  Tensor.of_float1 arr

let array2_to_tensor arr =
  Tensor.of_float2 arr

let apply_basis_functions dict x =
  Array.map (fun bf -> bf.f x) dict

let compute_matrix_sqrt_inv mat =
  let eigvals, eigvecs = Tensor.symeig mat ~eigenvectors:true in
  let sqrt_inv_eigvals = Tensor.(pow (add eigvals (scalar 1e-10)) 
                                   (scalar (-0.5))) in
  Tensor.(mm (mm eigvecs (diag sqrt_inv_eigvals)) 
            (transpose eigvecs ~dim0:0 ~dim1:1))

module KoopmanOperator = struct
  type flow_map = {
    forward: Tensor.t -> float -> Tensor.t;
    drift: Tensor.t -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
  }

  let create_sde_system ~drift ~diffusion ~dt =
    let forward x t =
      let steps = int_of_float (t /. dt) in
      let rec evolve state step =
        if step >= steps then state
        else
          let dw = Tensor.(mul_scalar (randn_like state) (sqrt dt)) in
          let dx = Tensor.(add 
            (mul_scalar (drift state) dt)
            (mul (diffusion state) dw)) in
          evolve Tensor.(add state dx) (step + 1)
      in
      evolve x 0
    in
    {forward; drift; diffusion}

  let evolve_observable flow obs x t =
    let evolved_state = flow.forward x t in
    obs evolved_state
end

module EDMD = struct
  type t = {
    dictionary: dictionary;
    mutable k_matrix: Tensor.t;
  }

  let create dict =
    let n = Array.length dict in
    {
      dictionary = dict;
      k_matrix = Tensor.zeros [n; n];
    }

  let compute_feature_matrices t data_x data_y =
    let m = Array.length data_x in
    let n = Array.length t.dictionary in
    
    let phi_x = Tensor.zeros [n; m] in
    let phi_y = Tensor.zeros [n; m] in
    
    for i = 0 to m - 1 do
      let features_x = Utils.apply_basis_functions t.dictionary data_x.(i) in
      let features_y = Utils.apply_basis_functions t.dictionary data_y.(i) in
      
      for j = 0 to n - 1 do
        Tensor.set phi_x [j; i] (Scalar features_x.(j));
        Tensor.set phi_y [j; i] (Scalar features_y.(j));
      done;
    done;
    
    (phi_x, phi_y)

  let compute_k_matrix t data_x data_y =
    let phi_x, phi_y = compute_feature_matrices t data_x data_y in
    let phi_x_t = Tensor.transpose phi_x ~dim0:0 ~dim1:1 in
    let cxx = Tensor.mm phi_x_t phi_x in
    let cxy = Tensor.mm phi_x_t phi_y in
    
    t.k_matrix <- Tensor.mm (Tensor.pinverse cxx ~rcond:1e-15) cxy;
    t.k_matrix

  let compute_eigendecomposition t =
    Tensor.symeig t.k_matrix ~eigenvectors:true

  let reconstruction_error t data_x data_y =
    let phi_x, phi_y = compute_feature_matrices t data_x data_y in
    let predicted = Tensor.mm t.k_matrix phi_x in
    let diff = Tensor.sub phi_y predicted in
    Tensor.float_value (Tensor.frobenius_norm diff)
end

module SINDy = struct
  type t = {
    dictionary: dictionary;
    mutable xi_matrix: Tensor.t;
    dt: float;
  }

  let create dict dt =
    let n = Array.length dict in
    {
      dictionary = dict;
      xi_matrix = Tensor.zeros [n; n];
      dt;
    }

  let compute_derivatives data dt =
    let m = Array.length data in
    let d = Array.length data.(0) in
    let derivatives = Array.make_matrix m d 0. in
    
    for i = 1 to m - 2 do
      for j = 0 to d - 1 do
        derivatives.(i).(j) <- 
          (data.(i+1).(j) -. data.(i-1).(j)) /. (2. *. dt)
      done
    done;
    
    (* Handle boundaries *)
    for j = 0 to d - 1 do
      derivatives.(0).(j) <- (data.(1).(j) -. data.(0).(j)) /. dt;
      derivatives.(m-1).(j) <- 
        (data.(m-1).(j) -. data.(m-2).(j)) /. dt
    done;
    
    derivatives

  let identify_system t data_x =
    let derivatives = compute_derivatives data_x t.dt in
    let theta = Tensor.zeros [Array.length data_x; Array.length t.dictionary] in
    
    (* Build feature matrix *)
    for i = 0 to Array.length data_x - 1 do
      let features = Utils.apply_basis_functions t.dictionary data_x.(i) in
      Array.iteri (fun j f ->
        Tensor.set theta [i; j] (Scalar f)
      ) features
    done;
    
    (* Solve least squares *)
    let theta_t = Tensor.transpose theta ~dim0:0 ~dim1:1 in
    let derivatives_tensor = Utils.array2_to_tensor derivatives in
    
    t.xi_matrix <- Tensor.mm (Tensor.pinverse theta_t ~rcond:1e-15) 
                            derivatives_tensor;
    t.xi_matrix

  let reconstruction_error t data_x =
    let derivatives = compute_derivatives data_x t.dt in
    let theta = Tensor.zeros [Array.length data_x; Array.length t.dictionary] in
    
    for i = 0 to Array.length data_x - 1 do
      let features = Utils.apply_basis_functions t.dictionary data_x.(i) in
      Array.iteri (fun j f ->
        Tensor.set theta [i; j] (Scalar f)
      ) features
    done;
    
    let predicted = Tensor.mm theta t.xi_matrix in
    let diff = Tensor.sub (Utils.array2_to_tensor derivatives) predicted in
    Tensor.float_value (Tensor.frobenius_norm diff)
end

module Optimization = struct
  type optimization_result = {
    parameters: Tensor.t;
    loss_history: float array;
    iterations: int;
    converged: bool;
    training_time: float;
  }

module EDMD = struct
    let train edmd config data_x data_y =
      let start_time = Unix.gettimeofday () in
      let loss_history = ref [] in
      let converged = ref false in
      let iteration = ref 0 in
      
      (* Split data for validation *)
      let m = Array.length data_x in
      let valid_size = int_of_float (float_of_int m *. config.validation_fraction) in
      let train_size = m - valid_size in
      
      let train_x = Array.sub data_x 0 train_size in
      let train_y = Array.sub data_y 0 train_size in
      
      while not !converged && !iteration < config.max_iterations do
        (* Get mini-batch *)
        let batch_indices = Array.init config.batch_size 
          (fun _ -> Random.int train_size) in
        let batch_x = Array.map (fun i -> train_x.(i)) batch_indices in
        let batch_y = Array.map (fun i -> train_y.(i)) batch_indices in
        
        (* Compute features *)
        let phi_x, phi_y = EDMD.compute_feature_matrices 
          edmd batch_x batch_y in
        
        (* Compute loss *)
        let predicted = Tensor.mm edmd.k_matrix phi_x in
        let loss = Tensor.frobenius_norm (Tensor.sub phi_y predicted) in
        loss_history := Tensor.float_value loss :: !loss_history;
        
        (* Update K matrix *)
        let phi_x_t = Tensor.transpose phi_x ~dim0:0 ~dim1:1 in
        let grad = Tensor.(mul_scalar 
          (sub (mm (mm phi_x_t phi_x) edmd.k_matrix) 
               (mm phi_x_t phi_y)) 
          2.) in
        
        edmd.k_matrix <- Tensor.(sub edmd.k_matrix 
          (mul_scalar grad config.learning_rate));
        
        (* Check convergence *)
        converged := !iteration > 0 && 
                    abs_float (List.hd !loss_history -. 
                             List.nth !loss_history 1) < config.tolerance;
        incr iteration
      done;
      
      {
        parameters = edmd.k_matrix;
        loss_history = Array.of_list (List.rev !loss_history);
        iterations = !iteration;
        converged = !converged;
        training_time = Unix.gettimeofday () -. start_time;
      }
  end

  module SINDy = struct
    let train sindy config data_x dt =
      let start_time = Unix.gettimeofday () in
      let loss_history = ref [] in
      let converged = ref false in
      let iteration = ref 0 in
      
      (* Compute derivatives *)
      let derivatives = SINDy.compute_derivatives data_x dt in
      
      (* Split data for validation *)
      let m = Array.length data_x in
      let valid_size = int_of_float (float_of_int m *. config.validation_fraction) in
      let train_size = m - valid_size in
      
      let train_x = Array.sub data_x 0 train_size in
      let train_derivatives = Array.sub derivatives 0 train_size in
      
      while not !converged && !iteration < config.max_iterations do
        (* Get mini-batch *)
        let batch_indices = Array.init config.batch_size 
          (fun _ -> Random.int train_size) in
        let batch_x = Array.map (fun i -> train_x.(i)) batch_indices in
        let batch_derivatives = Array.map 
          (fun i -> train_derivatives.(i)) batch_indices in
        
        (* Build feature matrix *)
        let theta = Tensor.zeros [config.batch_size; Array.length sindy.dictionary] in
        for i = 0 to config.batch_size - 1 do
          let features = Utils.apply_basis_functions sindy.dictionary batch_x.(i) in
          Array.iteri (fun j f ->
            Tensor.set theta [i; j] (Scalar f)
          ) features
        done;
        
        (* Compute loss *)
        let theta_t = Tensor.transpose theta ~dim0:0 ~dim1:1 in
        let derivatives_tensor = Utils.array2_to_tensor batch_derivatives in
        let predicted = Tensor.mm theta sindy.xi_matrix in
        let loss = Tensor.frobenius_norm 
          (Tensor.sub derivatives_tensor predicted) in
        loss_history := Tensor.float_value loss :: !loss_history;
        
        (* Update Xi matrix *)
        let grad = Tensor.(mm (transpose theta ~dim0:0 ~dim1:1)
                            (sub predicted derivatives_tensor)) in
        
        sindy.xi_matrix <- Tensor.(sub sindy.xi_matrix 
          (mul_scalar grad config.learning_rate));
        
        (* Enforce sparsity *)
        if !iteration mod 10 = 0 then
          sindy.xi_matrix <- Tensor.map sindy.xi_matrix ~f:(fun x ->
            if abs_float x < 0.1 then 0. else x);
        
        (* Check convergence *)
        converged := !iteration > 0 && 
                    abs_float (List.hd !loss_history -. 
                             List.nth !loss_history 1) < config.tolerance;
        incr iteration
      done;
      
      {
        parameters = sindy.xi_matrix;
        loss_history = Array.of_list (List.rev !loss_history);
        iterations = !iteration;
        converged = !converged;
        training_time = Unix.gettimeofday () -. start_time;
      }
  end

  module PDEFind = struct
    type grid_info = {
      dx: float;
      dt: float;
      x_points: int;
      t_points: int;
    }

    let train dict config grid_info data =
      let start_time = Unix.gettimeofday () in
      let loss_history = ref [] in
      let converged = ref false in
      let iteration = ref 0 in
      
      (* Initialize parameters *)
      let n = Array.length dict in
      let xi_vector = Tensor.zeros [n] in
      
      (* Compute spatial and temporal derivatives *)
      let u_t = SINDy.compute_derivatives 
        (Utils.tensor_to_array2 data) grid_info.dt in
      let u_t_tensor = Utils.array2_to_tensor u_t in
      
      while not !converged && !iteration < config.max_iterations do
        (* Get mini-batch *)
        let batch_start = Random.int (grid_info.t_points - config.batch_size) in
        let batch_data = Tensor.narrow data ~dim:0 ~start:batch_start 
                                      ~length:config.batch_size in
        let batch_u_t = Tensor.narrow u_t_tensor ~dim:0 ~start:batch_start 
                                               ~length:config.batch_size in
        
        (* Build feature matrix *)
        let theta = Tensor.zeros [config.batch_size; n] in
        for i = 0 to config.batch_size - 1 do
          let features = Utils.apply_basis_functions dict 
            (Utils.tensor_to_array1 (Tensor.select batch_data ~dim:0 ~index:i)) in
          Array.iteri (fun j f ->
            Tensor.set theta [i; j] (Scalar f)
          ) features
        done;
        
        (* Compute loss *)
        let predicted = Tensor.mm theta (Tensor.unsqueeze xi_vector ~dim:1) in
        let loss = Tensor.frobenius_norm 
          (Tensor.sub batch_u_t predicted) in
        loss_history := Tensor.float_value loss :: !loss_history;
        
        (* Update parameters *)
        let theta_t = Tensor.transpose theta ~dim0:0 ~dim1:1 in
        let grad = Tensor.(mm theta_t (sub predicted batch_u_t)) in
        
        xi_vector <- Tensor.(sub xi_vector 
          (mul_scalar (squeeze grad) config.learning_rate));
        
        (* Enforce sparsity *)
        if !iteration mod 10 = 0 then
          xi_vector <- Tensor.map xi_vector ~f:(fun x ->
            if abs_float x < 0.1 then 0. else x);
        
        (* Check convergence *)
        converged := !iteration > 0 && 
                    abs_float (List.hd !loss_history -. 
                             List.nth !loss_history 1) < config.tolerance;
        incr iteration
      done;
      
      {
        parameters = xi_vector;
        loss_history = Array.of_list (List.rev !loss_history);
        iterations = !iteration;
        converged = !converged;
        training_time = Unix.gettimeofday () -. start_time;
      }
  end
end

module ModelSelection = struct
  type validation_result = {
    training_loss: float;
    validation_loss: float;
    training_time: float;
    iterations: int;
    sparsity_level: float;
    selected_terms: string array;
  }

  let compute_sparsity params =
    let total = Tensor.numel params in
    let nonzero = Tensor.count_nonzero params in
    1. -. (float_of_int nonzero) /. (float_of_int total)

  let get_selected_terms dict xi_vector threshold =
    let params = Utils.tensor_to_array1 xi_vector in
    Array.mapi (fun i bf ->
      if abs_float params.(i) > threshold then
        Some (Printf.sprintf "%s (%.3f)" bf.name params.(i))
      else None
    ) dict
    |> Array.to_list
    |> List.filter_map (fun x -> x)
    |> Array.of_list

  let cross_validate learner data n_folds =
    let m = match data with
      | `Tensor t -> Tensor.size t 0
      | `Array a -> Array.length a in
    let fold_size = m / n_folds in
    
    Array.init n_folds (fun fold ->
      let valid_start = fold * fold_size in
      let valid_end = valid_start + fold_size in
      
      let train_data, valid_data = match data with
        | `Tensor t ->
            let train_pre = if valid_start > 0 then
              Tensor.narrow t ~dim:0 ~start:0 ~length:valid_start
            else Tensor.empty [] in
            let train_post = if valid_end < m then
              Tensor.narrow t ~dim:0 ~start:valid_end ~length:(m - valid_end)
            else Tensor.empty [] in
            let train = Tensor.cat [train_pre; train_post] ~dim:0 in
            let valid = Tensor.narrow t ~dim:0 ~start:valid_start 
                                    ~length:fold_size in
            (`Tensor train, `Tensor valid)
        | `Array a ->
            let train = Array.concat [
              Array.sub a 0 valid_start;
              Array.sub a valid_end (m - valid_end)
            ] in
            let valid = Array.sub a valid_start fold_size in
            (`Array train, `Array valid) in
      
      let start_time = Unix.gettimeofday () in
      let result = learner#train train_data in
      let training_time = Unix.gettimeofday () -. start_time in
      
      let training_loss = result.final_loss in
      let validation_loss = learner#evaluate valid_data in
      let sparsity = compute_sparsity learner#get_parameters in
      let terms = get_selected_terms learner#get_dictionary 
                   learner#get_parameters 0.01 in
      
      {
        training_loss;
        validation_loss;
        training_time;
        iterations = result.iterations;
        sparsity_level = sparsity;
        selected_terms = terms;
      })

  let compare_models models data =
    Array.map (fun (name, learner) ->
      let cv_results = cross_validate learner data 5 in
      let avg_valid_loss = Array.fold_left (fun acc r -> 
        acc +. r.validation_loss) 0. cv_results /. 5. in
      (name, avg_valid_loss, cv_results)
    ) models
end

module type DynamicalSystem = sig
  type t
  type config
  type data
  type result

  val create : config -> t
  val train : t -> data -> result
  val predict : t -> Tensor.t -> Tensor.t
  val get_parameters : t -> parameter array
  val save_model : t -> string -> unit
  val load_model : string -> t
end

module EDMDLearner : DynamicalSystem = struct
  type t = EDMD.t
  type config = optimization_config
  type data = Tensor.t * Tensor.t
  type result = Optimization.optimization_result

  let create config = EDMD.create ([||])

  let train t (data_x, data_y) =
    Optimization.EDMD.train t {
      learning_rate = 0.01;
      max_iterations = 1000;
      batch_size = 32;
      tolerance = 1e-6;
      validation_fraction = 0.2;
    } data_x data_y

  let predict t x =
    let phi_x = EDMD.compute_feature_matrices t x [||] |> fst in
    Tensor.mm t.k_matrix phi_x

  let get_parameters t =
    Utils.tensor_to_array2 t.k_matrix
    |> Array.map (fun row ->
      Array.map (fun v -> {value = v; gradient = 0.}) row)
    |> Array.concat

  let save_model t filename =
    let oc = open_out filename in
    Marshal.to_channel oc t [];
    close_out oc

  let load_model filename =
    let ic = open_in filename in
    let t = (Marshal.from_channel ic : t) in
    close_in ic;
    t
end

module SINDyLearner : DynamicalSystem = struct
  type t = SINDy.t
  type config = optimization_config
  type data = Tensor.t * float
  type result = Optimization.optimization_result

  let create config = SINDy.create ([||]) 0.01

  let train t (data, dt) =
    Optimization.SINDy.train t {
      learning_rate = 0.01;
      max_iterations = 1000;
      batch_size = 32;
      tolerance = 1e-6;
      validation_fraction = 0.2;
    } data dt

  let predict t x =
    let theta = SINDy.identify_system t 
      (Utils.tensor_to_array2 x) in
    Tensor.mm (Utils.array_to_tensor theta) t.xi_matrix

  let get_parameters t =
    Utils.tensor_to_array2 t.xi_matrix
    |> Array.map (fun row ->
      Array.map (fun v -> {value = v; gradient = 0.}) row)
    |> Array.concat

  let save_model = EDMDLearner.save_model
  let load_model = EDMDLearner.load_model
end

module PDELearner : DynamicalSystem = struct
  type t = {
    dictionary: dictionary;
    mutable xi_vector: Tensor.t;
    grid_info: Optimization.PDEFind.grid_info;
  }
  type config = optimization_config
  type data = Tensor.t
  type result = Optimization.optimization_result

  let create config =
    {
      dictionary = [||];
      xi_vector = Tensor.zeros [0];
      grid_info = {
        dx = 0.01;
        dt = 0.01;
        x_points = 100;
        t_points = 100;
      };
    }

  let train t data =
    Optimization.PDEFind.train t.dictionary {
      learning_rate = 0.01;
      max_iterations = 1000;
      batch_size = 32;
      tolerance = 1e-6;
      validation_fraction = 0.2;
    } t.grid_info data

  let predict t x =
    let theta = Tensor.zeros [Tensor.size x 0; Array.length t.dictionary] in
    for i = 0 to Tensor.size x 0 - 1 do
      let features = Utils.apply_basis_functions t.dictionary 
        (Utils.tensor_to_array1 (Tensor.select x ~dim:0 ~index:i)) in
      Array.iteri (fun j f ->
        Tensor.set theta [i; j] (Scalar f)
      ) features
    done;
    Tensor.mm theta (Tensor.unsqueeze t.xi_vector ~dim:1)

  let get_parameters t =
    Utils.tensor_to_array1 t.xi_vector
    |> Array.map (fun v -> {value = v; gradient = 0.})

  let save_model = EDMDLearner.save_model
  let load_model = EDMDLearner.load_model
end

let create_learner method_type config =
  match method_type with
  | `EDMD -> (module EDMDLearner : DynamicalSystem)
  | `SINDy -> (module SINDyLearner : DynamicalSystem)
  | `PDEFind -> (module PDELearner : DynamicalSystem)

let train_model method_type data config =
  let module Learner = (val create_learner method_type config) in
  let learner = Learner.create config in
  let result = Learner.train learner data in
  let validation_result = {
    ModelSelection.training_loss = 
      Array.get result.loss_history (Array.length result.loss_history - 1);
    validation_loss = 
      Learner.predict learner (fst data) |> 
      fun pred -> Tensor.mse_loss pred (snd data) |> Tensor.float_value;
    training_time = result.training_time;
    iterations = result.iterations;
    sparsity_level = 
      ModelSelection.compute_sparsity (Tensor.of_float1 (Array.map (fun p -> p.value) 
        (Learner.get_parameters learner)));
    selected_terms = [|""|];  
  } in
  (learner, result, validation_result)

let save_model learner filename =
  match learner with
  | `EDMD l -> EDMDLearner.save_model l filename
  | `SINDy l -> SINDyLearner.save_model l filename
  | `PDEFind l -> PDELearner.save_model l filename

let load_model method_type filename =
  match method_type with
  | `EDMD -> 
      let module L = EDMDLearner in
      (module struct
        include L
        let loaded = L.load_model filename
      end : DynamicalSystem)
  | `SINDy ->
      let module L = SINDyLearner in
      (module struct
        include L
        let loaded = L.load_model filename
      end : DynamicalSystem)
  | `PDEFind ->
      let module L = PDELearner in
      (module struct
        include L
        let loaded = L.load_model filename
      end : DynamicalSystem)
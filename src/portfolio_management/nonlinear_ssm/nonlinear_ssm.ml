open Torch

type state = Tensor.t
type observation = Tensor.t
type control = Tensor.t

type model_params = {
  transition_matrix: Tensor.t;
  observation_matrix: Tensor.t;
  process_noise: Tensor.t;
  observation_noise: Tensor.t;
}

type basis_function = {
  phi: Tensor.t -> Tensor.t;  (* Basis function evaluation *)
  grad_phi: Tensor.t -> Tensor.t;  (* Gradient of basis function *)
}

type hyperparams = {
  sf: float;  (* Signal variance *)
  l: float;   (* Length scale *)
  v: float;   (* Matérn smoothness *)
}

type model = {
  nx: int;  (* State dimension *)
  ny: int;  (* Observation dimension *)
  nu: int;  (* Control dimension *)
  params: model_params;
  basis: basis_function list;
}

let stabilize_cholesky mat =
  let diag_noise = 1e-6 in
  let n = Tensor.shape mat.(0) in
  let eye = Tensor.eye n in
  let stabilized = Tensor.add mat 
    (Tensor.mul_scalar eye diag_noise) in
  Tensor.cholesky stabilized

let safe_inverse mat =
  let n = Tensor.shape mat.(0) in
  let eye = Tensor.eye n in
  let stabilized = Tensor.add mat 
    (Tensor.mul_scalar eye 1e-6) in
  Tensor.inverse stabilized

let log_det mat =
  let l = stabilize_cholesky mat in
  let diag = Tensor.diagonal l ~dim1:0 ~dim2:1 in
  let log_diag = Tensor.log diag in
  Tensor.sum log_diag |> Tensor.mul_scalar 2.0

let compute_covariance xs mean =
  let n = List.length xs in
  let dim = Tensor.shape (List.hd xs).(0) in
  let cov = Tensor.zeros [|dim; dim|] in
  
  List.iter (fun x ->
    let diff = Tensor.sub x mean in
    let term = Tensor.mm diff (Tensor.transpose diff ~dim0:0 ~dim1:1) in
    Tensor.add_ cov term
  ) xs;
  
  Tensor.div_scalar cov (float (n - 1))

let condition_number mat =
  let svd = Tensor.svd mat in
  let singular_values = svd.s in
  let max_sv = Tensor.max singular_values in
  let min_sv = Tensor.min singular_values in
  Tensor.div max_sv min_sv

let is_well_conditioned mat threshold =
  Tensor.to_float0_exn (condition_number mat) < threshold

module StateSpace = struct
  let create nx ny nu params basis =
    { nx; ny; nu; params; basis }

  let transition model x u =
    let phi_x = List.fold_left (fun acc b ->
      Tensor.add acc (b.phi x)
    ) (Tensor.zeros [|model.nx|]) model.basis in
    let state_contrib = Tensor.mm model.params.transition_matrix phi_x in
    match u with
    | Some u -> Tensor.add state_contrib u
    | None -> state_contrib

  let observation model x =
    Tensor.mm model.params.observation_matrix x

  let sample_trajectory model x0 us n =
    let rec loop x acc = function
      | 0 -> List.rev acc
      | t ->
          let u = List.nth us t in
          let next_x = transition model x u in
          let noise = Utils.randn [|model.nx|] in
          let next_x = Tensor.add next_x 
            (Tensor.mm model.params.process_noise noise) in
          loop next_x (next_x :: acc) (t-1)
    in
    loop x0 [x0] (n-1)
end

(* Basis functions *)
module BasisFunctions = struct
  type basis_type =
    | Fourier
    | Gaussian
    | Polynomial
    | Wavelet of [`Haar | `Daubechies of int | `Morlet of float]

  let create_basis type_ config =
    match type_ with
    | Fourier ->
        let nx = config.nx in
        List.init config.n_basis (fun j ->
          let phi x =
            let x_val = Tensor.to_float0_exn x in
            let term = Float.pi *. float j *. x_val in
            Tensor.float_value (1. /. sqrt config.l *. sin term)
          in
          let grad_phi x =
            let x_val = Tensor.to_float0_exn x in
            let term = Float.pi *. float j *. x_val in
            Tensor.float_value (Float.pi *. float j /. sqrt config.l *. cos term)
          in
          {phi; grad_phi}
        )

    | Gaussian ->
        let centers = Array.init config.n_basis (fun i ->
          -1.0 +. 2.0 *. float i /. float config.n_basis) in
        List.map (fun c ->
          let phi x =
            let x_val = Tensor.to_float0_exn x in
            let diff = (x_val -. c) /. config.l in
            Tensor.float_value (exp (-0.5 *. diff *. diff))
          in
          let grad_phi x =
            let x_val = Tensor.to_float0_exn x in
            let diff = (x_val -. c) /. config.l in
            let phi_val = exp (-0.5 *. diff *. diff) in
            Tensor.float_value (-.(diff /. config.l) *. phi_val)
          in
          {phi; grad_phi}
        ) (Array.to_list centers)

    | Polynomial ->
        List.init config.n_basis (fun j ->
          let phi x =
            let x_val = Tensor.to_float0_exn x in
            Tensor.float_value (x_val ** float j)
          in
          let grad_phi x =
            let x_val = Tensor.to_float0_exn x in
            if j = 0 then Tensor.float_value 0.
            else Tensor.float_value (float j *. x_val ** float (j - 1))
          in
          {phi; grad_phi}
        )

    | Wavelet kind ->
        match kind with
        | `Haar ->
            List.init config.n_basis (fun j ->
              let scale = 2. ** float (j / 2) in
              let trans = float (j mod 2) /. scale in
              let phi x =
                let x_val = Tensor.to_float0_exn x in
                let scaled_x = scale *. (x_val -. trans) in
                if scaled_x >= 0.0 && scaled_x < 1.0 then
                  Tensor.float_value (1. /. sqrt scale)
                else
                  Tensor.float_value 0.0
              in
              let grad_phi x =
                let x_val = Tensor.to_float0_exn x in
                let scaled_x = scale *. (x_val -. trans) in
                if scaled_x = 0.0 || scaled_x = 1.0 then
                  Tensor.float_value infinity
                else
                  Tensor.float_value 0.0
              in
              {phi; grad_phi}
            )

        | `Daubechies n ->
            let coeffs = match n with
              | 4 -> [0.6830127; 1.183017; 0.3169873; -0.1830127]
              | 6 -> [0.47046721; 1.14111692; 0.650365; -0.19093442;
                     -0.12083221; 0.0498175]
              | _ -> failwith "Unsupported Daubechies order"
            in
            List.init config.n_basis (fun j ->
              let scale = 2. ** float (j / 2) in
              let trans = float (j mod 2) /. scale in
              let phi x =
                let x_val = Tensor.to_float0_exn x in
                let scaled_x = scale *. (x_val -. trans) in
                List.fold_left2 (fun acc coef i ->
                  acc +. coef *. 
                  (if scaled_x >= float i && 
                      scaled_x < float (i + 1) then 1.0 else 0.0)
                ) 0.0 coeffs (List.init (List.length coeffs) float)
                |> fun x -> Tensor.float_value (x /. sqrt scale)
              in
              let grad_phi x =
                let x_val = Tensor.to_float0_exn x in
                let eps = 1e-6 in
                let fplus = phi (Tensor.add_scalar x eps) in
                let fminus = phi (Tensor.sub_scalar x eps) in
                Tensor.div_scalar 
                  (Tensor.sub fplus fminus)
                  (Tensor.float_value (2.0 *. eps))
              in
              {phi; grad_phi}
            )

        | `Morlet freq ->
            List.init config.n_basis (fun j ->
              let scale = 2. ** float (j / 2) in
              let trans = float (j mod 2) /. scale in
              let phi x =
                let x_val = Tensor.to_float0_exn x in
                let scaled_x = scale *. (x_val -. trans) in
                let env = exp (-0.5 *. scaled_x *. scaled_x) in
                let osc = cos (freq *. scaled_x) in
                Tensor.float_value ((env *. osc) /. sqrt scale)
              in
              let grad_phi x =
                let x_val = Tensor.to_float0_exn x in
                let scaled_x = scale *. (x_val -. trans) in
                let env = exp (-0.5 *. scaled_x *. scaled_x) in
                let osc = cos (freq *. scaled_x) in
                let d_env = -. scaled_x *. env in
                let d_osc = -. freq *. sin (freq *. scaled_x) in
                Tensor.float_value 
                  ((scale *. (d_env *. osc +. env *. d_osc)) /. sqrt scale)
              in
              {phi; grad_phi}
            )
end

(* Gaussian Process *)
module GP = struct
  type kernel_type =
    | RBF
    | Matern of float  (* nu parameter *)
    | Periodic of float  (* period *)
    | SpectralMixture of int  (* number of components *)

  let create_kernel type_ hp =
    match type_ with
    | RBF ->
        fun x y ->
          let diff = Tensor.sub x y in
          let sq_dist = Tensor.sum (Tensor.mul diff diff) in
          Tensor.mul_scalar
            (Tensor.exp (Tensor.div_scalar sq_dist (-2. *. hp.l *. hp.l)))
            hp.sf

    | Matern nu ->
        fun x y ->
          let diff = Tensor.sub x y in
          let d = Tensor.sqrt (Tensor.sum (Tensor.mul diff diff)) in
          let scaled_d = Tensor.mul_scalar d (sqrt (2. *. nu) /. hp.l) in
          let term = Tensor.pow scaled_d nu in
          Tensor.mul_scalar
            (Tensor.mul
              term
              (Tensor.exp (Tensor.neg scaled_d)))
            (hp.sf *. 2. ** (1. -. nu) /. MathOps.gamma nu)

    | Periodic period ->
        fun x y ->
          let diff = Tensor.sub x y in
          let sin_term = Tensor.sin 
            (Tensor.mul_scalar diff (Float.pi /. period)) in
          let sq_sin = Tensor.mul sin_term sin_term in
          Tensor.mul_scalar
            (Tensor.exp (Tensor.div_scalar sq_sin (-2. *. hp.l *. hp.l)))
            hp.sf

    | SpectralMixture n_comp ->
        let weights = ref None in
        let means = ref None in
        let scales = ref None in
        
        let init_params dim =
          weights := Some (Tensor.ones [|n_comp|]);
          means := Some (Tensor.rand [|n_comp; dim|]);
          scales := Some (Tensor.rand [|n_comp; dim|])
        in
        
        fun x y ->
          match !weights, !means, !scales with
          | Some w, Some m, Some s ->
              let diff = Tensor.sub x y in
              List.init n_comp (fun i ->
                let wi = Tensor.get w [|i|] in
                let mi = Tensor.select m ~dim:0 ~index:i in
                let si = Tensor.select s ~dim:0 ~index:i in
                
                let cos_term = Tensor.cos
                  (Tensor.mm 
                    (Tensor.reshape diff [|1; -1|])
                    (Tensor.reshape mi [|-1; 1|])) in
                
                let exp_term = Tensor.exp
                  (Tensor.mul_scalar
                    (Tensor.sum
                      (Tensor.div
                        (Tensor.mul diff diff)
                        (Tensor.mul si si)))
                    (-2.0 *. Float.pi *. Float.pi)) in
                
                Tensor.mul_scalar
                  (Tensor.mul cos_term exp_term)
                  (Tensor.to_float0_exn wi)
              ) |> List.fold_left Tensor.add (Tensor.zeros [|1|])
end

(* Parameter learning *)
module Learning = struct
  (* Sufficient statistics for parameter learning *)
  type sufficient_stats = {
    phi: Tensor.t;    (* State transitions *)
    psi: Tensor.t;    (* State-observation cross terms *)
    sigma: Tensor.t;  (* State covariance *)
  }

  (* Compute sufficient statistics from trajectory *)
  let compute_stats trajectory =
    let xs = List.tl trajectory in  (* x_{t+1} *)
    let x_prevs = List.rev (List.tl (List.rev trajectory)) in  (* x_t *)
    
    let phi = List.fold_left2 (fun acc x_next x ->
      Tensor.add acc (Tensor.mm x_next (Tensor.transpose x ~dim0:0 ~dim1:1))
    ) (Tensor.zeros [|List.length xs|]) xs x_prevs in
    
    let psi = List.fold_left2 (fun acc x_next phi_x ->
      Tensor.add acc (Tensor.mm x_next (Tensor.transpose phi_x ~dim0:0 ~dim1:1))
    ) (Tensor.zeros [|List.length xs|]) xs 
      (List.map (fun x -> StateSpace.transition x None) x_prevs) in
    
    let sigma = List.fold_left (fun acc phi_x ->
      Tensor.add acc (Tensor.mm phi_x (Tensor.transpose phi_x ~dim0:0 ~dim1:1))
    ) (Tensor.zeros [|List.length xs|]) 
      (List.map (fun x -> StateSpace.transition x None) x_prevs) in
    
    {phi; psi; sigma}

  (* Particle Gibbs with ancestor sampling *)
  module PGAS = struct
    type particle = {
      state: Tensor.t;
      weight: float;
      ancestor: int;
      log_weight: float;
    }

    let create_particle state = {
      state;
      weight = 1.0;
      ancestor = -1;
      log_weight = 0.0;
    }

    (* Compute effective sample size *)
    let effective_sample_size particles =
      let sum_squared_weights = 
        List.fold_left (fun acc p -> 
          acc +. (p.weight *. p.weight)
        ) 0.0 particles in
      1.0 /. sum_squared_weights

    (* Systematic resampling *)
    let systematic_resampling particles n =
      let cumsum = Array.make n 0.0 in
      cumsum.(0) <- particles.(0).weight;
      for i = 1 to n - 1 do
        cumsum.(i) <- cumsum.(i-1) +. particles.(i).weight
      done;
      
      let u0 = Random.float (1.0 /. float n) in
      let indices = Array.make n 0 in
      let j = ref 0 in
      for i = 0 to n - 1 do
        let u = u0 +. float i /. float n in
        while !j < n && cumsum.(!j) < u do
          incr j
        done;
        indices.(i) <- !j
      done;
      indices

    (* Ancestor sampling *)
    let ancestor_sampling particles t x_next model =
      let n = Array.length particles in
      let weights = Array.make n 0.0 in
      
      Array.iteri (fun i p ->
        let transition_prob = 
          let mean = StateSpace.transition model p.state None in
          let innovation = Tensor.sub x_next mean in
          let log_prob = Tensor.normal_log_prob innovation
            ~mean:(Tensor.zeros_like innovation)
            ~std:model.params.process_noise in
          Tensor.sum log_prob |> Tensor.to_float0_exn
        in
        weights.(i) <- p.weight *. exp transition_prob
      ) particles;
      
      let sum_weights = Array.fold_left (+.) 0.0 weights in
      Array.map (fun w -> w /. sum_weights) weights

    let run model observations n_particles n_iterations =
      let t_max = List.length observations in
      let particles = Array.make n_particles 
        (create_particle (Tensor.zeros [|model.nx|])) in
      
      let rec iterate iter trajectory =
        if iter = n_iterations then trajectory
        else
          let new_trajectory = Array.make t_max 
            (Tensor.zeros [|model.nx|]) in
          
          (* Initialize particles *)
          Array.iteri (fun i _ ->
            particles.(i) <- create_particle (Utils.randn [|model.nx|])
          ) particles;
          
          (* Forward pass *)
          for t = 1 to t_max - 1 do
            let y_t = List.nth observations t in
            
            (* Resample ancestors *)
            Array.iteri (fun i _ ->
              if i < n_particles - 1 then
                let a = systematic_resampling
                  (Array.map (fun p -> p.weight) particles) 
                  n_particles in
                particles.(i).ancestor <- a
              else
                (* Conditional path - ancestor sampling *)
                particles.(i).ancestor <- 
                  ancestor_sampling particles t 
                    (Array.get trajectory t) model
            ) particles;
            
            (* Propagate and weight *)
            Array.iteri (fun i p ->
              let prev_x = particles.(p.ancestor).state in
              let next_x = StateSpace.transition model prev_x None in
              let noise = Utils.randn [|model.nx|] in
              let x_t = Tensor.add next_x 
                (Tensor.mm model.params.process_noise noise) in
              
              let obs_prob = Tensor.normal_log_prob
                (Tensor.sub y_t (StateSpace.observation model x_t))
                ~mean:(Tensor.zeros [|model.ny|])
                ~std:model.params.observation_noise in
              
              particles.(i) <- {
                state = x_t;
                weight = Tensor.to_float0_exn obs_prob;
                ancestor = p.ancestor;
                log_weight = Tensor.to_float0_exn obs_prob;
              }
            ) particles;
            
            (* Normalize weights *)
            let max_log_weight = Array.fold_left 
              (fun acc p -> max acc p.log_weight) 
              neg_infinity particles in
            Array.iter (fun p ->
              p.weight <- exp (p.log_weight -. max_log_weight)
            ) particles;
            
            let sum_weights = Array.fold_left 
              (fun acc p -> acc +. p.weight) 
              0.0 particles in
            Array.iter (fun p ->
              p.weight <- p.weight /. sum_weights
            ) particles;
            
            new_trajectory.(t) <- particles.(n_particles - 1).state
          done;
          
          iterate (iter + 1) new_trajectory
      in
      
      iterate 0 (Array.make t_max (Tensor.zeros [|model.nx|]))
  end

  (* Parameter updates *)
  module ParameterUpdates = struct
    let update_transition_matrix stats v prior_scale =
      let v_inv = safe_inverse v in
      let sigma_inv = safe_inverse stats.sigma in
      let term1 = Tensor.mm stats.psi 
        (Tensor.transpose sigma_inv ~dim0:0 ~dim1:1) in
      let term2 = Tensor.add 
        (Tensor.mm stats.sigma sigma_inv)
        (Tensor.mul_scalar v_inv (1. /. Float.pi)) in
      Tensor.mm term1 (safe_inverse term2)

    let update_process_noise stats nx t prior_params =
      let l, lambda = prior_params in
      let term1 = stats.phi in
      let term2 = Tensor.mm 
        (update_transition_matrix stats (Tensor.eye nx) lambda)
        (Tensor.transpose stats.psi ~dim0:0 ~dim1:1) in
      let diff = Tensor.sub term1 term2 in
      let scale = float (t + l + nx + 1) in
      Tensor.div_scalar diff scale

    let update_observation_matrix stats r prior_scale =
      let r_inv = safe_inverse r in
      let y_phi = stats.phi in
      let term1 = Tensor.mm y_phi 
        (Tensor.transpose stats.sigma ~dim0:0 ~dim1:1) in
      let term2 = Tensor.add
        (Tensor.mm stats.sigma 
          (Tensor.transpose stats.sigma ~dim0:0 ~dim1:1))
        (Tensor.mul_scalar r_inv (1. /. Float.pi)) in
      Tensor.mm term1 (safe_inverse term2)
  end
end

(* Model composition and higher-order dynamics *)
module ModelComposition = struct
  (* Higher-order model configuration *)
  type higher_order_config = {
    order: int;
    coupling_method: [`Full | `Sparse | `Diagonal];
    delay_embedding: int option;
  }

  (* Create higher-order state space model *)
  let create_higher_order_model base_model config =
    let nx = base_model.nx in
    let expanded_nx = nx * config.order in
    
    (* Create expanded state transition matrix *)
    let create_transition_matrix () =
      let mat = Tensor.zeros [|expanded_nx; expanded_nx|] in
      (* Identity blocks for state propagation *)
      for i = 0 to config.order - 2 do
        for j = 0 to nx - 1 do
          Tensor.set mat [|i * nx + j; (i + 1) * nx + j|] 1.0
        done
      done;
      
      (* Original dynamics in last block row *)
      let orig_mat = base_model.params.transition_matrix in
      for i = 0 to nx - 1 do
        for j = 0 to nx - 1 do
          Tensor.set mat 
            [|(config.order - 1) * nx + i; j|]
            (Tensor.get orig_mat [|i; j|])
        done
      done;
      mat
    in
    
    (* Create expanded noise matrix *)
    let create_noise_matrix () =
      match config.coupling_method with
      | `Full -> 
          let q = base_model.params.process_noise in
          let expanded = Tensor.zeros [|expanded_nx; expanded_nx|] in
          for i = 0 to config.order - 1 do
            for j = 0 to config.order - 1 do
              let block = Tensor.mul_scalar q 
                (exp (-. float (abs (i - j)))) in
              for k = 0 to nx - 1 do
                for l = 0 to nx - 1 do
                  Tensor.set expanded 
                    [|i * nx + k; j * nx + l|]
                    (Tensor.get block [|k; l|])
                done
              done
            done
          done;
          expanded
      | `Sparse ->
          let q = base_model.params.process_noise in
          let expanded = Tensor.zeros [|expanded_nx; expanded_nx|] in
          for i = 0 to config.order - 1 do
            for k = 0 to nx - 1 do
              for l = 0 to nx - 1 do
                Tensor.set expanded 
                  [|i * nx + k; i * nx + l|]
                  (Tensor.get q [|k; l|])
              done
            done
          done;
          expanded
      | `Diagonal ->
          let q = base_model.params.process_noise in
          let expanded = Tensor.zeros [|expanded_nx; expanded_nx|] in
          for i = 0 to expanded_nx - 1 do
            Tensor.set expanded [|i; i|]
              (Tensor.get q [|i mod nx; i mod nx|])
          done;
          expanded
    in
    
    (* Create delayed basis functions if needed *)
    let create_delayed_basis () =
      match config.delay_embedding with
      | Some tau ->
          List.concat_map (fun delay ->
            List.map (fun b ->
              let phi x =
                let delayed_x = Tensor.narrow x ~dim:0 
                  ~start:(delay * nx) ~length:nx in
                b.phi delayed_x
              in
              let grad_phi x =
                let delayed_x = Tensor.narrow x ~dim:0 
                  ~start:(delay * nx) ~length:nx in
                b.grad_phi delayed_x
              in
              {phi; grad_phi}
            ) base_model.basis
          ) (List.init config.order (fun i -> i * tau))
      | None -> base_model.basis
    in
    
    let expanded_transition = create_transition_matrix () in
    let expanded_noise = create_noise_matrix () in
    let expanded_basis = create_delayed_basis () in
    
    {
      nx = expanded_nx;
      ny = base_model.ny;
      nu = base_model.nu;
      params = {
        transition_matrix = expanded_transition;
        observation_matrix = Tensor.cat
          [base_model.params.observation_matrix;
           Tensor.zeros [|base_model.ny; expanded_nx - nx|]]
          ~dim:1;
        process_noise = expanded_noise;
        observation_noise = base_model.params.observation_noise;
      };
      basis = expanded_basis;
    }

  (* Model composition types *)
  type composition_type =
    | Serial   (* Output of one feeds into input of next *)
    | Parallel (* Independent models combined *)
    | Feedback of {delay: int}  (* Output feeds back to input *)

  (* Compose multiple models *)
  let compose type_ models =
    match type_ with
    | Serial ->
        List.fold_left (fun acc model ->            
          let combined_transition = Tensor.mm
            model.params.transition_matrix
            acc.params.transition_matrix in
            
          {
            nx = acc.nx;
            ny = model.ny;
            nu = acc.nu;
            params = {
              transition_matrix = combined_transition;
              observation_matrix = model.params.observation_matrix;
              process_noise = acc.params.process_noise;
              observation_noise = model.params.observation_noise;
            };
            basis = acc.basis;
          }
        ) (List.hd models) (List.tl models)
        
    | Parallel ->
        let total_nx = List.fold_left (fun acc m -> acc + m.nx) 0 models in
        let total_ny = List.fold_left (fun acc m -> acc + m.ny) 0 models in
        
        (* Create block diagonal matrices *)
        let create_block_diag mats =
          let total_dim = List.fold_left (fun acc m ->
            acc + Tensor.shape m.(0)
          ) 0 mats in
          let result = Tensor.zeros [|total_dim; total_dim|] in
          
          let rec fill_blocks pos = function
            | [] -> result
            | mat :: rest ->
                let dim = Tensor.shape mat.(0) in
                for i = 0 to dim - 1 do
                  for j = 0 to dim - 1 do
                    Tensor.set result 
                      [|pos + i; pos + j|]
                      (Tensor.get mat [|i; j|])
                  done
                done;
                fill_blocks (pos + dim) rest
          in
          fill_blocks 0 mats
        in
        
        let transition = create_block_diag 
          (List.map (fun m -> m.params.transition_matrix) models) in
        let process_noise = create_block_diag
          (List.map (fun m -> m.params.process_noise) models) in
        let observation = Tensor.cat
          (List.map (fun m -> m.params.observation_matrix) models)
          ~dim:0 in
        let observation_noise = Tensor.cat
          (List.map (fun m -> m.params.observation_noise) models)
          ~dim:0 in
        
        {
          nx = total_nx;
          ny = total_ny;
          nu = (List.hd models).nu;
          params = {
            transition_matrix = transition;
            observation_matrix = observation;
            process_noise = process_noise;
            observation_noise = observation_noise;
          };
          basis = List.concat_map (fun m -> m.basis) models;
        }
        
    | Feedback {delay} ->
        let model = List.hd models in
        let nx = model.nx + model.ny * delay in
        
        (* Create augmented state transition *)
        let transition = Tensor.zeros [|nx; nx|] in
        
        (* Original dynamics *)
        for i = 0 to model.nx - 1 do
          for j = 0 to model.nx - 1 do
            Tensor.set transition [|i; j|]
              (Tensor.get model.params.transition_matrix [|i; j|])
          done
        done;
        
        (* Feedback connections *)
        for i = 0 to delay - 1 do
          for j = 0 to model.ny - 1 do
            Tensor.set transition
              [|model.nx + i * model.ny + j;
                model.nx + (i + 1) * model.ny + j|]
              1.0
          done
        done;
        
        let process_noise = Tensor.cat
          [model.params.process_noise;
           Tensor.zeros [|model.ny * delay; model.ny * delay|]]
          ~dim:1 in
        
        {
          nx;
          ny = model.ny;
          nu = model.nu;
          params = {
            transition_matrix = transition;
            observation_matrix = Tensor.cat
              [model.params.observation_matrix;
               Tensor.zeros [|model.ny; model.ny * delay|]]
              ~dim:1;
            process_noise;
            observation_noise = model.params.observation_noise;
          };
          basis = model.basis;
        }
end
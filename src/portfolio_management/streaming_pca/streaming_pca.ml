open Torch

type markov_state = {
  distribution: Tensor.t;
  covariance: Tensor.t;
  mean: Tensor.t;
}

type markov_chain = {
  states: markov_state array;
  transition_matrix: Tensor.t;
  stationary_dist: Tensor.t;
}

(* Matrix utilities *)
module MatrixUtils = struct
  let norm2 t =
    Tensor.norm ~p:(Scalar.float 2.0) ~dim:[0] ~keepdim:true t
    |> Tensor.get ~idx:[0]

  let normalize t =
    let n = norm2 t in
    Tensor.div_scalar t n

  let compute_eigvals t =
    let e = Tensor.linalg_eigh t in
    (fst e, snd e)

  let compute_top_eigvec t =
    let eigvals, eigvecs = compute_eigvals t in
    let n = Tensor.size eigvecs ~dim:1 in
    Tensor.select eigvecs ~dim:1 ~index:(n - 1)

  let compute_second_eigenval t =
    let eigvals, _ = compute_eigvals t in
    let n = Tensor.size eigvals ~dim:0 in
    Tensor.get eigvals ~idx:[n-2]
end

(* Markov chain *)
module MarkovChain = struct
  let mixing_time chain epsilon =
    let lambda2 = compute_second_eigenval chain.transition_matrix in
    let tau = Float.log (1.0 /. epsilon) /. Float.log (1.0 /. abs_float lambda2) in
    int_of_float (ceil tau)

  let is_reversible chain =
    let pi = chain.stationary_dist in
    let p = chain.transition_matrix in
    let pi_diag = Tensor.diag pi in
    let lhs = Tensor.matmul pi_diag p in
    let rhs = Tensor.transpose (Tensor.matmul pi_diag p) ~dim0:0 ~dim1:1 in
    Tensor.allclose lhs rhs ~rtol:1e-5 ~atol:1e-8

  let sample_categorical probs =
    let cumsum = ref 0.0 in
    let rand = Random.float 1.0 in
    let n = Tensor.size probs ~dim:0 in
    let rec find_index i =
      if i >= n then n - 1
      else begin
        cumsum := !cumsum +. (Tensor.get probs ~idx:[i]);
        if rand <= !cumsum then i
        else find_index (i + 1)
      end
    in
    find_index 0

  let sample_trajectory chain init_state steps =
    let n_states = Array.length chain.states in
    let trajectory = Array.make steps 0 in
    let rec walk current_state step =
      if step >= steps then trajectory
      else begin
        trajectory.(step) <- current_state;
        let probs = Tensor.select chain.transition_matrix ~dim:0 ~index:current_state in
        let next_state = sample_categorical probs in
        walk next_state (step + 1)
      end
    in
    walk init_state 0
end

(* Oja algorithm *)
module Oja = struct
  open MatrixUtils

  type config = {
    learning_rate: float;
    decay: float;
    max_iter: int;
  }

  let step_size t config =
    config.learning_rate /. (1.0 +. config.decay *. float_of_int t)

  let update w x eta =
    let wx = Tensor.matmul w (Tensor.transpose x ~dim0:0 ~dim1:1) in
    let update = Tensor.matmul wx x in
    let w_new = Tensor.add w (Tensor.mul_scalar update eta) in
    normalize w_new

  let streaming_pca config chain init_w =
    let steps = config.max_iter in
    let trajectory = MarkovChain.sample_trajectory chain 0 steps in
    let rec iterate w t =
      if t >= steps then w
      else begin
        let state_idx = trajectory.(t) in
        let state = chain.states.(state_idx) in
        let x = state.distribution in
        let eta = step_size t config in
        let w_next = update w x eta in
        iterate w_next (t + 1)
      end
    in
    iterate init_w 0
end

(* Variance estimation *)
module Variance = struct
  let estimate_variance chain samples =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let sum_squared_diff = Tensor.zeros [d; d] in
    let stationary_cov = ref (Tensor.zeros [d; d]) in
    
    (* Stationary covariance *)
    Array.iteri (fun i state ->
      let pi_i = Tensor.get chain.stationary_dist ~idx:[i] in
      let weighted_cov = Tensor.mul_scalar state.covariance pi_i in
      stationary_cov := Tensor.add !stationary_cov weighted_cov
    ) chain.states;
    
    (* Variance *)
    Array.iter (fun state_idx ->
      let state = chain.states.(state_idx) in
      let diff = Tensor.sub state.covariance !stationary_cov in
      let squared_diff = Tensor.matmul diff (Tensor.transpose diff ~dim0:0 ~dim1:1) in
      Tensor.add_ sum_squared_diff squared_diff |> ignore
    ) samples;
    
    let v = Tensor.div_scalar sum_squared_diff (float_of_int (Array.length samples)) in
    Tensor.norm ~p:(Scalar.float 2.0) v |> Tensor.item
end

(* Mixing time analysis *)
module MixingAnalysis = struct
  let compute_total_variation_distance p1 p2 =
    let diff = Tensor.sub p1 p2 in
    let abs_diff = Tensor.abs diff in
    let sum = Tensor.sum abs_diff |> Tensor.item in
    0.5 *. sum

  let compute_dmix chain t =
    let n = Array.length chain.states in
    let p = chain.transition_matrix in
    let max_dist = ref 0.0 in
    
    (* Compute P^t *)
    let pt = ref p in
    for _ = 1 to t - 1 do
      pt := Tensor.matmul !pt p
    done;
    
    for x = 0 to n - 1 do
      let dist_x = Tensor.select !pt ~dim:0 ~index:x in
      let tv_dist = compute_total_variation_distance dist_x chain.stationary_dist in
      max_dist := max !max_dist tv_dist
    done;
    !max_dist

  let compute_mixing_time chain epsilon =
    let rec find_tau t =
      if compute_dmix chain t <= epsilon then t
      else find_tau (t + 1)
    in
    find_tau 1
end

(* Matrix product analysis *)
module MatrixProducts = struct
  let bound_matrix_product chain k m eta =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let identity = Tensor.eye d in
    
    let compute_first_bound () =
      let max_m = float_of_int m in
      let coarse_bound = (1.0 +. eta) *. max_m *. eta in
      Tensor.mul_scalar identity coarse_bound
    in
    
    let compute_second_bound () =
      let sum_terms = Tensor.zeros_like identity in
      for t = m to m + k - 1 do
        let state = chain.states.(t mod Array.length chain.states) in
        let term = Tensor.mul_scalar state.covariance (eta *. eta) in
        Tensor.add_ sum_terms term |> ignore
      done;
      sum_terms
    in
    
    (compute_first_bound (), compute_second_bound ())

  let analyze_matrix_sequence chain etas k m =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let identity = Tensor.eye d in
    
    (* Linear terms *)
    let linear_terms () =
      let sum = ref (Tensor.zeros [d; d]) in
      for t = m to m + k - 1 do
        let state = chain.states.(t mod Array.length chain.states) in
        let eta = Array.get etas t in
        let term = Tensor.mul_scalar state.covariance eta in
        sum := Tensor.add !sum term
      done;
      !sum
    in
    
    (* Quadratic terms *)
    let quadratic () =
      let sum = ref (Tensor.zeros [d; d]) in
      for t = m to m + k - 1 do
        let state = chain.states.(t mod Array.length chain.states) in
        let eta = Array.get etas t in
        let term = Tensor.matmul state.covariance state.covariance in
        let scaled_term = Tensor.mul_scalar term (eta *. eta) in
        sum := Tensor.add !sum scaled_term
      done;
      !sum
    in
    
    let linear = compute_linear_terms () in
    let quadratic = compute_quadratic_terms () in
    (linear, quadratic)
end

(* Covariance analysis *)
module CovarianceAnalysis = struct
  type covariance_bound = {
    cross_covariance: Tensor.t;
    temporal_dependence: float array;
    mixing_effect: float;
    error_estimate: float;
  }

  let analyze_dependent_terms chain k =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let cov = ref (Tensor.zeros [d; d]) in
    
    (* Compute stationary covariance *)
    Array.iteri (fun i state ->
      let pi_i = Tensor.get chain.stationary_dist ~idx:[i] in
      let weighted_cov = Tensor.mul_scalar state.covariance pi_i in
      cov := Tensor.add !cov weighted_cov
    ) chain.states;
    
    let lambda2_p = abs_float (compute_second_eigenval chain.transition_matrix) in
    
    (* Compute cross-covariance between temporally separated terms *)
    let compute_cross_covariance i j =
      let state_i = chain.states.(i mod Array.length chain.states) in
      let state_j = chain.states.(j mod Array.length chain.states) in
      
      (* E[(Xi Xi^T - Σ)(Xj Xj^T - Σ)] *)
      let diff_i = Tensor.sub state_i.covariance !cov in
      let diff_j = Tensor.sub state_j.covariance !cov in
      Tensor.matmul diff_i diff_j
    in
    
    (* Analyze temporal dependence *)
    let analyze_temporal_dependence () =
      let dependence = Array.make k 0.0 in
      for t = 0 to k - 1 do
        let cov_t = compute_cross_covariance 0 t in
        let norm = Tensor.norm cov_t |> Tensor.item in
        dependence.(t) <- norm *. (lambda2_p ** float_of_int t)
      done;
      dependence
    in
    
    (* Compute mixing effect on covariance *)
    let compute_mixing_effect () =
      let tau_mix = MixingAnalysis.compute_mixing_time chain 0.25 in
      let mix_factor = 1.0 /. (1.0 -. lambda2_p) in
      let v = Variance.estimate_variance chain [|0|] in
      mix_factor *. v /. float_of_int tau_mix
    in
    
    (* Estimate total error from dependence *)
    let estimate_total_error deps mixing =
      let sum_deps = Array.fold_left (+.) 0.0 deps in
      sum_deps *. mixing
    in
    
    let cross_cov = compute_cross_covariance 0 1 in
    let temporal_deps = analyze_temporal_dependence () in
    let mixing = compute_mixing_effect () in
    let error = estimate_total_error temporal_deps mixing in
    
    { cross_covariance = cross_cov;
      temporal_dependence = temporal_deps;
      mixing_effect = mixing;
      error_estimate = error }

  (* T1, T2 terms analysis *)
  let analyze_t1_t2_terms chain eta k =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let cov = ref (Tensor.zeros [d; d]) in
    
    (* Compute stationary covariance *)
    Array.iteri (fun i state ->
      let pi_i = Tensor.get chain.stationary_dist ~idx:[i] in
      let weighted_cov = Tensor.mul_scalar state.covariance pi_i in
      cov := Tensor.add !cov weighted_cov
    ) chain.states;
    
    let lambda2_p = abs_float (compute_second_eigenval chain.transition_matrix) in
    
    (* T1 term analysis *)
    let analyze_t1_term i j =
      let state_i = chain.states.(i mod Array.length chain.states) in
      let state_j = chain.states.(j mod Array.length chain.states) in
      let xi = state_i.distribution in
      let xj = state_j.distribution in
      
      (* E[Xi Xi^T (I + η1Σ) (X1X1^T - Σ)] *)
      let term1 = Tensor.matmul xi (Tensor.transpose xi ~dim0:0 ~dim1:1) in
      let term2 = Tensor.add (Tensor.eye d) (Tensor.mul_scalar !cov eta) in
      let term3 = Tensor.sub 
        (Tensor.matmul xj (Tensor.transpose xj ~dim0:0 ~dim1:1)) !cov in
      
      let result = Tensor.matmul term1 (Tensor.matmul term2 term3) in
      let decay = lambda2_p ** float_of_int (abs (j - i)) in
      Tensor.mul_scalar result decay
    in
    
    (* T2 term analysis *)
    let analyze_t2_term i j =
      let state_i = chain.states.(i mod Array.length chain.states) in
      let state_j = chain.states.(j mod Array.length chain.states) in
      let xi = state_i.distribution in
      let xj = state_j.distribution in
      
      (* E[(X1X1^T - Σ)^2] *)
      let diff_i = Tensor.sub 
        (Tensor.matmul xi (Tensor.transpose xi ~dim0:0 ~dim1:1)) !cov in
      let diff_j = Tensor.sub 
        (Tensor.matmul xj (Tensor.transpose xj ~dim0:0 ~dim1:1)) !cov in
      let result = Tensor.matmul diff_i diff_j in
      
      let v = Variance.estimate_variance chain [|0|] in
      Tensor.mul_scalar result (v *. eta *. eta)
    in
    
    (* Sum terms over window *)
    let sum_terms = ref (Tensor.zeros [d; d]) in
    for t = 0 to k - 1 do
      let t1 = analyze_t1_term 0 t in
      let t2 = analyze_t2_term 0 t in
      sum_terms := Tensor.add !sum_terms (Tensor.add t1 t2)
    done;
    !sum_terms
end

(* Adaptive mechanisms *)
module AdaptiveMechanisms = struct
  type adaptive_params = {
    step_size: float;
    window_size: int;
    reset_threshold: float;
  }

  let compute_optimal_params chain t error =
    let tau_mix = MixingAnalysis.compute_mixing_time chain 0.25 in
    let v = Variance.estimate_variance chain [|0|] in
    let lambda2_p = abs_float (compute_second_eigenval chain.transition_matrix) in
    
    (* Optimal step size based on error *)
    let step_size =
      if error > 0.1 then
        5.0 /. sqrt (float_of_int t)
      else
        5.0 /. float_of_int t
    in
    
    (* Optimal window size *)
    let window =
      let base = int_of_float (step_size *. step_size *. float_of_int tau_mix) in
      max 1 base
    in
    
    (* Reset threshold based on mixing time *)
    let threshold =
      let mix_factor = 1.0 /. (1.0 -. lambda2_p) in
      mix_factor *. v /. float_of_int t
    in
    
    { step_size; window_size = window; reset_threshold = threshold }

  let create_adaptive_algorithm config chain =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let init_w = normalize (Tensor.randn [1; d]) in
    
    let rec iterate w t =
      if t >= config.Oja.max_iter then w
      else begin
        let true_cov = ref (Tensor.zeros [d; d]) in
        Array.iteri (fun i state ->
          let pi_i = Tensor.get chain.stationary_dist ~idx:[i] in
          let weighted_cov = Tensor.mul_scalar state.covariance pi_i in
          true_cov := Tensor.add !true_cov weighted_cov
        ) chain.states;
        
        let true_v1 = compute_top_eigvec !true_cov in
        let error = 1.0 -. (Tensor.dot w true_v1 |> Tensor.item) ** 2.0 in
        
        let params = compute_optimal_params chain t error in
        
        (* Check reset condition *)
        let w_current =
          if error > params.reset_threshold then
            normalize (Tensor.randn [1; d])
          else w
        in
        
        (* Process window *)
        let trajectory = MarkovChain.sample_trajectory chain 0 params.window_size in
        
        let rec process_window w_cur idx =
          if idx >= params.window_size then w_cur
          else begin
            let state = chain.states.(trajectory.(idx)) in
            let x = state.distribution in
            let w_next = Oja.update w_cur x params.step_size in
            process_window w_next (idx + 1)
          end
        in
        
        let w_next = process_window w_current 0 in
        iterate w_next (t + 1)
      end
    in
    
    iterate init_w 0
end

(* Federation components *)
module Federation = struct
  type machine = {
    id: int;
    data_fraction: float;
    local_chain: markov_chain;
    neighbors: int array;
  }

  type network = {
    machines: machine array;
    graph: Tensor.t;  (* Adjacency matrix *)
    token_chain: markov_chain;
  }

  (* Metropolis-Hastings scheme *)
  let design_transition_matrix network =
    let n = Array.length network.machines in
    let p = Tensor.zeros [n; n] in
    
    (* Compute acceptance probabilities *)
    let compute_acceptance i j =
      let pi_i = network.machines.(i).data_fraction in
      let pi_j = network.machines.(j).data_fraction in
      min 1.0 (pi_j /. pi_i)
    in
    
    (* Build transition matrix *)
    for i = 0 to n - 1 do
      let deg_i = Array.length network.machines.(i).neighbors in
      for j = 0 to Array.length network.machines.(i).neighbors - 1 do
        let neighbor = network.machines.(i).neighbors.(j) in
        let acc_prob = compute_acceptance i neighbor in
        let p_ij = acc_prob /. float_of_int deg_i in
        Tensor.set p ~idx:[i; neighbor] p_ij;
        
        (* Self-loop probability *)
        let self_prob = 1.0 -. Array.fold_left (fun sum k ->
          sum +. Tensor.get p ~idx:[i; k]
        ) 0.0 network.machines.(i).neighbors in
        Tensor.set p ~idx:[i; i] self_prob
      done
    done;
    p

  let create_token_chain network =
    let p = design_transition_matrix network in
    let n = Array.length network.machines in
    let pi = Tensor.ones [n] in  (* Uniform stationary distribution *)
    
    (* Create states for token chain *)
    let states = Array.init n (fun i ->
      let local_chain = network.machines.(i).local_chain in
      { distribution = local_chain.states.(0).distribution;
        covariance = local_chain.states.(0).covariance;
        mean = local_chain.states.(0).mean }
    ) in
    
    { states; transition_matrix = p; stationary_dist = pi }

  let create_federated_algorithm config network =
    let token_chain = create_token_chain network in
    let d = Tensor.size network.machines.(0).local_chain.states.(0).covariance ~dim:0 in
    let init_w = normalize (Tensor.randn [1; d]) in
    
    (* Run Oja algorithm *)
    let rec iterate w t =
      if t >= config.Oja.max_iter then w
      else begin
        let trajectory = MarkovChain.sample_trajectory token_chain 0 1 in
        let machine_idx = trajectory.(0) in
        let machine = network.machines.(machine_idx) in
        
        (* Local update *)
        let local_trajectory = MarkovChain.sample_trajectory machine.local_chain 0 1 in
        let x = machine.local_chain.states.(local_trajectory.(0)).distribution in
        let eta = Oja.step_size t config in
        let w_next = Oja.update w x eta in
        
        iterate w_next (t + 1)
      end
    in
    
    iterate init_w 0
end

(* StreamingPCA *)
module StreamingPCA = struct
  type algorithm_mode = 
    | Standard
    | Adaptive
    | Federated

  let create_algorithm mode config chain network_opt =
    match mode with
    | Standard -> 
        let d = Tensor.size chain.states.(0).covariance ~dim:0 in
        let init_w = normalize (Tensor.randn [1; d]) in
        Oja.streaming_pca config chain init_w
    | Adaptive ->
        AdaptiveMechanisms.create_adaptive_algorithm config chain
    | Federated ->
        match network_opt with
        | Some network -> Federation.create_federated_algorithm config network
        | None -> failwith "Network configuration required for federated mode"

  let compute_error w chain =
    let d = Tensor.size chain.states.(0).covariance ~dim:0 in
    let true_cov = ref (Tensor.zeros [d; d]) in
    Array.iteri (fun i state ->
      let pi_i = Tensor.get chain.stationary_dist ~idx:[i] in
      let weighted_cov = Tensor.mul_scalar state.covariance pi_i in
      true_cov := Tensor.add !true_cov weighted_cov
    ) chain.states;
    
    let true_v1 = compute_top_eigvec !true_cov in
    1.0 -. (Tensor.dot w true_v1 |> Tensor.item) ** 2.0
end
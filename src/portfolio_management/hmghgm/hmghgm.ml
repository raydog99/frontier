open Torch

let mahalanobis_distance y mu sigma =
  let diff = Tensor.sub y mu in
  let inv_sigma = Tensor.inverse sigma in
  Tensor.mm (Tensor.mm diff inv_sigma) (Tensor.transpose diff ~dim0:0 ~dim1:1)
  |> Tensor.to_float0_exn

let l1_offdiag matrix =
  let d = Tensor.size matrix.(0) in
  let sum = ref 0.0 in
  for i = 0 to d - 1 do
    for j = 0 to d - 1 do
      if i <> j then
        sum := !sum +. abs_float (Tensor.get (Tensor.slice matrix [i]) j)
    done
  done;
  !sum

let log_sum_exp v =
  let max_v = Array.fold_left max neg_infinity v in
  let sum = Array.fold_left (fun acc x -> acc +. exp (x -. max_v)) 0.0 v in
  max_v +. log sum

let safe_inverse mat =
  let svd = Tensor.svd mat in
  let s = Tensor.to_float1_exn (Tensor.get_data1 svd) in
  let cond = s.(0) /. s.(Array.length s - 1) in
  if cond > 1e15 then
    let d = Tensor.size mat.(0) in
    let reg = Tensor.mul_scalar (Tensor.eye d) 1e-8 in
    Tensor.inverse (Tensor.add mat reg)
  else
    Tensor.inverse mat

let stable_covariance_update data weights =
  let n, d = Tensor.size data.(0), Tensor.size data.(1) in
  let mu = Tensor.mean data ~dim:[0] in
  let cov = Tensor.zeros [d; d] in
  let weight_sum = Array.fold_left (+.) 0.0 weights in
  for i = 0 to n - 1 do
    let x = Tensor.sub (Tensor.slice data [i]) mu in
    let w = weights.(i) /. weight_sum in
    let wx = Tensor.mul_scalar x w in
    Tensor.add_ cov (Tensor.mm (Tensor.transpose wx ~dim0:0 ~dim1:1) x)
  done;
  cov

let normalize_covariance sigma =
  let d = Tensor.size sigma.(0) in
  let det = Tensor.det sigma |> Tensor.to_float0_exn in
  let scale = Float.pow (abs_float det) (-1.0 /. float_of_int d) in
  Tensor.mul_scalar sigma scale

let constrain_covariance sigma =
  let d = Tensor.size sigma.(0) in
  let vals, vecs = Tensor.eig sigma in
  let min_eig = 1e-6 in
  let new_vals = Tensor.map (fun x -> 
    if x < min_eig then min_eig else x) vals in
  let scaled_vecs = Tensor.mm vecs (Tensor.diag new_vals) in
  let new_sigma = Tensor.mm scaled_vecs 
                  (Tensor.transpose vecs ~dim0:0 ~dim1:1) in
  let sym_sigma = Tensor.mul_scalar 
    (Tensor.add new_sigma 
     (Tensor.transpose new_sigma ~dim0:0 ~dim1:1)) 0.5 in
  normalize_covariance sym_sigma

module GH = struct
  type t = {
    mu: Tensor.t;
    sigma: Tensor.t;
    theta: Tensor.t;
    lambda: float;
    chi: float;
    psi: float;
    d: int;
  }

  let bessel_k v x =
    let x = abs_float x in
    if x < 1e-10 then invalid_arg "bessel_k: x too close to 0"
    else if x <= 2.0 then begin
      let rec sum_terms n acc =
        if n > 20 then acc
        else
          let term = Float.pow (-1.0) (float_of_int n) *. 
                    Float.pow (x /. 2.0) (2.0 *. float_of_int n +. v) /.
                    (exp (Stdlib.log_gamma (float_of_int n +. 1.0) +. 
                         Stdlib.log_gamma (v +. float_of_int n +. 1.0))) in
          if abs_float term < 1e-15 *. abs_float acc then acc
          else sum_terms (n + 1) (acc +. term)
      in
      Float.pow (x /. 2.0) v *. sum_terms 0 0.0
    end else begin
      let inv_x = 1.0 /. x in
      sqrt (Float.pi /. (2.0 *. x)) *. exp (-. x) *.
      (1.0 +. (4.0 *. v *. v -. 1.0) *. inv_x /. 8.0 +.
       (4.0 *. v *. v -. 1.0) *. (4.0 *. v *. v -. 9.0) *. inv_x *. inv_x /. 128.0)
    end

  let create ~mu ~sigma ~lambda ~chi ~psi =
    let d = Tensor.size mu.(0) in
    let det = Tensor.det sigma |> Tensor.to_float0_exn in
    let scale = Float.pow (abs_float det) (-1.0 /. float_of_int d) in
    let normalized_sigma = Tensor.mul_scalar sigma scale in
    {
      mu;
      sigma = normalized_sigma;
      theta = Tensor.inverse normalized_sigma;
      lambda;
      chi;
      psi;
      d;
    }

  let density t y =
    let delta = mahalanobis_distance y t.mu t.sigma in
    let term1 = 1.0 /. 
      (Float.pow (2.0 *. Float.pi) (float_of_int t.d /. 2.0)) in
    let term2 = 1.0 /. bessel_k t.lambda (sqrt (t.psi *. t.chi)) in
    let term3 = Float.pow ((t.chi +. delta) /. t.psi) 
                (t.lambda -. float_of_int t.d /. 2.0) /. 2.0 in
    let term4 = bessel_k (t.lambda -. float_of_int t.d /. 2.0) 
                (sqrt ((t.chi +. delta) *. t.psi)) in
    term1 *. term2 *. term3 *. term4

  let mixture_representation t =
    let z = Tensor.randn [t.d] ~dtype:Float in
    let w = GIG.sample ~lambda:t.lambda ~chi:t.chi ~psi:t.psi in
    let sqrt_w = sqrt w in
    let sigma_sqrt = Tensor.cholesky t.sigma in 
    let scaled_sigma = Tensor.mul_scalar sigma_sqrt sqrt_w in
    Tensor.add t.mu (Tensor.mv scaled_sigma z)

  module Hierarchical = struct
    type state = {
      w: float;
      y: Tensor.t;
    }

    let sample t =
      let w = GIG.sample ~lambda:t.lambda ~chi:t.chi ~psi:t.psi in
      let cov = Tensor.mul_scalar t.sigma w in
      let y = Tensor.multivariate_normal t.mu cov in
      {w; y}
  end
end

module GIG = struct
  type t = {
    lambda: float;
    chi: float;
    psi: float;
  }

  let density t x =
    if x <= 0.0 then 0.0
    else 
      let norm = 2.0 *. GH.bessel_k t.lambda (sqrt (t.chi *. t.psi)) in
      let term1 = Float.pow (t.psi /. t.chi) (t.lambda /. 2.0) /. norm in
      let term2 = Float.pow x (t.lambda -. 1.0) in
      let term3 = exp (-0.5 *. (t.chi /. x +. t.psi *. x)) in
      term1 *. term2 *. term3

  let sample ~lambda ~chi ~psi =
    let t = {lambda; chi; psi} in
    let mode = 
      if lambda >= 1.0 then
        let term = sqrt (lambda *. lambda -. 1.0) in
        (term +. lambda) *. sqrt (chi /. psi)
      else
        let term = sqrt (1.0 -. lambda *. lambda) in
        chi /. (term +. lambda *. sqrt (psi /. chi))
    in
    
    let rec sample_gig () =
      let r = sqrt (lambda *. lambda +. chi *. psi) in
      let s = sqrt (chi /. psi) in
      let m = mode /. s in
      
      let u1 = Random.float 1.0 in
      let u2 = Random.float 1.0 in
      
      let x = m *. (1.0 +. r *. (2.0 *. u1 -. 1.0)) in
      if x <= 0.0 then sample_gig ()
      else
        let density_ratio = density t x /. density t mode in
        if u2 <= density_ratio then x
        else sample_gig ()
    in
    sample_gig ()

  let expected_values t y mu sigma d =
    let delta = mahalanobis_distance y mu sigma in
    
    let e_w = 
      let term = sqrt ((t.chi +. delta) /. t.psi) in
      let num = GH.bessel_k (t.lambda -. float_of_int d /. 2.0 +. 1.0)
                  (sqrt ((t.chi +. delta) *. t.psi)) in
      let den = GH.bessel_k (t.lambda -. float_of_int d /. 2.0)
                  (sqrt ((t.chi +. delta) *. t.psi)) in
      term *. num /. den
    in
    
    let e_inv_w =
      let term = sqrt ((t.chi +. delta) /. t.psi) in
      let num = GH.bessel_k (t.lambda -. float_of_int d /. 2.0 -. 1.0)
                  (sqrt ((t.chi +. delta) *. t.psi)) in
      let den = GH.bessel_k (t.lambda -. float_of_int d /. 2.0)
                  (sqrt ((t.chi +. delta) *. t.psi)) in
      term *. num /. den -. 
      2.0 *. (t.lambda -. float_of_int d /. 2.0) /. (t.chi +. delta)
    in
    
    let e_log_w =
      let sqrt_term = sqrt ((t.chi +. delta) /. t.psi) in
      log sqrt_term +.
      (let bk = GH.bessel_k (t.lambda -. float_of_int d /. 2.0)
                  (sqrt ((t.chi +. delta) *. t.psi)) in
       log bk)
    in
    
    e_w, e_inv_w, e_log_w
end

module HMM = struct
  type t = {
    k: int;
    d: int;
    pi: Tensor.t;
    transition: Tensor.t;
    gh_params: GH.t array;
    theta: Tensor.t array;
  }

  let create ~k ~d ~pi ~transition ~gh_params =
    if Tensor.size pi.(0) <> k then
      invalid_arg "Initial probability vector must have length k";
    if Tensor.size transition.(0) <> k || Tensor.size transition.(1) <> k then
      invalid_arg "Transition matrix must be k x k";
    if Array.length gh_params <> k then
      invalid_arg "Must provide k GH distributions";
    
    let theta = Array.map (fun gh -> gh.GH.theta) gh_params in
    {k; d; pi; transition; gh_params; theta}

  module StateGraph = struct
    type edge = {
      from_node: int;
      to_node: int;
      weight: float;
    }

    type t = {
      state: int;
      nodes: int array;
      edges: edge list;
      precision: Tensor.t;
    }

    let from_precision state prec =
      let d = Tensor.size prec.(0) in
      let nodes = Array.init d (fun i -> i) in
      let edges = ref [] in
      
      for i = 0 to d - 1 do
        for j = i + 1 to d - 1 do
          let weight = Tensor.get (Tensor.slice prec [i]) j in
          if abs_float weight > 1e-10 then
            edges := {from_node = i; to_node = j; weight} :: !edges
        done
      done;
      
      {state; nodes; edges = !edges; precision = prec}

    let is_conditionally_independent graph j h c =
      let theta_jh = Tensor.get (Tensor.slice graph.precision [j]) h in
      abs_float theta_jh < 1e-10 ||
      let rec is_path_through_c visited curr target =
        if List.mem curr c then false
        else if curr = target then true
        else if List.mem curr visited then false
        else
          let next_nodes = List.filter_map 
            (fun edge -> 
              if edge.from_node = curr then Some edge.to_node
              else if edge.to_node = curr then Some edge.from_node
              else None) 
            graph.edges in
          List.exists (fun next -> 
            is_path_through_c (curr :: visited) next target) 
            next_nodes
      in
      not (is_path_through_c [] j h)
  end

  let emission_probability t obs state =
    GH.density t.gh_params.(state) obs

  let sequence_probability t obs =
    let n = Tensor.size obs.(0) in
    let alpha = Tensor.zeros [n; t.k] in
    
    (* Initialize *)
    for k = 0 to t.k - 1 do
      let emit = emission_probability t (Tensor.slice obs [0]) k in
      Tensor.set_ (Tensor.slice alpha [0]) k (emit *. Tensor.get t.pi k)
    done;
    
    (* Forward recursion *)
    for i = 1 to n - 1 do
      for j = 0 to t.k - 1 do
        let sum_term = ref 0.0 in
        for k = 0 to t.k - 1 do
          sum_term := !sum_term +. 
            (Tensor.get (Tensor.slice alpha [i-1]) k) *. 
            (Tensor.get (Tensor.slice t.transition [k]) j)
        done;
        let emit = emission_probability t (Tensor.slice obs [i]) j in
        Tensor.set_ (Tensor.slice alpha [i]) j (!sum_term *. emit)
      done
    done;
    
    Tensor.sum (Tensor.slice alpha [n-1]) |> Tensor.to_float0_exn

  let sample t n =
    let obs = Tensor.zeros [n; t.d] in
    let states = Array.make n 0 in
    
    (* Sample initial state *)
    let init_state = ref 0 in
    let prob = Random.float 1.0 in
    let sum = ref 0.0 in
    for k = 0 to t.k - 1 do
      sum := !sum +. Tensor.get t.pi k;
      if prob <= !sum then (
        init_state := k;
        raise Exit
      )
    done;
    states.(0) <- !init_state;
    
    (* Generate first observation *)
    let init_obs = GH.mixture_representation t.gh_params.(!init_state) in
    Tensor.copy_ (Tensor.slice obs [0]) init_obs;
    
    (* Generate remaining sequence *)
    for i = 1 to n - 1 do
      let prev_state = states.(i-1) in
      let next_state = ref 0 in
      let prob = Random.float 1.0 in
      let sum = ref 0.0 in
      for k = 0 to t.k - 1 do
        sum := !sum +. Tensor.get (Tensor.slice t.transition [prev_state]) k;
        if prob <= !sum then (
          next_state := k;
          raise Exit
        )
      done;
      states.(i) <- !next_state;
      
      let next_obs = GH.mixture_representation t.gh_params.(!next_state) in
      Tensor.copy_ (Tensor.slice obs [i]) next_obs
    done;
    
    obs, states

  let update_state_parameters t obs weights state =
    let sample_cov = stable_covariance_update obs weights in
    let constrained_cov = constrain_covariance sample_cov in
    let theta = PenalizedECME.graphical_lasso_penalized 
                  constrained_cov 0.1 1.0 100 in
    let graph = StateGraph.from_precision state theta in
    let new_gh = GH.create
      ~mu:(Tensor.mean obs ~dim:[0])
      ~sigma:constrained_cov
      ~lambda:t.gh_params.(state).GH.lambda
      ~chi:t.gh_params.(state).GH.chi
      ~psi:t.gh_params.(state).GH.psi in
    theta, graph, new_gh
end

module ForwardBackward = struct
  type scaled_probs = {
    alpha: Tensor.t;
    beta: Tensor.t;
    scaling: float array;
  }

  let forward hmm obs =
    let t_len = Tensor.size obs.(0) in
    let alpha = Tensor.zeros [t_len; hmm.k] in
    let scaling = Array.make t_len 0.0 in
    let log_scaling = Array.make t_len 0.0 in

    (* Initialize *)
    for k = 0 to hmm.k - 1 do
      let log_pi_k = log (Tensor.get hmm.pi k) in
      let log_emit = log (HMM.emission_probability hmm (Tensor.slice obs [0]) k) in
      let log_alpha = log_pi_k +. log_emit in
      Tensor.set_ (Tensor.slice alpha [0]) k (exp log_alpha);
      log_scaling.(0) <- log_sum_exp [|log_alpha|]
    done;
    scaling.(0) <- exp log_scaling.(0);

    (* Scale initial probabilities *)
    for k = 0 to hmm.k - 1 do
      let scaled = (Tensor.get (Tensor.slice alpha [0]) k) /. scaling.(0) in
      Tensor.set_ (Tensor.slice alpha [0]) k scaled
    done;

    (* Forward recursion *)
    for t = 1 to t_len - 1 do
      let log_alpha_t = Array.make hmm.k neg_infinity in
      
      for j = 0 to hmm.k - 1 do
        let log_sums = Array.make hmm.k neg_infinity in
        for i = 0 to hmm.k - 1 do
          let log_alpha_prev = log (Tensor.get (Tensor.slice alpha [t-1]) i) +. 
                             Array.fold_left (+.) 0.0 log_scaling in
          let log_trans = log (Tensor.get (Tensor.slice hmm.transition [i]) j) in
          log_sums.(i) <- log_alpha_prev +. log_trans
        done;
        
        let log_emit = log (HMM.emission_probability hmm 
                            (Tensor.slice obs [t]) j) in
        log_alpha_t.(j) <- log_sum_exp log_sums +. log_emit
      done;

      (* Scale and store *)
      log_scaling.(t) <- log_sum_exp log_alpha_t;
      scaling.(t) <- exp log_scaling.(t);
      
      for j = 0 to hmm.k - 1 do
        let scaled = exp (log_alpha_t.(j) -. log_scaling.(t)) in
        Tensor.set_ (Tensor.slice alpha [t]) j scaled
      done
    done;

    alpha, scaling, log_scaling

  let backward hmm obs scaling log_scaling =
    let t_len = Tensor.size obs.(0) in
    let beta = Tensor.zeros [t_len; hmm.k] in

    (* Initialize *)
    for k = 0 to hmm.k - 1 do
      Tensor.set_ (Tensor.slice beta [t_len-1]) k (1.0 /. scaling.(t_len-1))
    done;

    (* Backward recursion *)
    for t = t_len - 2 downto 0 do
      let log_beta_t = Array.make hmm.k neg_infinity in

      for i = 0 to hmm.k - 1 do
        let log_sums = Array.make hmm.k neg_infinity in
        for j = 0 to hmm.k - 1 do
          let log_beta_next = log (Tensor.get (Tensor.slice beta [t+1]) j) in
          let log_trans = log (Tensor.get (Tensor.slice hmm.transition [i]) j) in
          let log_emit = log (HMM.emission_probability hmm 
                              (Tensor.slice obs [t+1]) j) in
          log_sums.(j) <- log_beta_next +. log_trans +. log_emit
        done;
        log_beta_t.(i) <- log_sum_exp log_sums
      done;

      (* Scale and store *)
      for i = 0 to hmm.k - 1 do
        let scaled = exp (log_beta_t.(i) -. log_scaling.(t)) in
        Tensor.set_ (Tensor.slice beta [t]) i scaled
      done
    done;

    beta

  let smooth hmm obs =
    let alpha, scaling, log_scaling = forward hmm obs in
    let beta = backward hmm obs scaling log_scaling in
    let gamma = Tensor.mul alpha beta in

    (* Compute xi (transition probabilities) *)
    let t_len = Tensor.size obs.(0) in
    let xi = Tensor.zeros [t_len-1; hmm.k; hmm.k] in

    for t = 0 to t_len - 2 do
      for i = 0 to hmm.k - 1 do
        for j = 0 to hmm.k - 1 do
          let log_xi = log (Tensor.get (Tensor.slice alpha [t]) i) +.
                      log (Tensor.get (Tensor.slice hmm.transition [i]) j) +.
                      log (HMM.emission_probability hmm 
                             (Tensor.slice obs [t+1]) j) +.
                      log (Tensor.get (Tensor.slice beta [t+1]) j) -.
                      log_scaling.(t+1) in
          Tensor.set_ (Tensor.slice (Tensor.slice xi [t]) [i]) j (exp log_xi)
        done
      done
    done;

    {alpha; beta; scaling}, gamma, xi
end

module ECME = struct
  type params = {
    hmm: HMM.t;
    rho: float;
    state_weights: float array;
  }

  let e_step params obs =
    let scaled_probs, gamma, xi = ForwardBackward.smooth params.hmm obs in
    gamma, xi

  let cm_step1 params gamma xi =
    let t_len = Tensor.size gamma.(0) in
    
    (* Update initial probabilities *)
    let new_pi = Tensor.copy (Tensor.slice gamma [0]) in
    
    (* Update transition matrix *)
    let new_trans = Tensor.zeros [params.hmm.k; params.hmm.k] in
    for j = 0 to params.hmm.k - 1 do
      let denominator = ref 0.0 in
      for t = 0 to t_len - 2 do
        for k = 0 to params.hmm.k - 1 do
          denominator := !denominator +. Tensor.get 
            (Tensor.slice (Tensor.slice xi [t]) [j]) k
        done
      done;
      
      if !denominator > 0.0 then
        for k = 0 to params.hmm.k - 1 do
          let numerator = ref 0.0 in
          for t = 0 to t_len - 2 do
            numerator := !numerator +. Tensor.get 
              (Tensor.slice (Tensor.slice xi [t]) [j]) k
          done;
          Tensor.set_ (Tensor.slice new_trans [j]) k 
            (!numerator /. !denominator)
        done
    done;
    
    new_pi, new_trans

  let cm_step2 params obs gamma =
    Array.init params.hmm.k (fun k ->
      (* Update location parameter *)
      let mu_num = ref (Tensor.zeros [params.hmm.hmm.d]) in
      let mu_den = ref 0.0 in
      
      let t_len = Tensor.size obs.(0) in
      for t = 0 to t_len - 1 do
        let weight = Tensor.get (Tensor.slice gamma [t]) k in
        let w_tk = GIG.sample 
          ~lambda:params.hmm.gh_params.(k).GH.lambda
          ~chi:params.hmm.gh_params.(k).GH.chi
          ~psi:params.hmm.gh_params.(k).GH.psi in
        
        mu_num := Tensor.add !mu_num 
          (Tensor.mul_scalar (Tensor.slice obs [t]) (weight *. w_tk));
        mu_den := !mu_den +. weight *. w_tk
      done;
      
      let new_mu = Tensor.div !mu_num (Tensor.full [1] !mu_den) in
      
      (* Update scale matrix *)
      let sigma_sum = ref (Tensor.zeros [params.hmm.hmm.d; params.hmm.hmm.d]) in
      let sigma_den = ref 0.0 in
      
      for t = 0 to t_len - 1 do
        let weight = Tensor.get (Tensor.slice gamma [t]) k in
        let w_tk = GIG.sample 
          ~lambda:params.hmm.gh_params.(k).GH.lambda
          ~chi:params.hmm.gh_params.(k).GH.chi
          ~psi:params.hmm.gh_params.(k).GH.psi in
        
        let diff = Tensor.sub (Tensor.slice obs [t]) new_mu in
        let outer = Tensor.mm (Tensor.transpose diff ~dim0:0 ~dim1:1) diff in
        sigma_sum := Tensor.add !sigma_sum 
          (Tensor.mul_scalar outer (weight *. w_tk));
        sigma_den := !sigma_den +. weight
      done;
      
      let new_sigma = normalize_covariance 
        (Tensor.div !sigma_sum (Tensor.full [1] !sigma_den)) in
      
      GH.create 
        ~mu:new_mu
        ~sigma:new_sigma
        ~lambda:params.hmm.gh_params.(k).GH.lambda
        ~chi:params.hmm.gh_params.(k).GH.chi
        ~psi:params.hmm.gh_params.(k).GH.psi)

  let cm_step3 params obs gamma =
    Array.mapi (fun k gh ->
      let objective params =
        let lambda, chi, psi = params.(0), params.(1), params.(2) in
        let new_gh = GH.create 
          ~mu:gh.GH.mu 
          ~sigma:gh.GH.sigma
          ~lambda ~chi ~psi in
        
        let ll = ref 0.0 in
        let t_len = Tensor.size obs.(0) in
        for t = 0 to t_len - 1 do
          let weight = Tensor.get (Tensor.slice gamma [t]) k in
          let density = GH.density new_gh (Tensor.slice obs [t]) in
          if density > 0.0 && weight > 0.0 then
            ll := !ll +. weight *. log density
        done;
        -. !ll
      in
      
      (* Numerical gradient *)
      let gradient params =
        let eps = 1e-6 in
        let grad = Array.make 3 0.0 in
        for i = 0 to 2 do
          let params_plus = Array.copy params in
          let params_minus = Array.copy params in
          params_plus.(i) <- params.(i) +. eps;
          params_minus.(i) <- params.(i) -. eps;
          grad.(i) <- (objective params_plus -. 
                      objective params_minus) /. (2.0 *. eps)
        done;
        grad
      in
      
      (* BFGS optimization *)
      let init_params = [|gh.GH.lambda; gh.GH.chi; gh.GH.psi|] in
      let final_params = ref init_params in
      let prev_obj = ref (objective init_params) in
      
      for _ = 1 to 100 do
        let grad = gradient !final_params in
        let step_size = 0.01 in
        let new_params = Array.map2 (fun p g -> p -. step_size *. g) 
                          !final_params grad in
        let new_obj = objective new_params in
        if new_obj < !prev_obj then (
          final_params := new_params;
          prev_obj := new_obj
        )
      done;
      
      GH.create 
        ~mu:gh.GH.mu
        ~sigma:gh.GH.sigma
        ~lambda:!final_params.(0)
        ~chi:!final_params.(1)
        ~psi:!final_params.(2)) params.hmm.gh_params

  let optimize ~init_params ~data ~max_iter ~tol =
    let params = ref init_params in
    let prev_ll = ref Float.neg_infinity in
    let iter = ref 0 in
    
    while !iter < max_iter do
      (* E-step *)
      let gamma, xi = e_step !params data in
      
      (* CM-steps *)
      let new_pi, new_trans = cm_step1 !params gamma xi in
      let new_gh_params = cm_step2 !params data gamma in
      let final_gh_params = cm_step3 !params data gamma in
      
      (* Update parameters *)
      params := {
        !params with
        hmm = {(!params).hmm with
               pi = new_pi;
               transition = new_trans;
               gh_params = final_gh_params}
      };
      
      (* Check convergence *)
      let ll = HMM.sequence_probability (!params).hmm data in
      if abs_float (ll -. !prev_ll) < tol then
        !params
      else (
        prev_ll := ll;
        incr iter
      )
    done;
    !params
end

module PenalizedECME = struct
  type params = {
    hmm: HMM.t;
    rho: float;
    state_weights: float array;
    max_iter: int;
    tol: float;
  }

  let penalized_likelihood params sample_cov gamma k =
    let d = Tensor.size sample_cov.(0) in
    let theta = params.hmm.theta.(k) in
    
    let log_det = Tensor.det theta |> Tensor.to_float0_exn |> log in
    let trace = Tensor.trace (Tensor.mm sample_cov theta) 
                |> Tensor.to_float0_exn in
    let l1_norm = l1_offdiag theta in
    
    (log_det -. trace) /. 2.0 -. 
    params.rho *. sqrt params.state_weights.(k) *. l1_norm

  let weighted_covariance obs weights mu w_tk =
    stable_covariance_update obs (Array.map2 ( *. ) weights w_tk)

  let graphical_lasso_penalized s_tilde rho nu_k max_iter =
    let d = Tensor.size s_tilde.(0) in
    let theta = Tensor.eye d in
    let w = Tensor.copy s_tilde in
    
    for _ = 1 to max_iter do
      for i = 0 to d - 1 do
        let not_i = Array.init (d-1) (fun j -> if j >= i then j + 1 else j) in
        
        let w11 = Tensor.get (Tensor.slice w [i]) i in
        let w12 = Tensor.index_select w 0 (Tensor.of_int1 not_i) in
        let theta12 = Tensor.index_select theta 0 (Tensor.of_int1 not_i) in
        
        (* Solve penalized problem *)
        let beta = ref (Tensor.zeros [d-1]) in
        let max_coord_iter = 100 in
        let tol = 1e-6 in
        
        for _ = 1 to max_coord_iter do
          let max_diff = ref 0.0 in
          for j = 0 to d - 2 do
            let old_beta = Tensor.get !beta j in
            
            let s = Tensor.get (Tensor.slice w12 [j]) 0 -. 
                    Tensor.dot (Tensor.slice theta12 [j]) !beta +. 
                    w11 *. old_beta in
            
            let new_beta = 
              if s > rho *. sqrt nu_k then (s -. rho *. sqrt nu_k) /. w11
              else if s < -. rho *. sqrt nu_k then (s +. rho *. sqrt nu_k) /. w11
              else 0.0 in
            
            Tensor.set_ !beta j new_beta;
            max_diff := max !max_diff (abs_float (new_beta -. old_beta))
          done;
          
          if !max_diff < tol then raise Exit
        done;
        
        (* Update precision matrix *)
        Array.iteri (fun j idx ->
          Tensor.set_ (Tensor.slice theta [i]) idx (Tensor.get !beta j);
          Tensor.set_ (Tensor.slice theta [idx]) i (Tensor.get !beta j))
          not_i
      done
    done;
    theta

  let optimize_penalized ~init_params ~data ~max_iter ~tol =
    let params = ref init_params in
    let prev_ll = ref Float.neg_infinity in
    
    for iter = 1 to max_iter do
      (* E-step *)
      let gamma, xi = ECME.e_step !params data in
      
      (* CM-steps with penalization *)
      let new_pi, new_trans = ECME.cm_step1 !params gamma xi in
      
      (* Update GH parameters with penalty *)
      for k = 0 to (!params).hmm.k - 1 do
        let t_len = Tensor.size data.(0) in
        let weights = Array.init t_len 
          (fun t -> Tensor.get (Tensor.slice gamma [t]) k) in
        let new_mu = Tensor.mean data ~dim:[0] in
        
        let w_tk = Array.init t_len (fun _ ->
          GIG.sample ~lambda:(!params).hmm.gh_params.(k).GH.lambda
                    ~chi:(!params).hmm.gh_params.(k).GH.chi
                    ~psi:(!params).hmm.gh_params.(k).GH.psi) in
        
        let s_tilde = weighted_covariance data weights new_mu w_tk in
        
        let new_theta = graphical_lasso_penalized s_tilde 
          (!params).rho (!params).state_weights.(k) 100 in
        
        let new_sigma = Tensor.inverse new_theta in
        
        (!params).hmm.gh_params.(k) <- GH.create
          ~mu:new_mu
          ~sigma:new_sigma
          ~lambda:(!params).hmm.gh_params.(k).GH.lambda
          ~chi:(!params).hmm.gh_params.(k).GH.chi
          ~psi:(!params).hmm.gh_params.(k).GH.psi;
        (!params).hmm.theta.(k) <- new_theta
      done;
      
      (* Update remaining GH parameters *)
      let final_gh_params = ECME.cm_step3 !params data gamma in
      
      (* Update parameters *)
      params := {
        !params with
        hmm = {(!params).hmm with
               pi = new_pi;
               transition = new_trans;
               gh_params = final_gh_params}
      };
      
      (* Check convergence *)
      let ll = ref 0.0 in
      for k = 0 to (!params).hmm.k - 1 do
        let weights = Array.init (Tensor.size data.(0)) 
          (fun t -> Tensor.get (Tensor.slice gamma [t]) k) in
        let s_tilde = weighted_covariance data weights 
          (!params).hmm.gh_params.(k).GH.mu
          (Array.make (Tensor.size data.(0)) 1.0) in
        ll := !ll +. penalized_likelihood !params s_tilde gamma k
      done;
      
      if abs_float (!ll -. !prev_ll) < tol then
        !params
      else
        prev_ll := !ll
    done;
    !params

  let select_penalty_parameter data penalties =
    let best_params = ref None in
    let best_bic = ref Float.infinity in
    
    Array.iter (fun rho ->
      let init_params = {
        hmm = HMM.create 
          ~k:2  
          ~d:(Tensor.size data.(1))
          ~pi:(Tensor.ones [2])
          ~transition:(Tensor.ones [2; 2])
          ~gh_params:[|GH.create 
            ~mu:(Tensor.zeros [Tensor.size data.(1)])
            ~sigma:(Tensor.eye (Tensor.size data.(1)))
            ~lambda:1.0
            ~chi:1.0
            ~psi:1.0;
            GH.create
            ~mu:(Tensor.zeros [Tensor.size data.(1)])
            ~sigma:(Tensor.eye (Tensor.size data.(1)))
            ~lambda:1.0
            ~chi:1.0
            ~psi:1.0|];
        rho;
        state_weights = [|0.5; 0.5|];
        max_iter = 100;
        tol = 1e-6;
      } in
      
      (* Fit model with current penalty parameter *)
      let fitted = optimize_penalized ~init_params ~data 
                    ~max_iter:100 ~tol:1e-6 in
      
      (* Compute BIC *)
      let ll = ref 0.0 in
      let gamma, _ = ECME.e_step fitted data in
      for k = 0 to fitted.hmm.k - 1 do
        let weights = Array.init (Tensor.size data.(0)) 
          (fun t -> Tensor.get (Tensor.slice gamma [t]) k) in
        let s_tilde = weighted_covariance data weights 
          fitted.hmm.gh_params.(k).GH.mu
          (Array.make (Tensor.size data.(0)) 1.0) in
        ll := !ll +. penalized_likelihood fitted s_tilde gamma k
      done;
      
      (* Count non-zero parameters *)
      let count_nonzeros = ref 0 in
      Array.iter (fun theta ->
        let d = Tensor.size theta.(0) in
        for i = 0 to d - 1 do
          for j = i + 1 to d - 1 do
            if abs_float (Tensor.get (Tensor.slice theta [i]) j) > 1e-10 then
              incr count_nonzeros
          done
        done) fitted.hmm.theta;
      
      let n = Tensor.size data.(0) in
      let bic = -2.0 *. !ll +. 
                (log (float_of_int n)) *. float_of_int !count_nonzeros in
      
      if bic < !best_bic then begin
        best_bic := bic;
        best_params := Some fitted
      end) penalties;
    
    match !best_params with
    | Some params -> params
    | None -> failwith "No valid parameters found"
end
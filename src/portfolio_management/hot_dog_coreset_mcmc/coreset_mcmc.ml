open Torch

type log_likelihood = Tensor.t -> Tensor.t
type log_prior = Tensor.t -> Tensor.t
type markov_kernel = Tensor.t -> Tensor.t -> Tensor.t

let uniform_subsample ~size ~n =
  let indices = Array.init n (fun i -> i) in
  let rec shuffle i =
    if i <= 0 then indices
    else begin
      let j = Random.int (i + 1) in
      let tmp = indices.(i) in
      indices.(i) <- indices.(j);
      indices.(j) <- tmp;
      shuffle (i - 1)
    end in
  Array.sub (shuffle (n - 1)) 0 size

let linear_detrend values =
  let n = Array.length values in
  let x = Array.init n float_of_int in
  let y = Array.copy values in
  
  let sum_x = Array.fold_left (+.) 0. x in
  let sum_y = Array.fold_left (+.) 0. y in
  let sum_xy = Array.fold_left2 (fun acc x y -> acc +. x *. y) 0. x y in
  let sum_xx = Array.fold_left (fun acc x -> acc +. x *. x) 0. x in
  
  let slope = (float_of_int n *. sum_xy -. sum_x *. sum_y) /. 
             (float_of_int n *. sum_xx -. sum_x *. sum_x) in
  let intercept = (sum_y -. slope *. sum_x) /. float_of_int n in
  
  let detrended = Array.mapi 
    (fun i y -> y -. (slope *. float_of_int i +. intercept)) y in
  
  let mean = Array.fold_left (+.) 0. detrended /. float_of_int n in
  let variance = Array.fold_left 
    (fun acc v -> acc +. (v -. mean) ** 2.) 0. detrended /. 
    float_of_int (n - 2) in
  
  (mean, variance)

let estimate ~coreset ~states ~subsample_indices =
  let n_chains = Array.length states in
  let k_minus_1 = float_of_int (n_chains - 1) in
  
  (* Center log-likelihoods *)
  let centered_lls = Array.map (fun state ->
    let lls = Array.map (fun ll -> ll state) coreset.Coreset.log_likelihoods in
    let mean_ll = mean (stack lls) ~dim:[0] in
    Array.map (fun ll -> sub ll mean_ll) lls
  ) states in

  (* Compute covariance terms *)
  let cov_terms = Array.map (fun chain_lls ->
    (* Weighted sum term *)
    let weighted_sum = Array.fold_left2
      (fun acc w ll -> add acc (mul w ll))
      (zeros [])
      (Array.to_list coreset.Coreset.weights)
      chain_lls in
    
    (* Subsample sum term *)
    let subsample_sum = Array.fold_left
      (fun acc idx -> add acc chain_lls.(idx))
      (zeros [])
      subsample_indices in

    let scale = float_of_int (Array.length chain_lls) /. 
               float_of_int (Array.length subsample_indices) in
    (weighted_sum, mul subsample_sum (float_tensor scale))
  ) centered_lls in
  
  (* Combine terms *)
  let sum_terms = Array.fold_left
    (fun (acc_w, acc_s) (w, s) ->
      (add acc_w w, add acc_s s))
    (zeros [], zeros [])
    cov_terms in

  let scale = float_tensor (1. /. k_minus_1) in
  sub
    (mul (fst sum_terms) scale)
    (mul (snd sum_terms) scale)

module Coreset = struct
  type t = {
    weights: Tensor.t;
    log_likelihoods: log_likelihood array;
    log_prior: log_prior;
  }

  let create ~weights ~log_likelihoods ~log_prior = 
    {weights; log_likelihoods; log_prior}

  let log_posterior t theta =
    let weighted_sum = Array.fold_left2
      (fun acc w ll -> 
        add acc (mul w (ll theta)))
      (zeros [])
      (Array.to_list t.weights)
      t.log_likelihoods in
    
    add weighted_sum (t.log_prior theta)
end

module HotDog = struct
  type state = {
    v: Tensor.t;
    m: Tensor.t;
    d: Tensor.t;
    c: int;
    h: bool;
    log_potentials: float array;
  }

  let create_state weights = {
    v = zeros_like weights;
    m = zeros_like weights;
    d = zeros_like weights;
    c = 0;
    h = false;
    log_potentials = [||];
  }

  let hot_start_test log_potentials =
    let t = Array.length log_potentials in
    let n = t / 3 in
    if n < 1 then false
    else
      let (m1, v1) = linear_detrend 
        (Array.sub log_potentials n n) in
      let (m2, v2) = linear_detrend 
        (Array.sub log_potentials (2*n) n) in
      abs_float (m1 -. m2) /. (max (sqrt v1) (sqrt v2)) < 0.5

  let update ~beta1 ~beta2 ~epsilon ~r ~state ~weights ~grad =
    if not state.h then
      let h = hot_start_test state.log_potentials in
      {state with h}, weights
    else
      let c = state.c + 1 in
      
      (* Update moments with RMSProp-style acceleration *)
      let v = add 
        (mul (float_tensor beta2) state.v)
        (mul (float_tensor (1. -. beta2)) (square grad)) in
        
      let m = add
        (mul (float_tensor beta1) state.m)
        (mul (float_tensor (1. -. beta1)) grad) in
        
      let d = maximum
        (abs (sub weights state.d))
        (mul (float_tensor beta1) state.d) in
        
      (* Bias corrections *)
      let bc1 = 1. -. (beta1 ** float_of_int c) in
      let bc2 = 1. -. (beta2 ** float_of_int c) in
      
      let v_hat = div v (float_tensor bc2) in
      let m_hat = div m (float_tensor bc1) in
      let d_hat = if c = 1 then
        full_like d r
      else
        div d (float_tensor bc1) in
        
      let new_weights = sub weights
        (mul d_hat 
          (div m_hat 
            (add (sqrt v_hat) 
              (float_tensor epsilon)))) in
              
      {state with v; m; d; c; h = true}, new_weights
end

module CoresetMCMC = struct
  type config = {
    n_chains: int;
    subsample_size: int;
    max_iters: int;
    beta1: float;
    beta2: float;
    epsilon: float;
    r: float;
  }

  let default_config = {
    n_chains = 4;
    subsample_size = 100;
    max_iters = 200_000;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    r = 0.001;
  }

  let run config coreset init_states =
    let hot_dog_state = ref (HotDog.create_state coreset.Coreset.weights) in
    let weights = ref coreset.Coreset.weights in
    let states = ref init_states in
    let samples = ref [] in

    for t = 1 to config.max_iters do
      (* Subsample data *)
      let subsample = uniform_subsample 
        ~size:config.subsample_size 
        ~n:(Array.length coreset.Coreset.log_likelihoods) in

      (* Compute gradient estimate *)
      let grad = estimate 
        ~coreset 
        ~states:!states 
        ~subsample_indices:subsample in

      (* Update weights *)
      let (new_hot_dog_state, new_weights) = HotDog.update
        ~beta1:config.beta1
        ~beta2:config.beta2
        ~epsilon:config.epsilon
        ~r:config.r
        ~state:!hot_dog_state
        ~weights:!weights
        ~grad in

      hot_dog_state := new_hot_dog_state;
      weights := new_weights;

      (* Update Markov chains *)
      let new_states = Array.map (fun state ->
        let log_posterior = Coreset.log_posterior 
          {coreset with weights = !weights} in
        let proposed = add state (randn_like state) in
        let log_ratio = sub 
          (log_posterior proposed) 
          (log_posterior state) in
        let accept_prob = minimum 
          (exp log_ratio)
          (ones_like log_ratio) in
        if Tensor.to_float0_exn (rand_like accept_prob) < 
           Tensor.to_float0_exn accept_prob then
          proposed
        else
          state
      ) !states in

      states := new_states;
      samples := new_states :: !samples
    done;

    (!weights, List.rev !samples)
end
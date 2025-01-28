open Torch

type data_batch = {
    x: Tensor.t;
    y: Tensor.t;
    output_idx: int array;
    input_idx: int array;
}

type training_config = {
    num_epochs: int;
    batch_size: int;
    learning_rate: float;
    num_samples: int;
    num_data: int;
    num_outputs: int;
    jitter: float;
    whiten: bool;
}

(* Matrix operations with numerical stability *)
module MatrixOps = struct
  type decomposition = {
    eigvals: Tensor.t;
    eigvecs: Tensor.t;
  }

  let default_jitter = 1e-6
  let max_jitter = 1.0
  let jitter_factor = 10.0
  let condition_threshold = 1e8

  let safe_decompose mat =
  	let rec attempt jitter =
  		if jitter > max_jitter then
	      let stabilized = Tensor.add mat 
	        (Tensor.mul_scalar (Tensor.eye (Tensor.size mat 0)) max_jitter) in
	      let eigvals, eigvecs = Tensor.symeig stabilized ~eigenvectors:true in
	      {eigvals; eigvecs}
	    else
	      try
	        let stabilized = Tensor.add mat 
	          (Tensor.mul_scalar (Tensor.eye (Tensor.size mat 0)) jitter) in
	        let eigvals, eigvecs = Tensor.symeig stabilized ~eigenvectors:true in
	        if Tensor.min eigvals |> Tensor.item > 0. then
	          {eigvals; eigvecs}
	        else
	          attempt (jitter *. jitter_factor)
	      with _ -> attempt (jitter *. jitter_factor)
	  in attempt default_jitter

  let safe_cholesky mat =
  	let rec attempt jitter =
	    if jitter > max_jitter then
	      let stabilized = Tensor.add mat 
	        (Tensor.mul_scalar (Tensor.eye (Tensor.size mat 0)) max_jitter) in
	      Tensor.cholesky stabilized
	    else
	      try
	        let stabilized = Tensor.add mat 
	          (Tensor.mul_scalar (Tensor.eye (Tensor.size mat 0)) jitter) in
	        Tensor.cholesky stabilized
	      with _ -> attempt (jitter *. jitter_factor)
	  in attempt default_jitter

  let condition_number mat =
    match safe_decompose mat with
    | Ok {eigvals; _} ->
        let max_eigval = Tensor.max eigvals in
        let min_eigval = Tensor.min eigvals in
        Tensor.div max_eigval min_eigval
    | Error msg -> failwith msg

  let safe_inverse mat =
    match safe_decompose mat with
    | Ok {eigvals; eigvecs} ->
        let inv_eigvals = Tensor.reciprocal 
          (Tensor.add eigvals (Tensor.full_like eigvals default_jitter)) in
        let scaled = Tensor.mm 
          (Tensor.mm eigvecs (Tensor.diag inv_eigvals))
          (Tensor.transpose eigvecs ~dim0:0 ~dim1:1) in
        Ok scaled
    | Error msg -> Error msg

  let trace_product a b =
    Tensor.sum (Tensor.mul a (Tensor.transpose b ~dim0:0 ~dim1:1))

  let efficient_trace mat =
    Tensor.sum (Tensor.diagonal mat 0)

  let kron a b =
    let a_shape = Tensor.shape a in
    let b_shape = Tensor.shape b in
    let m1, n1 = a_shape.(0), a_shape.(1) in
    let m2, n2 = b_shape.(0), b_shape.(1) in
    
    let result = Tensor.zeros [m1 * m2; n1 * n2] in
    for i = 0 to m1 - 1 do
      for j = 0 to n1 - 1 do
        let a_ij = Tensor.get a [|i; j|] in
        let sub_matrix = Tensor.mul_scalar b a_ij in
        let start_i = i * m2 in
        let start_j = j * n2 in
        Tensor.copy_ (Tensor.narrow result ~dim:0 ~start:start_i ~length:m2
                     |> Tensor.narrow ~dim:1 ~start:start_j ~length:n2) sub_matrix
      done
    done;
    result
end

(* Kernel module interface *)
module type Kernel = sig
  type t
  val create : ?ard:bool -> int -> t
  val forward : t -> Tensor.t -> Tensor.t -> Tensor.t
  val diag : t -> Tensor.t -> Tensor.t
  val get_params : t -> Tensor.t list
  val set_params : t -> Tensor.t list -> t
  val num_params : t -> int
end

(* SE-ARD Kernel *)
module SE_ARD_Kernel : Kernel = struct
  type t = {
    lengthscales: Tensor.t;
    variance: Tensor.t;
    input_dim: int;
  }

  let create ?(ard=true) input_dim =
    let lengthscales = if ard then
      Tensor.ones [input_dim]
    else
      Tensor.ones [1] in
    let variance = Tensor.ones [1] in
    {lengthscales; variance; input_dim}

  let forward t x y =
    let x2 = Tensor.sum (Tensor.mul x x) ~dim:[1] ~keepdim:true in
    let y2 = Tensor.sum (Tensor.mul y y) ~dim:[1] ~keepdim:true in
    let xy = Tensor.mm x (Tensor.transpose y ~dim0:0 ~dim1:1) in
    let dist = Tensor.add (Tensor.add x2 
      (Tensor.transpose y2 ~dim0:0 ~dim1:1)) 
      (Tensor.mul_scalar xy (-2.)) in
    Tensor.mul t.variance (Tensor.exp (Tensor.mul_scalar dist (-0.5)))

  let diag t x =
    Tensor.full [Tensor.size x 0] (Tensor.item t.variance)

  let get_params t =
    [t.lengthscales; t.variance]

  let set_params t params =
    {t with
     lengthscales = List.hd params;
     variance = List.nth params 1}

  let num_params t =
    Tensor.size t.lengthscales 0 + 1
end

(* Exponential Kernel *)
module ExponentialKernel : Kernel = struct
  type t = {
    lengthscales: Tensor.t;
    variance: Tensor.t;
    input_dim: int;
  }

  let create ?(ard=true) input_dim =
    let lengthscales = if ard then
      Tensor.ones [input_dim]
    else
      Tensor.ones [1] in
    let variance = Tensor.ones [1] in
    {lengthscales; variance; input_dim}

  let forward t x y =
    let x2 = Tensor.sum (Tensor.mul x x) ~dim:[1] ~keepdim:true in
    let y2 = Tensor.sum (Tensor.mul y y) ~dim:[1] ~keepdim:true in
    let xy = Tensor.mm x (Tensor.transpose y ~dim0:0 ~dim1:1) in
    let dist = Tensor.sqrt (Tensor.add (Tensor.add x2 
      (Tensor.transpose y2 ~dim0:0 ~dim1:1))
      (Tensor.mul_scalar xy (-2.))) in
    Tensor.mul t.variance (Tensor.exp (Tensor.neg dist))

  let diag t x =
    Tensor.full [Tensor.size x 0] (Tensor.item t.variance)

  let get_params t =
    [t.lengthscales; t.variance]

  let set_params t params =
    {t with
     lengthscales = List.hd params;
     variance = List.nth params 1}

  let num_params t =
    Tensor.size t.lengthscales 0 + 1
end

(* Likelihood module interface *)
module type Likelihood = sig
  type t
  val create : ?noise_var:float -> unit -> t
  val log_prob : t -> Types.data_batch -> Tensor.t -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t -> (Tensor.t * Tensor.t)
  val get_params : t -> Tensor.t list
end

(* Gaussian Likelihood *)
module GaussianLikelihood : Likelihood = struct
  type t = {
    noise: Tensor.t;  (* Output-specific noise variance *)
  }

  let create ?(noise_var=1.0) () =
    {noise = Tensor.full [1] noise_var}

  let log_prob t batch f =
    let diff = Tensor.sub batch.y f in
    let scaled_diff = Tensor.div diff (Tensor.sqrt t.noise) in
    let log_noise = Tensor.mul (Tensor.log t.noise) (Tensor.full_like t.noise 0.5) in
    let const_term = Tensor.full_like t.noise (-0.91893853) in (* -0.5 * log(2π) *)
    Tensor.sum (Tensor.add (Tensor.add (Tensor.neg (Tensor.mul scaled_diff scaled_diff))
                                     (Tensor.neg log_noise))
                         const_term)

  let predict t mean var =
    mean, Tensor.add var t.noise

  let get_params t =
    [t.noise]
end

(* Variational distribution module *)
module VariationalDist = struct
  type t = {
    mean: Tensor.t;
    scale_tril: Tensor.t;
    input_dim: int;
  }

  let create input_dim =
    {
      mean = Tensor.zeros [input_dim];
      scale_tril = Tensor.eye input_dim;
      input_dim;
    }

  let sample t ~num_samples =
    let epsilon = Tensor.randn [num_samples; t.input_dim] in
    let scaled = Tensor.mm epsilon t.scale_tril in
    Tensor.add scaled t.mean

  let kl_divergence t =
    let diag = Tensor.diagonal t.scale_tril 0 in
    let trace_term = Tensor.sum (Tensor.mul diag diag) in
    let logdet_term = Tensor.sum (Tensor.log diag) in
    let mahalanobis = Tensor.sum (Tensor.mul t.mean t.mean) in
    let dim_term = float t.input_dim in
    Tensor.add_scalar (Tensor.add (Tensor.sub trace_term (Tensor.mul_scalar logdet_term 2.))
                                mahalanobis)
                    (-. dim_term)

  let get_params t =
    [t.mean; t.scale_tril]

  let set_params t params =
    {t with
     mean = List.nth params 0;
     scale_tril = List.nth params 1}
end

(* Diagonalize transformations *)
module Diagonalize = struct
  type t = {
    mean: Tensor.t;
    scale: Tensor.t;
    condition_threshold: float;
  }

  let compute_diagonalize_transform data ?(condition_threshold=1e8) () =
    let mean = Tensor.mean data ~dim:0 in
    let centered = Tensor.sub data mean in
    let cov = Tensor.mm (Tensor.transpose centered ~dim0:0 ~dim1:1) centered in
    let eigendecomp = MatrixOps.safe_decompose cov in
    match eigendecomp with
    | Ok {eigvals; eigvecs} ->
        let scaled_eigvals = Tensor.where
          (Tensor.gt eigvals (Tensor.full_like eigvals 1e-8))
          (Tensor.sqrt eigvals)
          (Tensor.ones_like eigvals) in
        
        let condition = Tensor.div
          (Tensor.max scaled_eigvals)
          (Tensor.min scaled_eigvals) in
        
        if Tensor.item condition > condition_threshold then
          let reg = Tensor.mul_scalar 
            (Tensor.eye (Tensor.size scaled_eigvals 0)) 1e-6 in
          let eigvals_reg = Tensor.add eigvals reg in
          let scaled_eigvals = Tensor.sqrt eigvals_reg in
          let scale = Tensor.mm
            (Tensor.mm eigvecs 
                      (Tensor.diag scaled_eigvals))
            (Tensor.transpose eigvecs ~dim0:0 ~dim1:1) in
          Ok {mean; scale; condition_threshold}
        else
          let scale = Tensor.mm
            (Tensor.mm eigvecs 
                      (Tensor.diag scaled_eigvals))
            (Tensor.transpose eigvecs ~dim0:0 ~dim1:1) in
          Ok {mean; scale; condition_threshold}
    | Error msg -> Error msg

  let transform t data =
    let centered = Tensor.sub data t.mean in
    Tensor.mm centered t.scale

  let inverse_transform t data =
    let scaled = Tensor.mm data (Tensor.inverse t.scale) in
    Tensor.add scaled t.mean
end

(* ELBO *)
module ELBO = struct
  type t = {
    num_data: int;
    num_outputs: int;
    jitter: float;
  }

  let create ~num_data ~num_outputs ?(jitter=1e-6) () =
    {num_data; num_outputs; jitter}

  let compute_kl_gaussian mu_q sigma_q mu_p sigma_p =
    let sigma_p_inv = match MatrixOps.safe_inverse sigma_p with
      | Ok inv -> inv
      | Error msg -> failwith msg in
    let term1 = MatrixOps.trace_product sigma_p_inv sigma_q in
    let mu_diff = Tensor.sub mu_q mu_p in
    let term2 = Tensor.mm (Tensor.mm mu_diff sigma_p_inv) 
                         (Tensor.transpose mu_diff ~dim0:0 ~dim1:1) in
    let term3 = MatrixOps.efficient_trace 
      (Tensor.log (Tensor.div sigma_p sigma_q)) in
    let n = float (Tensor.size mu_q 0) in
    Tensor.add_scalar (Tensor.add (Tensor.add term1 term2) term3) (-. n)

  let compute_expected_ll t f_mean f_var y =
    let const = -0.5 *. log (2. *. Float.pi) in
    let var_term = Tensor.log f_var in
    let diff = Tensor.sub y f_mean in
    let quad_term = Tensor.div (Tensor.mul diff diff) f_var in
    Tensor.sum (Tensor.add_scalar (Tensor.sub (Tensor.neg var_term) quad_term) const)

  let compute_trace_terms t kuu kuf kfu kff =
    match MatrixOps.safe_inverse kuu with
    | Ok kuu_inv ->
        let term1 = MatrixOps.trace_product kuu_inv 
          (Tensor.mm kuf (Tensor.transpose kfu ~dim0:0 ~dim1:1)) in
        let term2 = MatrixOps.efficient_trace kff in
        Ok (Tensor.sub term2 term1)
    | Error msg -> Error msg

  let compute_full_elbo t kernel_expectations u_samples q_h batch =
    let ell = List.fold_left2 
      (fun acc (kuu, kuu_chol, kuf) q ->
        let alpha = match MatrixOps.safe_cholesky kuu with
          | Ok chol ->
              Tensor.trtrs u_samples chol 
                         ~upper:false ~transpose:false ~unitriangular:false
          | Error _ -> 
              Tensor.mm (MatrixOps.safe_inverse kuu |> Result.get_ok) u_samples in
        
        let mean = Tensor.mm kuf alpha in
        let k_diag = Tensor.diagonal kuu 0 in
        let var_term = Tensor.sum (Tensor.mul kuf kuf) ~dim:[1] in
        let var = Tensor.add (Tensor.sub k_diag var_term) 
                            (Tensor.full_like k_diag t.jitter) in
        
        let batch_mean = Tensor.gather mean ~dim:0 
                                     ~index:(Tensor.of_int1 batch.Types.output_idx) in
        let batch_var = Tensor.gather var ~dim:0 
                                    ~index:(Tensor.of_int1 batch.Types.output_idx) in
        
        let ell_q = compute_expected_ll t batch_mean batch_var batch.Types.y in
        Tensor.add acc ell_q
      ) (Tensor.zeros []) kernel_expectations (List.init t.num_outputs (fun i -> i)) in
    
    (* Compute KL divergences *)
    let kl_u = Variational_Dist.kl_divergence u_samples in
    let kl_h = List.fold_left (fun acc h ->
      Tensor.add acc (Variational_Dist.kl_divergence h)
    ) (Tensor.zeros []) q_h in
    
    let batch_scale = float t.num_outputs *. float (Tensor.size batch.Types.x 0) in
    Tensor.sub (Tensor.mul_scalar ell batch_scale) (Tensor.add kl_u kl_h)
end

(* Stochastic optimization *)
module StochasticOptimizer = struct
  type t = {
    params: Tensor.t list;
    momentum: Tensor.t list;
    velocity: Tensor.t list;
    beta1: float;
    beta2: float;
    epsilon: float;
    learning_rate: float;
  }

  let create ?(beta1=0.9) ?(beta2=0.999) ?(epsilon=1e-8) ~learning_rate params =
    let momentum = List.map (fun p -> Tensor.zeros_like p) params in
    let velocity = List.map (fun p -> Tensor.zeros_like p) params in
    {params; momentum; velocity; beta1; beta2; epsilon; learning_rate}

  let step t grads =
    List.map4 (fun p m v g ->
      (* Update momentum *)
      let m' = Tensor.(add (mul_scalar m t.beta1) 
                          (mul_scalar g (1. -. t.beta1))) in
      (* Update velocity *)
      let v' = Tensor.(add (mul_scalar v t.beta2) 
                          (mul_scalar (mul g g) (1. -. t.beta2))) in
      (* Compute bias-corrected updates *)
      let m_hat = Tensor.div m' (Tensor.full_like m' (1. -. t.beta1)) in
      let v_hat = Tensor.div v' (Tensor.full_like v' (1. -. t.beta2)) in
      (* Update parameters *)
      let update = Tensor.(div m_hat (add (sqrt v_hat) 
                                         (full_like v_hat t.epsilon))) in
      Tensor.sub p (Tensor.mul_scalar update t.learning_rate), m', v'
    ) t.params t.momentum t.velocity grads
    |> fun (p, m, v) -> {t with params = p; momentum = m; velocity = v}
end

(* Natural gradient optimizer *)
module NaturalGradient = struct
  type t = {
    learning_rate: float;
    momentum: float;
    damping: float;
    fisher_factors: (string * Tensor.t) list;
  }

  let create ?(learning_rate=0.1) ?(momentum=0.9) ?(damping=1e-4) () =
    {learning_rate; momentum; damping; fisher_factors = []}

  let compute_fisher_factor grads =
    List.map (fun grad ->
      let flat = Tensor.reshape grad [-1; 1] in
      Tensor.mm flat (Tensor.transpose flat ~dim0:0 ~dim1:1)
    ) grads

  let update_fisher_factors t grads =
    let new_factors = compute_fisher_factor grads in
    let updated = List.map2
      (fun (name, old_factor) new_factor ->
        let factor = Tensor.add
          (Tensor.mul_scalar old_factor t.momentum)
          (Tensor.mul_scalar new_factor (1. -. t.momentum)) in
        (name, factor))
      t.fisher_factors new_factors in
    {t with fisher_factors = updated}

  let compute_natural_gradient t grads =
    List.map2
      (fun (_, fisher) grad ->
        match MatrixOps.safe_inverse 
          (Tensor.add fisher 
            (Tensor.mul_scalar (Tensor.eye (Tensor.size fisher 0)) t.damping)) with
        | Ok inv_fisher -> 
            Tensor.mm inv_fisher grad
        | Error _ -> 
            grad)
      t.fisher_factors grads
end

(* Parameter constraints *)
module ParameterConstraints = struct
  type constraint_type =
    | Positive
    | Bounded of {lower: float; upper: float}
    | Probability
    | Custom of (float -> float)

  type t = {
    constraints: (string * constraint_type) list;
    transforms: (string * (float -> float)) list;
    inv_transforms: (string * (float -> float)) list;
  }

  let create constraints =
    let make_transform = function
      | Positive -> (exp, log)
      | Bounded {lower; upper} ->
          ((fun x -> lower +. (upper -. lower) *. (tanh x +. 1.) /. 2.),
           (fun y -> atanh (2. *. (y -. lower) /. (upper -. lower) -. 1.)))
      | Probability ->
          ((fun x -> 1. /. (1. +. exp (-.x))),
           (fun p -> -. log (1. /. p -. 1.)))
      | Custom f -> (f, fun x -> x) in
    
    let transforms, inv_transforms = List.split
      (List.map (fun (name, c) ->
        let f, g = make_transform c in
        ((name, f), (name, g))) constraints) in
    {constraints; transforms; inv_transforms}

  let apply_transform t param_name value =
    try
      let transform = List.assoc param_name t.transforms in
      transform value
    with Not_found -> value

  let apply_inv_transform t param_name value =
    try
      let inv_transform = List.assoc param_name t.inv_transforms in
      inv_transform value
    with Not_found -> value

  let check_constraints t params =
    List.for_all (fun (name, value) ->
      try
        let constraint_type = List.assoc name t.constraints in
        match constraint_type with
        | Positive -> value > 0.
        | Bounded {lower; upper} -> value >= lower && value <= upper
        | Probability -> value >= 0. && value <= 1.
        | Custom f -> try ignore (f value); true with _ -> false
      with Not_found -> true) params
end

(* Linear Model of Coregionalization (LMC) *)
module LMC = struct
  type t = {
    num_outputs: int;
    num_latent: int;
    input_dim: int;
    kernels: SE_ARD_Kernel.t list;
    weights: Tensor.t;  (* D x Q matrix *)
  }

  let create ~num_outputs ~num_latent ~input_dim = 
    let kernels = List.init num_latent 
      (fun _ -> SE_ARD_Kernel.create ~ard:true input_dim) in
    let weights = Tensor.randn [num_outputs; num_latent] in
    {num_outputs; num_latent; input_dim; kernels; weights}

  let coregionalization t =
    let weights_t = Tensor.transpose t.weights ~dim0:0 ~dim1:1 in
    Tensor.mm t.weights weights_t

  let compute_kernel t x1 x2 =
    let base_kernels = List.map (fun k -> SE_ARD_Kernel.forward k x1 x2) t.kernels in
    let base_k = Tensor.stack base_kernels ~dim:0 in
    let coreg = coregionalization t in
    MatrixOps.kron coreg base_k

  let predict t x_test =
    let k_test = compute_kernel t x_test x_test in
    let mean = Tensor.zeros [Tensor.size x_test 0; t.num_outputs] in
    mean, k_test
end

(* Latent Variable MOGP (LV-MOGP) *)
module LV_MOGP = struct
  type t = {
    num_outputs: int;
    latent_dim: int;
    input_kernel: SE_ARD_Kernel.t;
    latent_kernel: SE_ARD_Kernel.t;
    inducing_points: Tensor.t;
    q_mu: Tensor.t;
    q_sqrt: Tensor.t;
    latent_means: Tensor.t list;
    latent_vars: Tensor.t list;
  }

  let create ~num_outputs ~latent_dim ~input_dim ~num_inducing =
    let input_kernel = SE_ARD_Kernel.create ~ard:true input_dim in
    let latent_kernel = SE_ARD_Kernel.create ~ard:true latent_dim in
    let inducing_points = Tensor.randn [num_inducing; input_dim] in
    let q_mu = Tensor.zeros [num_inducing; num_outputs] in
    let q_sqrt = Tensor.eye num_inducing in
    let latent_means = List.init num_outputs 
      (fun _ -> Tensor.zeros [latent_dim]) in
    let latent_vars = List.init num_outputs 
      (fun _ -> Tensor.ones [latent_dim]) in
    {
      num_outputs;
      latent_dim;
      input_kernel;
      latent_kernel;
      inducing_points;
      q_mu;
      q_sqrt;
      latent_means;
      latent_vars;
    }

  let compute_kuu t =
    SE_ARD_Kernel.forward t.input_kernel t.inducing_points t.inducing_points

  let compute_kuf t x =
    SE_ARD_Kernel.forward t.input_kernel t.inducing_points x

  let compute_latent_dist t =
    let means = Tensor.stack t.latent_means ~dim:0 in
    let vars = Tensor.stack t.latent_vars ~dim:0 in
    means, vars

  let compute_latent_cov t =
    let means, vars = compute_latent_dist t in
    let k = SE_ARD_Kernel.forward t.latent_kernel means means in
    k, means, vars

  let compute_elbo t batch =
    let kuu = compute_kuu t in
    let kuf = compute_kuf t batch.Types.x in
    let k_latent, means, vars = compute_latent_cov t in
    
    (* Compute expected log likelihood *)
    let alpha = match MatrixOps.safe_cholesky kuu with
      | Ok chol -> 
          Tensor.trtrs t.q_mu chol ~upper:false ~transpose:false ~unitriangular:false
      | Error _ ->
          Tensor.mm (MatrixOps.safe_inverse kuu |> Result.get_ok) t.q_mu in
    
    let mean = Tensor.mm kuf (Tensor.transpose alpha ~dim0:0 ~dim1:1) in
    let k_diag = SE_ARD_Kernel.diag t.input_kernel batch.Types.x in
    let var_term = Tensor.sum (Tensor.mul kuf kuf) ~dim:[1] in
    let var = Tensor.add (Tensor.sub k_diag var_term) 
                        (Tensor.full_like k_diag 1e-6) in

    (* KL divergences *)
    let kl_u = Variational_Dist.kl_divergence 
      {mean = t.q_mu; scale_tril = t.q_sqrt; input_dim = t.num_outputs} in
    
    let kl_h = List.fold_left2 (fun acc mean var ->
      let prior_mean = Tensor.zeros_like mean in
      let prior_var = Tensor.ones_like var in
      acc +. Tensor.item (ELBO.compute_kl_gaussian mean var prior_mean prior_var)
    ) 0. t.latent_means t.latent_vars in

    (* Complete ELBO *)
    let batch_size = float (Tensor.size batch.Types.x 0) in
    let ll = ELBO.compute_expected_ll
      {num_data = Tensor.size batch.Types.x 0;
       num_outputs = t.num_outputs;
       jitter = 1e-6}
      mean var batch.Types.y in
    
    Tensor.sub (Tensor.mul_scalar ll batch_size)
               (Tensor.add kl_u (Tensor.full [1] kl_h))

  let predict t x =
    let kuu = compute_kuu t in
    let kuf = compute_kuf t x in
    let k_latent, means, vars = compute_latent_cov t in
    
    (* Compute predictive mean *)
    let alpha = match MatrixOps.safe_cholesky kuu with
      | Ok chol ->
          Tensor.trtrs t.q_mu chol ~upper:false ~transpose:false ~unitriangular:false
      | Error _ ->
          Tensor.mm (MatrixOps.safe_inverse kuu |> Result.get_ok) t.q_mu in
    
    let mean = Tensor.mm kuf (Tensor.transpose alpha ~dim0:0 ~dim1:1) in
    
    (* Compute predictive variance *)
    let k_diag = SE_ARD_Kernel.diag t.input_kernel x in
    let var_term = Tensor.sum (Tensor.mul kuf kuf) ~dim:[1] in
    let var = Tensor.add (Tensor.sub k_diag var_term)
                        (Tensor.full_like k_diag 1e-6) in
    
    mean, var
end

(* Generalized Scalable LV-MOGP (GS-LVMOGP) *)
module GS_LVMOGP = struct
  type t = {
    num_outputs: int;
    num_latents: int;
    latent_dim: int;
    input_dim: int;
    num_inducing_x: int;
    num_inducing_h: int;
    input_kernels: SE_ARD_Kernel.t list;
    latent_kernels: SE_ARD_Kernel.t list;
    inducing_x: Tensor.t;
    inducing_h: Tensor.t list;
    q_u: Variational_Dist.t;
    q_h: Variational_Dist.t array array;  (* D x Q array *)
    likelihood: Gaussian_Likelihood.t;
    config: Types.training_config;
    optimizer: StochasticOptimizer.t;
  }

  let create ~num_outputs ~num_latents ~latent_dim ~input_dim 
            ~num_inducing_x ~num_inducing_h ?(config=Types.{
              num_epochs = 100;
              batch_size = 100;
              learning_rate = 1e-3;
              num_samples = 1;
              num_data = 0;
              num_outputs = 0;
              jitter = 1e-6;
              whiten = true;
            }) () =
    let input_kernels = List.init num_latents 
      (fun _ -> SE_ARD_Kernel.create input_dim) in
    let latent_kernels = List.init num_latents 
      (fun _ -> SE_ARD_Kernel.create latent_dim) in
    let inducing_x = Tensor.(mul_scalar (randn [num_inducing_x; input_dim]) 0.1) in
    let inducing_h = List.init num_latents 
      (fun _ -> Tensor.(mul_scalar (randn [num_inducing_h; latent_dim]) 0.1)) in
    
    (* Initialize variational distributions *)
    let kuu = List.map2 
      (fun k_x h_x -> SE_ARD_Kernel.compute_k_symm k_x h_x)
      input_kernels inducing_h in
    let l = if config.whiten then
      List.map MatrixOps.safe_cholesky kuu
      |> List.map Result.get_ok
    else
      List.map (fun _ -> Tensor.eye num_inducing_x) kuu in
    
    let q_u = Variational_Dist.create (num_inducing_x * num_inducing_h) in
    let q_h = Array.make_matrix num_outputs num_latents 
      (Variational_Dist.create latent_dim) in
    let likelihood = Gaussian_Likelihood.create () in
    
    let params = [q_u] @ 
                (Array.to_list (Array.map Array.to_list q_h) |> List.concat) in
    let optimizer = StochasticOptimizer.create 
      ~learning_rate:config.learning_rate params in

    {
      num_outputs;
      num_latents;
      latent_dim;
      input_dim;
      num_inducing_x;
      num_inducing_h;
      input_kernels;
      latent_kernels;
      inducing_x;
      inducing_h;
      q_u;
      q_h;
      likelihood;
      config;
      optimizer;
    }

  let compute_kernel_expectations t batch =
    let latent_samples = Array.init t.num_outputs (fun d ->
      Array.init t.num_latents (fun q ->
        Variational_Dist.sample t.q_h.(d).(q) ~num_samples:t.config.num_samples
      )) in
    
    List.init t.num_latents (fun q ->
      let kuu = SE_ARD_Kernel.compute_k_symm 
        (List.nth t.input_kernels q) t.inducing_x in
      let kuu_chol = if t.config.whiten then
        MatrixOps.safe_cholesky ~jitter:t.config.jitter kuu |> Result.get_ok
      else
        Tensor.eye (Tensor.size kuu 0) in
      
      let h_batch = Tensor.stack 
        (Array.to_list (Array.map (fun ls -> Array.get ls q) latent_samples))
        ~dim:0 in
      let kuf = SE_ARD_Kernel.forward 
        (List.nth t.input_kernels q) t.inducing_x batch.Types.x in
      kuu, kuu_chol, kuf, h_batch
    )

  let compute_elbo t batch =
    let kernel_expectations = compute_kernel_expectations t batch in
    let u_samples = Variational_Dist.sample t.q_u 
      ~num_samples:t.config.num_samples in
    
    (* Compute expected log likelihood *)
    let ell = List.fold_left2 
      (fun acc (kuu, kuu_chol, kuf, h_batch) q ->
        let alpha = if t.config.whiten then
          Tensor.trtrs u_samples kuu_chol 
            ~upper:false ~transpose:false ~unitriangular:false
        else
          Tensor.mm (MatrixOps.safe_inverse kuu |> Result.get_ok) u_samples in
        
        let mean = Tensor.mm kuf alpha in
        let k_diag = SE_ARD_Kernel.diag (List.nth t.input_kernels q) batch.Types.x in
        let var_term = MatrixOps.trace_product 
          (Tensor.mm kuf (MatrixOps.safe_inverse kuu |> Result.get_ok))
          kuf in
        let var = Tensor.add k_diag (Tensor.full_like k_diag t.config.jitter) in
        
        let ell_q = Gaussian_Likelihood.log_prob t.likelihood batch mean in
        Tensor.add acc ell_q
      ) (Tensor.zeros []) kernel_expectations (List.init t.num_latents (fun i -> i)) in
    
    (* Compute KL divergences *)
    let kl_u = Variational_Dist.kl_divergence t.q_u in
    let kl_h = Array.fold_left (fun acc q_h_row ->
      Array.fold_left (fun acc q_h ->
        Tensor.add acc (Variational_Dist.kl_divergence q_h)
      ) acc q_h_row
    ) (Tensor.zeros []) t.q_h in
    
    (* Scale and combine terms *)
    let batch_scale = float t.num_outputs *. float (Tensor.size batch.Types.x 0) in
    Tensor.sub (Tensor.mul_scalar ell batch_scale) (Tensor.add kl_u kl_h)

  let train t x y =
    let num_batches = (Tensor.size x 0 + t.config.batch_size - 1) / 
                     t.config.batch_size in
    
    for epoch = 1 to t.config.num_epochs do
      let perm = Tensor.randperm (Tensor.size x 0) in
      
      for batch = 0 to num_batches - 1 do
        let start_idx = batch * t.config.batch_size in
        let batch_size = min t.config.batch_size 
                            (Tensor.size x 0 - start_idx) in
        let batch_indices = Tensor.narrow perm ~dim:0 
                                        ~start:start_idx 
                                        ~length:batch_size in
        let batch = {
          Types.x = Tensor.index_select x ~dim:0 ~index:batch_indices;
          y = Tensor.index_select y ~dim:0 ~index:batch_indices;
          output_idx = Array.init batch_size (fun _ -> 
            Random.int t.num_outputs);
          input_idx = Array.init batch_size (fun i -> i);
        } in
        
        (* Compute divergence and gradients *)
        let divergence = Tensor.neg (compute_elbo t batch) in
        let grads = Tensor.backward divergence in
        
        (* Update parameters using optimizer *)
        t.optimizer <- StochasticOptimizer.step t.optimizer grads
      done;

      if epoch mod 10 = 0 then
        Printf.printf "Epoch %d: Loss = %f\n" epoch (Tensor.item divergence);
    done

  let predict t x =
    let num_test = Tensor.size x 0 in
    let means = Array.make t.num_outputs (Tensor.zeros [num_test]) in
    let vars = Array.make t.num_outputs (Tensor.zeros [num_test]) in
    
    Array.iteri (fun d _ ->
      let batch = {
        Types.x;
        y = Tensor.zeros [num_test];
        output_idx = [|d|];
        input_idx = Array.init num_test (fun i -> i);
      } in
      
      let kernel_expectations = compute_kernel_expectations t batch in
      let u_samples = Variational_Dist.sample t.q_u 
        ~num_samples:t.config.num_samples in
      
      List.iteri (fun q (kuu, kuu_chol, kuf, _) ->
        let alpha = if t.config.whiten then
          Tensor.trtrs u_samples kuu_chol
            ~upper:false ~transpose:false ~unitriangular:false
        else
          Tensor.mm (MatrixOps.safe_inverse kuu |> Result.get_ok) u_samples in
        
        let mean = Tensor.mm kuf alpha in
        let k_diag = SE_ARD_Kernel.diag (List.nth t.input_kernels q) x in
        let var_term = MatrixOps.trace_product 
          (Tensor.mm kuf (MatrixOps.safe_inverse kuu |> Result.get_ok))
          kuf in
        let var = Tensor.add (Tensor.sub k_diag var_term)
                            (Tensor.full_like k_diag t.config.jitter) in
        
        means.(d) <- Tensor.add means.(d) (Tensor.mean mean ~dim:[0]);
        vars.(d) <- Tensor.add vars.(d) var;
      ) kernel_expectations;
    ) means;
    
    Array.to_list means, Array.to_list vars

  let sample_predictive t x num_samples =
    let means, vars = predict t x in
    let samples = List.map2 (fun mean var ->
      let noise = Tensor.randn [num_samples; Tensor.size mean 0] in
      let std = Tensor.sqrt var in
      Tensor.add mean (Tensor.mul noise std)
    ) means vars in
    samples

  let get_parameters t =
    let q_u_params = Variational_Dist.get_params t.q_u in
    let q_h_params = Array.to_list (Array.map (fun row ->
      Array.to_list (Array.map Variational_Dist.get_params row)
    ) t.q_h) |> List.concat |> List.concat in
    let kernel_params = List.concat [
      List.concat_map SE_ARD_Kernel.get_params t.input_kernels;
      List.concat_map SE_ARD_Kernel.get_params t.latent_kernels;
    ] in
    q_u_params @ q_h_params @ kernel_params

  let set_parameters t params =
    (* Helper to split list at given index *)
    let split_at n lst =
      let rec aux n acc = function
        | [] -> List.rev acc, []
        | h :: t as l -> if n = 0 then List.rev acc, l
                        else aux (n-1) (h :: acc) t in
      aux n [] lst in
    
    (* Update q_u parameters *)
    let q_u_param_count = 2 in (* mean and scale_tril *)
    let q_u_params, rest = split_at q_u_param_count params in
    let q_u' = Variational_Dist.set_params t.q_u q_u_params in
    
    (* Update q_h parameters *)
    let q_h_param_count = 2 * t.num_outputs * t.num_latents in
    let q_h_params, rest = split_at q_h_param_count rest in
    let q_h' = Array.mapi (fun i row ->
      Array.mapi (fun j q_h ->
        let idx = (i * t.num_latents + j) * 2 in
        let params = List.sub q_h_params idx 2 in
        Variational_Dist.set_params q_h params
      ) row
    ) t.q_h in
    
    (* Update kernel parameters *)
    let input_kernels' = List.mapi (fun i k ->
      let param_count = SE_ARD_Kernel.num_params k in
      let params, r = split_at param_count rest in
      SE_ARD_Kernel.set_params k params
    ) t.input_kernels in
    
    let latent_kernels' = List.mapi (fun i k ->
      let param_count = SE_ARD_Kernel.num_params k in
      let params, r = split_at param_count rest in
      SE_ARD_Kernel.set_params k params
    ) t.latent_kernels in
    
    {t with 
     q_u = q_u';
     q_h = q_h';
     input_kernels = input_kernels';
     latent_kernels = latent_kernels'}
end

(* Utility functions for model usage *)
module ModelUtils = struct
  (* Data preprocessing *)
  let preprocess_data x y =
    let data_handler = Data_Handler.create x y in
    Data_Handler.transform data_handler x y

  (* Model selection using cross-validation *)
  let cross_validate model x y num_folds =
    let fold_indices = Validation.k_fold_split y num_folds in
    List.map (fun (train_idx, val_idx) ->
      let x_train = Tensor.index_select x ~dim:0 ~index:train_idx in
      let y_train = Tensor.index_select y ~dim:0 ~index:train_idx in
      let x_val = Tensor.index_select x ~dim:0 ~index:val_idx in
      let y_val = Tensor.index_select y ~dim:0 ~index:val_idx in
      
      GS_LVMOGP.train model x_train y_train;
      let predictions = GS_LVMOGP.predict model x_val in
      let mse = Validation.compute_metric Validation.MSE 
        (fst predictions) y_val None in
      Tensor.item mse
    ) fold_indices

  (* Hyperparameter optimization *)
  let optimize_hyperparams model x y =
    let param_bounds = [
      ("learning_rate", {
        Hyperparameter_Optimizer.lower = 1e-4;
        upper = 1e-1;
        log_scale = true;
      });
      ("num_latents", {
        lower = 1.;
        upper = 10.;
        log_scale = false;
      });
    ] in
    
    let optimizer = Hyperparameter_Optimizer.create 
      (Array.of_list param_bounds)
      (Random_Search {num_trials = 20}) in
    
    let objective params =
      let config = {model.GS_LVMOGP.config with
        learning_rate = List.assoc "learning_rate" params;
        num_outputs = int_of_float (List.assoc "num_latents" params);
      } in
      let model' = {model with config} in
      let errors = cross_validate model' x y 5 in
      List.fold_left (+.) 0. errors /. float (List.length errors)
    in
    
    Hyperparameter_Optimizer.optimize optimizer objective

  (* Performance monitoring *)
  let monitor_performance model x y =
    let monitor = Performance_Monitor.create () in
    let compute_metrics preds targets =
      [
        ("mse", Validation.compute_metric Validation.MSE preds targets None);
        ("mae", Validation.compute_metric Validation.MAE preds targets None);
        ("nlpd", Validation.compute_metric Validation.NLPD preds targets None);
      ]
    in
    
    let start_time = Unix.gettimeofday () in
    GS_LVMOGP.train model x y;
    let train_time = Unix.gettimeofday () -. start_time in
    
    let predictions = GS_LVMOGP.predict model x in
    let metrics = compute_metrics (fst predictions) y in
    let memory = Gc.stat () in
    
    Performance_Monitor.update_metrics monitor
      ~divergence:0.0  
      ~metrics:[metrics]
      ~time:train_time
      ~memory:memory.heap_words

  (* Save and load models *)
  let save_model model filename =
    let oc = open_out filename in
    Marshal.to_channel oc model [];
    close_out oc

  let load_model filename =
    let ic = open_in filename in
    let model = Marshal.from_channel ic in
    close_in ic;
    model
end
open Torch

type divergence_config = {
    mu: float;          (* Smoothness parameter *)
    beta1: float;       (* First regularization parameter *)
    beta2: float;       (* Second regularization parameter *)
}

type model_config = {
    input_dim: int;     (* Input dimension *)
    rank: int;          (* Rank of factorization *)
    max_order: int;     (* Maximum order of feature interactions *)
    learning_rate: float;
    divergence: divergence_config;
}

let epsilon = 1e-7

let stable_log x = 
    if x < epsilon then log epsilon
    else log x

let stable_exp x =
    if x > 88.0 then exp 88.0
    else if x < -88.0 then exp (-88.0)
    else exp x

let clip gradient clip_value =
    Tensor.clamp gradient ~min:(-.clip_value) ~max:clip_value

let stable_sum values =
    let n = Array.length values in
    if n = 0 then 0.0
    else begin
      let max_val = Array.fold_left max values.(0) values in
      let sum_exp = Array.fold_left (fun acc x ->
        acc +. exp (x -. max_val)
      ) 0.0 values in
      max_val +. log (sum_exp)
end

(* Divergence functions *)
module Divergence = struct
  type t =
    | SquaredDivergence
    | LogisticDivergence
    | HingeDivergence

  let compute divergence_type pred target =
    match divergence_type with
    | SquaredDivergence -> 
        let diff = pred -. target in
        diff *. diff *. 0.5
    | LogisticDivergence ->
        let z = -.target *. pred in
        log1p (exp z)
    | HingeDivergence ->
        max 0.0 (1.0 -. target *. pred)

  let gradient divergence_type pred target =
    match divergence_type with
    | SquaredDivergence -> pred -. target
    | LogisticDivergence ->
        let z = -.target *. pred in
        -.target *. (exp z /. (1.0 +. exp z))
    | HingeDivergence ->
        if target *. pred < 1.0 then -.target else 0.0
end

(* Parameter initialization utilities *)
module Init = struct
  type method_t =
    | Xavier of float  (* gain *)
    | HeNormal of float  (* gain *)
    | HeUniform of float  (* gain *)
    | LeCunNormal of float  (* gain *)
    | LeCunUniform of float  (* gain *)
    | Orthogonal of float   (* gain *)
    | Zeros
    | Constant of float

  let xavier_bound n_in n_out gain =
    gain *. sqrt (6.0 /. float (n_in + n_out))

  let he_bound n_in gain =
    gain *. sqrt (2.0 /. float n_in)

  let lecun_bound n_in gain =
    gain *. sqrt (1.0 /. float n_in)

  (* Generate orthogonal matrix using QR decomposition *)
  let orthogonal_matrix rows cols =
    let device = Device.Gpu in
    let mat = Tensor.randn [rows; cols] ~device:device in
    let q, r = Tensor.qr mat in
    let d = Tensor.sign (Tensor.diag r) in
    Tensor.mm q (Tensor.diag d)

  let init method_t shape =
    let n_in = List.hd shape in
    let n_out = if List.length shape > 1 then List.nth shape 1 else n_in in
    let device = Device.Gpu in
    
    match method_t with
    | Xavier gain ->
        let bound = xavier_bound n_in n_out gain in
        Tensor.(sub (mul_scalar (rand shape) (2.0 *. bound)) 
                   (float_vec [bound]))
    | HeNormal gain ->
        let std = he_bound n_in gain in
        Tensor.(mul_scalar (randn shape) std)
    | HeUniform gain ->
        let bound = he_bound n_in gain *. sqrt 3.0 in
        Tensor.(sub (mul_scalar (rand shape) (2.0 *. bound))
                   (float_vec [bound]))
    | LeCunNormal gain ->
        let std = lecun_bound n_in gain in
        Tensor.(mul_scalar (randn shape) std)
    | LeCunUniform gain ->
        let bound = lecun_bound n_in gain *. sqrt 3.0 in
        Tensor.(sub (mul_scalar (rand shape) (2.0 *. bound))
                   (float_vec [bound]))
    | Orthogonal gain ->
        if List.length shape <> 2 then
          failwith "Orthogonal initialization requires 2D shape"
        else
          let rows, cols = List.hd shape, List.nth shape 1 in
          let mat = orthogonal_matrix rows cols in
          Tensor.(mul_scalar mat gain)
    | Zeros -> Tensor.zeros shape ~device:device
    | Constant c -> Tensor.full shape c ~device:device
end

(* Dynamic programming for ANOVA kernel computations *)
module DynamicProgramming = struct
  let compute_anova_table p x m =
    let d = Tensor.shape p |> List.hd in
    let device = Device.device p in
    
    (* Initialize DP table with proper dimensions *)
    let table = Tensor.zeros [d + 1; m + 1] ~device:device in
    
    (* Base case initialization *)
    Tensor.fill_float (Tensor.slice table [0; 0] [d+1; 1]) 1.0;
    
    (* Compute table entries with numerical stability *)
    for t = 1 to m do
      for j = t to d do
        let prev = Tensor.get table [j-1; t] in
        let p_j = Tensor.get p [j-1] in
        let x_j = Tensor.get x [j-1] in
        let prod = p_j *. x_j *. (Tensor.get table [j-1; t-1]) in
        
        (* Stable accumulation *)
        let value = if abs_float prod < Numerical.epsilon then prev
                   else prev +. prod in
        Tensor.set table [j; t] value
      done
    done;
    
    table

  let compute_gradient_table table p x m =
    let d = Tensor.shape p |> List.hd in
    let device = Device.device p in
    
    (* Initialize gradient table *)
    let grad_table = Tensor.zeros [d + 1; m + 1] ~device:device in
    
    (* Set initial gradient *)
    Tensor.set grad_table [d; m] 1.0;
    
    (* Backward pass with numerical stability *)
    for t = m downto 1 do
      for j = d-1 downto t-1 do
        let grad_curr = Tensor.get grad_table [j+1; t] in
        let grad_next = Tensor.get grad_table [j+1; t+1] in
        let x_j = Tensor.get x [j] in
        let p_j = Tensor.get p [j] in
        
        let term1 = grad_curr in
        let term2 = if abs_float (p_j *. x_j) < Numerical.epsilon then 0.0
                   else grad_next *. p_j *. x_j in
        Tensor.set grad_table [j; t] (term1 +. term2)
      done
    done;
    
    grad_table
end

(* ANOVA kernel *)
module Anova = struct
  type t = {
    p: Tensor.t;             (* Parameter vector *)
    order: int;              (* Kernel order *)
    cache: (int, float array) Hashtbl.t;  (* Cache for intermediate computations *)
  }

  let create config =
    {
      p = Init.init (Init.Xavier 1.0) [config.input_dim];
      order = config.max_order;
      cache = Hashtbl.create 256;
    }

  (* Compute ANOVA kernel value efficiently *)
  let eval_kernel t x =
    let d = Tensor.shape t.p |> List.hd in
    let device = Device.device t.p in
    
    (* Check cache first *)
    let x_hash = Hashtbl.hash (Tensor.to_float_list x) in
    match Hashtbl.find_opt t.cache x_hash with
    | Some cached_result -> Array.get cached_result (t.order - 1)
    | None ->
        let table = DynamicProgramming.compute_anova_table t.p x t.order in
        let results = Array.init t.order (fun m ->
          Tensor.get table [d; m+1]
        ) in
        Hashtbl.add t.cache x_hash results;
        Array.get results (t.order - 1)

  (* Compute gradient efficiently *)
  let grad_kernel t x =
    let d = Tensor.shape t.p |> List.hd in
    let device = Device.device t.p in
    
    (* Compute forward and backward tables *)
    let fwd_table = DynamicProgramming.compute_anova_table t.p x t.order in
    let grad_table = DynamicProgramming.compute_gradient_table fwd_table t.p x t.order in
    
    (* Compute final gradients *)
    let grad = Tensor.zeros [d] ~device:device in
    for j = 0 to d - 1 do
      let mut_sum = ref 0.0 in
      for m = 1 to t.order do
        let a_grad = Tensor.get grad_table [j; m] in
        let x_j = Tensor.get x [j] in
        let a_prev = Tensor.get fwd_table [j-1; m-1] in
        mut_sum := !mut_sum +. (a_grad *. a_prev *. x_j)
      done;
      Tensor.set grad [j] !mut_sum
    done;
    grad
end

(* Higher order factorization machine *)
module HOFM = struct
  type t = {
    w: Tensor.t;                 (* Linear weights *)
    p: Tensor.t array;           (* Factor matrices for each order *)
    config: Types.model_config;
  }

  let create config =
    let device = Device.Gpu in
    {
      w = Init.init Init.Zeros [config.input_dim];
      p = Array.init (config.max_order - 1) (fun _ -> 
        Init.init (Init.Xavier 1.0) [config.input_dim; config.rank]
      );
      config;
    }

  (* Efficient prediction implementation *)
  let predict t x =
    let device = Device.device t.w in
    
    (* Linear term *)
    let linear = Tensor.(dot t.w x) in
    
    (* Higher-order terms with memory efficiency *)
    let order_terms = Array.mapi (fun m p_m ->
      let order = m + 2 in
      let k = t.config.rank in
      let mut_sum = ref 0.0 in
      
      (* Process in blocks for cache efficiency *)
      let block_size = 32 in
      for s = 0 to k - 1 step block_size do
        let end_s = min k (s + block_size) in
        let block_sum = ref 0.0 in
        
        for s' = s to end_s - 1 do
          let p_s = Tensor.select p_m 1 s' in
          block_sum := !block_sum +. Anova.eval_kernel 
            { p = p_s; order; cache = Hashtbl.create 16 } x
        done;
        
        mut_sum := !mut_sum +. !block_sum
      done;
      
      !mut_sum
    ) t.p in
    
    linear +. Array.fold_left (+.) 0.0 order_terms

  (* Memory-efficient gradient computation *)
  let grad_kernel t p x order =
    let d = Tensor.shape p |> List.hd in
    let device = Device.device p in
    
    let anova = { Anova.p = p; order; cache = Hashtbl.create 16 } in
    Anova.grad_kernel anova x
end

(* Parameter sharing implementation *)
module SharedParameters = struct
  type t = {
    w: Tensor.t;                (* Linear weights *)
    shared_factors: Tensor.t;    (* Shared factor matrix *)
    degree_weights: Tensor.t;    (* Weights for each degree *)
    dummy_features: Tensor.t;    (* Dummy features for parameter sharing *)
    config: Types.model_config;
    mutable feature_map: Tensor.t option;  (* Cached feature mapping *)
  }

  let create config =
    let device = Device.Gpu in
    {
      w = Init.init Init.Zeros [config.input_dim];
      shared_factors = Init.init (Init.Xavier 1.0) [config.input_dim; config.rank];
      degree_weights = Tensor.ones [config.max_order] ~device:device;
      dummy_features = Init.init (Init.Normal 0.01) [config.max_order - 1];
      config;
      feature_map = None;
    }

  (* Efficient feature mapping computation *)
  let compute_feature_mapping t x =
    let device = Device.device t.shared_factors in
    let x_aug = Tensor.cat [t.dummy_features; x] ~dim:0 in
    
    match t.feature_map with
    | Some cached when Tensor.shape cached = Tensor.shape x -> cached
    | _ ->
        let d, r = Tensor.shape2_exn t.shared_factors in
        let mapping = Tensor.zeros [d; r] ~device:device in
        
        (* Compute mapping in blocks for cache efficiency *)
        let block_size = 32 in
        for i = 0 to d - 1 step block_size do
          let end_i = min d (i + block_size) in
          let factors_block = Tensor.slice t.shared_factors [i; 0] [end_i - i; r] in
          let map_block = Tensor.(mm (unsqueeze x_aug 0) factors_block) in
          Tensor.copy_ (Tensor.slice mapping [i; 0] [end_i - i; r]) map_block
        done;
        
        t.feature_map <- Some mapping;
        mapping

  (* Efficient prediction with shared parameters *)
  let predict t x =
    let mapping = compute_feature_mapping t x in
    let device = Device.device mapping in
    
    (* Linear term *)
    let linear = Tensor.(dot t.w x) in
    
    (* Higher-order terms *)
    let terms = Array.init (Tensor.shape t.degree_weights).(0) (fun m ->
      let degree = m + 1 in
      let mut_sum = ref 0.0 in
      
      (* Process in blocks *)
      let block_size = 32 in
      for i = 0 to (Tensor.shape mapping).(0) - 1 step block_size do
        let end_i = min ((Tensor.shape mapping).(0)) (i + block_size) in
        let block = Tensor.slice mapping [i; 0] [end_i - i; -1] in
        
        (* Compute degree-specific interactions *)
        let block_sum = Tensor.(sum (pow block (float degree))) in
        mut_sum := !mut_sum +. (Tensor.get t.degree_weights [m] *. 
                               Tensor.to_float0_exn block_sum)
      done;
      
      !mut_sum
    ) in
    
    linear +. Array.fold_left (+.) 0.0 terms

  (* Memory-efficient gradient computation *)
  let compute_gradient t x pred target =
    let mapping = compute_feature_mapping t x in
    let d, r = Tensor.shape2_exn t.shared_factors in
    let device = Device.device mapping in
    
    (* Initialize gradients *)
    let w_grad = Tensor.(mul_scalar x (pred -. target)) in
    let factor_grads = Tensor.zeros [d; r] ~device:device in
    let weight_grads = Tensor.zeros_like t.degree_weights in
    
    (* Compute gradients in blocks *)
    let block_size = 32 in
    for i = 0 to d - 1 step block_size do
      let end_i = min d (i + block_size) in
      
      for m = 0 to (Tensor.shape t.degree_weights).(0) - 1 do
        let degree = m + 1 in
        let block = Tensor.slice mapping [i; 0] [end_i - i; r] in
        
        (* Compute degree-specific gradients *)
        let power_term = Tensor.(pow block (float (degree - 1))) in
        let grad_term = Tensor.(
          mul_scalar block (
            (pred -. target) *. 
            (Tensor.get t.degree_weights [m]) *. 
            (float degree)
          )
        ) in
        
        Tensor.(add_ 
          (slice factor_grads [i; 0] [end_i - i; r])
          (mul power_term grad_term)
        );
        
        (* Update degree weight gradients *)
        let weight_grad = Tensor.(
          sum (pow block (float degree))
        ) |> Tensor.to_float0_exn in
        
        Tensor.set weight_grads [m] (
          Tensor.get weight_grads [m] +. 
          weight_grad *. (pred -. target)
        )
      done
    done;
    
    (w_grad, factor_grads, weight_grads)
end

(* Optimizer *)
module Optimizer = struct
  type t =
    | SGD of float  (* learning rate *)
    | Adam of {
        alpha: float;
        beta1: float;
        beta2: float;
        epsilon: float;
        mutable m: Tensor.t;  (* First moment *)
        mutable v: Tensor.t;  (* Second moment *)
        mutable t: int;       (* Timestep *)
      }

  let create = function
    | `SGD lr -> SGD lr
    | `Adam (alpha, beta1, beta2, epsilon) ->
        Adam {
          alpha; beta1; beta2; epsilon;
          m = Tensor.zeros [];
          v = Tensor.zeros [];
          t = 0;
        }

  (* Update parameters with proper numerical stability *)
  let step opt param grad =
    match opt with
    | SGD lr ->
        let grad' = Numerical.clip grad 1.0 in
        Tensor.(sub param (mul_scalar grad' lr))
        
    | Adam ({alpha; beta1; beta2; epsilon; _} as state) ->
        state.t <- state.t + 1;
        
        (* Update biased first moment *)
        state.m <- Tensor.(
          add (mul_scalar state.m beta1)
              (mul_scalar grad (1.0 -. beta1))
        );
        
        (* Update biased second moment *)
        state.v <- Tensor.(
          add (mul_scalar state.v beta2)
              (mul_scalar (pow grad 2.0) (1.0 -. beta2))
        );
        
        (* Compute bias-corrected moments *)
        let m_hat = Tensor.(
          div_scalar state.m (1.0 -. (beta1 ** float state.t))
        ) in
        let v_hat = Tensor.(
          div_scalar state.v (1.0 -. (beta2 ** float state.t))
        ) in
        
        (* Update parameters *)
        Tensor.(
          sub param (
            mul_scalar (div m_hat (sqrt v_hat +. epsilon)) alpha
          )
        )
end
open Torch

type kernel = Tensor.t -> Tensor.t -> Tensor.t

type density = {
log_prob: Tensor.t -> float;
grad_log_prob: Tensor.t -> Tensor.t;
}

type config = {
length_scale: float;
nugget: float;
}

(* Core PDE operators *)
module PDEOperators = struct
  type linear_operator = Tensor.t -> Tensor.t

  let gradient ~h f x =
    let d = Tensor.size x 0 in
    let grad = Tensor.zeros [d] in
    
    for i = 0 to d - 1 do
      let ei = Tensor.zeros [d] in
      Tensor.(copy_ (select ei i) (ones [1]));
      let x_plus = Tensor.(x + (ei * h)) in
      let x_minus = Tensor.(x - (ei * h)) in
      let df = Tensor.((f x_plus - f x_minus) / (2.0 * h)) in
      Tensor.(copy_ (select grad i) df)
    done;
    grad

  let divergence ~h f x =
    let d = Tensor.size x 0 in
    let div = ref (Tensor.zeros [1]) in
    
    for i = 0 to d - 1 do
      let ei = Tensor.zeros [d] in
      Tensor.(copy_ (select ei i) (ones [1]));
      let x_plus = Tensor.(x + (ei * h)) in
      let x_minus = Tensor.(x - (ei * h)) in
      let f_plus = f x_plus in
      let f_minus = f x_minus in
      div := Tensor.(!div + (select (f_plus - f_minus) i) / (2.0 * h))
    done;
    !div

  let laplacian ~h f x =
    divergence ~h (gradient ~h f) x
end

let stable_solve matrix b =
    let n = Tensor.size matrix 0 in
    let jitter = 1e-8 in
    let matrix_reg = Tensor.(matrix + (eye n * jitter)) in
    try
      Tensor.solve matrix_reg b
    with _ ->
      (* Fallback to more stable but slower method *)
      let q, r = QR.qr matrix_reg in
      let y = Tensor.matmul (Tensor.transpose q ~dim0:0 ~dim1:1) b in
      Tensor.triangular_solve ~upper:true r y

let condition_number matrix =
    let svd = Tensor.svd matrix in
    let s = match svd with
      | _, s, _ -> s in
    let max_s = Tensor.max s in
    let min_s = Tensor.min s in
    Tensor.(max_s / min_s)

let stable_cholesky matrix =
    let n = Tensor.size matrix 0 in
    let jitter_sequence = [1e-12; 1e-11; 1e-10; 1e-9; 1e-8; 1e-7; 1e-6] in
    
    let rec try_cholesky = function
      | [] -> failwith "Failed to compute stable Cholesky decomposition"
      | jitter :: rest ->
          try
            let matrix_reg = Tensor.(matrix + (eye n * jitter)) in
            Tensor.cholesky matrix_reg
          with _ -> try_cholesky rest
    in
    try_cholesky jitter_sequence

module VectorField = struct
 type t = {
   apply: Tensor.t -> Tensor.t;  (* Field evaluation *)
   grad: Tensor.t -> Tensor.t;   (* Jacobian *)
   div: Tensor.t -> Tensor.t;    (* Divergence *)
 }

 let create ~f ~h =
   let apply = f in
   let grad x = PDEOperators.gradient ~h f x in
   let div x = PDEOperators.divergence ~h f x in
   { apply; grad; div }

 let combine vf1 vf2 =
   let apply x = Tensor.(vf1.apply x + vf2.apply x) in
   let grad x = Tensor.(vf1.grad x + vf2.grad x) in
   let div x = Tensor.(vf1.div x + vf2.div x) in
   { apply; grad; div }
end

(* Core RKHS *)
module RKHS = struct
 type vector_kernel = Tensor.t -> Tensor.t -> Tensor.t

 let inner_product ~kernel f g points =
   let n = Tensor.size points 0 in
   let sum = ref (Tensor.zeros [1]) in
   
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     let fi = f xi in
     let gi = g xi in
     sum := Tensor.(!sum + dot (kernel xi fi) gi)
   done;
   !sum

 let norm ~kernel f points =
   Tensor.(sqrt (inner_product ~kernel f f points))

 let min_norm_interpolation ~kernel ~points ~values =
   let n = Tensor.size points 0 in
   let d = Tensor.size points 1 in
   
   (* Compute kernel matrix *)
   let k_matrix = Tensor.zeros [n; n] in
   for i = 0 to n - 1 do
     for j = 0 to n - 1 do
       let kij = kernel (Tensor.select points 0 i) (Tensor.select points 0 j) in
       Tensor.(copy_ (select (select k_matrix i) j) kij)
     done
   done;
   
   (* Solve system for coefficients *)
   let coeffs = NumericalStability.stable_solve k_matrix values in
   
   (* Return interpolating function *)
   fun x ->
     let kx = Tensor.zeros [n] in
     for i = 0 to n - 1 do
       let ki = kernel x (Tensor.select points 0 i) in
       Tensor.(copy_ (select kx i) ki)
     done;
     Tensor.matmul kx coeffs
end

let langevin_operator ~density f x =
   let open Tensor in
   let p_x = exp (of_float (density.log_prob x)) in
   let div_term = PDEOperators.divergence ~h:1e-6 f x in
   let grad_term = dot (f x) (density.grad_log_prob x) in
   (div_term + grad_term) / p_x

let variational_min ~kernel ~points ~f =
   let n = Tensor.size points 0 in
   let k_matrix = Tensor.zeros [n; n] in
   
   (* Build kernel matrix *)
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     for j = 0 to n - 1 do
       let xj = Tensor.select points 0 j in
       let kij = kernel xi xj in
       Tensor.(copy_ (select (select k_matrix i) j) kij)
     done
   done;
   
   (* Compute function values *)
   let f_vals = Tensor.zeros [n] in
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     Tensor.(copy_ (select f_vals i) (f xi))
   done;
   
   (* Solve minimization problem *)
   NumericalStability.stable_solve k_matrix f_vals

let stein_kernel ~density ~base_kernel x x' =
   let grad1_k = PDEOperators.gradient ~h:1e-6 
     (fun y -> base_kernel y x') x in
   let grad2_k = PDEOperators.gradient ~h:1e-6 
     (fun y -> base_kernel x y) x' in
   let grad_log_p_x = density.grad_log_prob x in
   let grad_log_p_x' = density.grad_log_prob x' in
   
   let term1 = Tensor.(dot grad1_k grad2_k) in
   let term2 = Tensor.(dot grad1_k grad_log_p_x') in
   let term3 = Tensor.(dot grad_log_p_x grad2_k) in
   let term4 = Tensor.(base_kernel x x' * dot grad_log_p_x grad_log_p_x') in
   
   Tensor.(term1 + term2 + term3 + term4)

let base_kernel ~length_scale x y =
   let diff = Tensor.(x - y) in
   let squared_dist = Tensor.(sum (diff * diff) ~dim:[0] ~keepdim:false) in
   Tensor.(exp (-squared_dist / (2.0 * length_scale * length_scale)))

(* Kernel optimization *)
module KernelOptimization = struct
 type cross_validation_result = {
   optimal_length_scale: float;
   cv_scores: (float * float) array;
   validation_error: float;
 }

 let cross_validate ~config ~density ~points ~target_fn ~k_folds ~length_scales =
   let n = Tensor.size points 0 in
   let fold_size = n / k_folds in

   (* Create folds *)
   let create_folds () =
     let indices = Array.init n (fun i -> i) in
     for i = n - 1 downto 1 do
       let j = Random.int (i + 1) in
       let temp = indices.(i) in
       indices.(i) <- indices.(j);
       indices.(j) <- temp
     done;
     Array.init k_folds (fun k ->
       let start_idx = k * fold_size in
       Array.sub indices start_idx fold_size)
   in
   
   let folds = create_folds () in
   let scores = Array.make (Array.length length_scales) 0.0 in
   
   Array.iteri (fun i length_scale ->
     let fold_errors = Array.make k_folds 0.0 in
     
     for k = 0 to k_folds - 1 do
       (* Split data *)
       let validation_indices = folds.(k) in
       let training_indices = Array.concat (
         Array.to_list (Array.filteri (fun j _ -> j <> k) folds)
       ) in
       
       let training_points = Tensor.index_select points 0 
         (Tensor.of_int1 training_indices) in
       let validation_points = Tensor.index_select points 0
         (Tensor.of_int1 validation_indices) in
       
       (* Train on fold *)
       let config' = { config with length_scale } in
       let solution = SteinSolver.solve ~config:config' ~density 
         ~points:training_points ~target_fn in
       
       (* Compute validation error *)
       let error = solution.residual_norm in
       fold_errors.(k) <- error
     done;
     
     scores.(i) <- Array.fold_left (+.) 0.0 fold_errors /. 
       float_of_int k_folds
   ) length_scales;
   
   (* Find optimal length scale *)
   let best_idx = ref 0 in
   let best_score = ref scores.(0) in
   for i = 1 to Array.length scores - 1 do
     if scores.(i) < !best_score then begin
       best_score := scores.(i);
       best_idx := i
     end
   done;
   
   {
     optimal_length_scale = length_scales.(!best_idx);
     cv_scores = Array.init (Array.length length_scales) (fun i -> 
       (length_scales.(i), scores.(i)));
     validation_error = !best_score;
   }
end

(* Chunked matrix-vector multiplication *)
let chunked_mv ~chunk_size matrix_action v =
   let (n, _) = matrix_action.MatrixOps.size in
   let result = Tensor.zeros [n] in
   
   for i = 0 to n - 1 step chunk_size do
     let chunk_end = min (i + chunk_size) n in
     let chunk_size = chunk_end - i in
     
     (* Extract chunk *)
     let v_chunk = Tensor.narrow v ~dim:0 ~start:i ~length:chunk_size in
     
     (* Compute chunk result *)
     let chunk_result = MatrixOps.mv matrix_action v_chunk in
     
     (* Update result *)
     for j = 0 to chunk_size - 1 do
       Tensor.(copy_ (select result (i + j)) (select chunk_result j))
     done
   done;
   result

(* Memory-efficient kernel matrix construction *)
let stream_kernel_matrix ~config ~density ~points f =
   let n = Tensor.size points 0 in
   let chunk_size = min 1000 n in
   
   for i = 0 to n - 1 step chunk_size do
     let chunk_end = min (i + chunk_size) n in
     let chunk_size = chunk_end - i in
     
     let chunk = Tensor.zeros [chunk_size; n] in
     
     for j = 0 to chunk_size - 1 do
       let xi = Tensor.select points 0 (i + j) in
       for k = 0 to n - 1 do
         let xk = Tensor.select points 0 k in
         let kij = KernelOps.stein_kernel 
           ~density 
           ~base_kernel:(KernelOps.base_kernel ~length_scale:config.length_scale)
           xi xk in
         Tensor.(copy_ (select (select chunk j) k) kij)
       done
     done;
     
     f ~start_idx:i ~chunk
   done

let random_features ~n_features ~length_scale points =
   let n = Tensor.size points 0 in
   let d = Tensor.size points 1 in
   
   (* Generate random weights and biases *)
   let omega = Tensor.(randn [n_features; d] / length_scale) in
   let b = Tensor.(uniform [n_features] ~low:0.0 ~high:(2. *. Float.pi)) in
   
   (* Compute feature map *)
   let compute_features x =
     let wx_b = Tensor.(matmul omega x + b) in
     Tensor.(cos wx_b * (sqrt (float_of_int (2 * n_features))))
   in
   
   (* Compute features for all points *)
   let all_features = Tensor.zeros [n; n_features] in
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     let phi_i = compute_features xi in
     Tensor.(copy_ (select all_features i) phi_i)
   done;
   
   compute_features, all_features

(* Create random feature preconditioner *)
let create ~n_features ~length_scale ~nugget matrix_action points =
   let n = Tensor.size points 0 in
   let compute_features, phi = random_features ~n_features ~length_scale points in
   
   (* Compute approximate kernel matrix *)
   let k_approx = Tensor.matmul phi (Tensor.transpose phi ~dim0:0 ~dim1:1) in
   let k_reg = Tensor.(k_approx + (eye n * nugget)) in
   
   fun v -> Tensor.matmul k_reg v

(* Preconditioner composition *)
module PreconditionerComposition = struct
 type t = Tensor.t -> Tensor.t

 (* Sequential composition of preconditioners *)
 let compose p1 p2 =
   fun v -> p1 (p2 v)

 (* Parallel composition of preconditioners with weights *)
 let combine ps weights =
   fun v ->
     let results = List.map2 (fun p w -> 
       Tensor.(p v * w)) ps weights in
     List.fold_left Tensor.(+) (Tensor.zeros_like v) results
end

(* Matrix operations *)
module MatrixOps = struct
 type matrix_action = {
   apply: Tensor.t -> Tensor.t;
   size: int * int;
 }

 let create_kernel_action ~config ~density ~points = 
   let n = Tensor.size points 0 in
   let apply v =
     let result = Tensor.zeros [n] in
     for i = 0 to n - 1 do
       let xi = Tensor.select points 0 i in
       let sum = ref (Tensor.zeros [1]) in
       for j = 0 to n - 1 do
         let xj = Tensor.select points 0 j in
         let kij = KernelOps.stein_kernel 
           ~density 
           ~base_kernel:(KernelOps.base_kernel ~length_scale:config.length_scale)
           xi xj in
         let vj = Tensor.select v j in
         sum := Tensor.(!sum + kij * vj)
       done;
       Tensor.(copy_ (select result i) !sum)
     done;
     result
   in
   { apply; size = (n, n) }

 let mv action v =
   let (m, n) = action.size in
   assert (Tensor.size v 0 = n);
   action.apply v
end

(* Basic uniform sampling *)
let uniform ~n_samples n =
   let indices = Array.init n (fun i -> i) in
   for i = n - 1 downto 1 do
     let j = Random.int (i + 1) in
     let temp = indices.(i) in
     indices.(i) <- indices.(j);
     indices.(j) <- temp
   done;
   Array.sub indices 0 n_samples

(* Diagonal sampling *)
let diagonal ~n_samples ~kernel_diag n =
   let probs = Array.init n (fun i -> Tensor.get kernel_diag [i]) in
   let sum = Array.fold_left (+.) 0.0 probs in
   let probs = Array.map (fun p -> p /. sum) probs in
   
   let selected = Array.make n_samples (-1) in
   let used = Array.make n false in
   
   for i = 0 to n_samples - 1 do
     let r = Random.float 1.0 in
     let mutable sum = 0.0 in
     let mutable idx = 0 in
     while idx < n && sum < r do
       if not used.(idx) then
         sum <- sum +. probs.(idx);
       idx <- idx + 1
     done;
     let selected_idx = idx - 1 in
     used.(selected_idx) <- true;
     selected.(i) <- selected_idx
   done;
   selected
end

(* Symmetric collocation method *)
let symmetric_collocation ~operator ~kernel ~points ~f =
   let n = Tensor.size points 0 in
   
   (* Build collocation matrix *)
   let a = Tensor.zeros [n; n] in
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     for j = 0 to n - 1 do
       let xj = Tensor.select points 0 j in
       let aij = operator (fun x -> kernel x xj) xi in
       Tensor.(copy_ (select (select a i) j) aij)
     done
   done;
   
   (* Build RHS *)
   let b = Tensor.zeros [n] in
   for i = 0 to n - 1 do
     let xi = Tensor.select points 0 i in
     Tensor.(copy_ (select b i) (f xi))
   done;
   
   (* Solve collocation system *)
   NumericalStability.stable_solve a b

(* Remove duplicate collocation points *)
let deduplicate_points points =
   let n = Tensor.size points 0 in
   let d = Tensor.size points 1 in
   let seen = Hashtbl.create n in
   let unique = ref [] in
   
   for i = 0 to n - 1 do
     let point = Tensor.select points 0 i in
     let key = Tensor.to_float_list point in
     if not (Hashtbl.mem seen key) then begin
       Hashtbl.add seen key ();
       unique := point :: !unique
     end
   done;
   
   let n_unique = List.length !unique in
   let result = Tensor.zeros [n_unique; d] in
   List.iteri (fun i point ->
     Tensor.(copy_ (select result i) point)
   ) (List.rev !unique);
   result

(* Linear solver *)
module LinearSolver = struct
 type solution = {
   value: Tensor.t;
   iterations: int;
   error: float;
 }

 (* Conjugate gradient solver *)
 let conjugate_gradient ~matrix_action ~preconditioner ~b ~tol ~max_iter =
   let n = Tensor.size b 0 in
   let x = Tensor.zeros [n] in
   let r = Tensor.(b - matrix_action x) in
   let z = preconditioner r in
   let p = z in
   
   let rec iterate x r z p iter =
     if iter >= max_iter then
       { value = x; iterations = iter; error = Tensor.(norm r) |> float_of_float }
     else
       let ap = matrix_action p in
       let rz = Tensor.(dot r z) in
       let alpha = Tensor.(rz / dot p ap) in
       let x_new = Tensor.(x + p * alpha) in
       let r_new = Tensor.(r - ap * alpha) in
       let r_norm = Tensor.(norm r_new) |> float_of_float in
       
       if r_norm < tol then
         { value = x_new; iterations = iter; error = r_norm }
       else
         let z_new = preconditioner r_new in
         let rz_new = Tensor.(dot r_new z_new) in
         let beta = rz_new /. rz in
         let p_new = Tensor.(z_new + p * beta) in
         iterate x_new r_new z_new p_new (iter + 1)
   in
   iterate x r z p 0

 (* MINRES solver for symmetric indefinite systems *)
 let minres ~matrix_action ~b ~tol ~max_iter =
   let n = Tensor.size b 0 in
   let x = Tensor.zeros [n] in
   let r = Tensor.(b - matrix_action x) in
   let v_old = Tensor.zeros [n] in
   let v = Tensor.(r / norm r) in
   
   let rec iterate x v v_old beta iter =
     if iter >= max_iter then
       { value = x; iterations = iter; error = beta }
     else
       let w = Tensor.(matrix_action v - v_old * beta) in
       let alpha = Tensor.(dot v w) in
       let w = Tensor.(w - v * alpha) in
       let beta_next = Tensor.(norm w) |> float_of_float in
       let w = Tensor.(w / beta_next) in
       
       let x_new = Tensor.(x + v * alpha) in
       if beta_next < tol then
         { value = x_new; iterations = iter; error = beta_next }
       else
         iterate x_new w v beta_next (iter + 1)
   in
   let beta_init = Tensor.(norm r) |> float_of_float in
   iterate x v v_old beta_init 0
end

(* Core preconditioner types and interface *)
module PreconditionerInterface = struct
 type strategy = [
   | `None
   | `Jacobi of int
   | `BlockJacobi of { block_size: int; n: int }
   | `Nystrom of {
       n_samples: int;
       nugget: float;
       sampling: [ `Uniform | `Diagonal of Tensor.t ]
     }
   | `FITC of {
       n_samples: int;
       nugget: float;
       sampling: [ `Uniform | `Diagonal of Tensor.t ]
     }
   | `RandomFeatures of {
       n_features: int;
       length_scale: float;
       nugget: float;
     }
 ]

 let create strategy matrix_action points =
   match strategy with
   | `None -> fun v -> v
   | `Jacobi n -> 
       Preconditioner.jacobi matrix_action n
   | `BlockJacobi { block_size; n } ->
       Preconditioner.block_jacobi ~block_size matrix_action n
   | `Nystrom { n_samples; nugget; sampling } ->
       AdvancedPreconditioner.nystrom 
         ~n_samples ~nugget ~sampling_strategy:sampling 
         matrix_action points
   | `FITC { n_samples; nugget; sampling } ->
       AdvancedPreconditioner.fitc
         ~n_samples ~nugget ~sampling_strategy:sampling
         matrix_action points
   | `RandomFeatures { n_features; length_scale; nugget } ->
       RandomFeature.create
         ~n_features ~length_scale ~nugget
         matrix_action points
end
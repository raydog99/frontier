open Torch

type point = Tensor.t
type multi_index = int array

type coefficient = 
  | Constant of float
  | Function of (point -> float)
  | Tensor of Tensor.t

type linear_operator = {
  order: int;
  eval: Tensor.t -> point -> Tensor.t;
  adjoint: Tensor.t -> point -> Tensor.t;
  domain_dim: int;
  range_dim: int;
}

module type Kernel = sig
  type rkhs = {
    centers: point array;
    coefficients: Tensor.t array;
    length_scale: float;
  }

  val length_scale : float
  val evaluate : point -> point -> Tensor.t
  val gradient : point -> point -> Tensor.t
  val partial_derivative : multi_index -> point -> point -> Tensor.t
end

module FunctionSpace = struct
  type norm_type = L2 | H1 | H2 | Sobolev of int

  let compute_norm space_type f points =
    match space_type with
    | L2 -> 
        let values = Array.map f points in
        Tensor.(sqrt (mean (stack values 0 * stack values 0)))
    | H1 ->
        let values = Array.map f points in
        let grads = Array.map (Tensor.grad_of_fn f) points in
        let l2_part = Tensor.(mean (stack values 0 * stack values 0)) in
        let grad_part = Array.fold_left (fun acc g ->
          Tensor.(acc + mean (g * g))) (Tensor.zeros [1]) grads in
        Tensor.(sqrt (l2_part + grad_part))
    | H2 ->
        let values = Array.map f points in
        let grads = Array.map (Tensor.grad_of_fn f) points in
        let hessians = Array.map (fun x ->
          Tensor.grad_of_fn (Tensor.grad_of_fn f) x) points in
        let l2_part = Tensor.(mean (stack values 0 * stack values 0)) in
        let grad_part = Array.fold_left (fun acc g ->
          Tensor.(acc + mean (g * g))) (Tensor.zeros [1]) grads in
        let hess_part = Array.fold_left (fun acc h ->
          Tensor.(acc + mean (h * h))) (Tensor.zeros [1]) hessians in
        Tensor.(sqrt (l2_part + grad_part + hess_part))
    | Sobolev k ->
        if k < 0 then invalid_arg "Negative Sobolev order";
        let parts = Array.init (k+1) (fun i ->
          let derivatives = Array.map (fun x ->
            let rec deriv n f x =
              if n = 0 then f x
              else Tensor.grad_of_fn (deriv (n-1) f) x
            in deriv i f x) points in
          Array.fold_left (fun acc d ->
            Tensor.(acc + mean (d * d))) (Tensor.zeros [1]) derivatives
        ) in
        Tensor.(sqrt (sum (stack parts 0)))

  let test_embedding f ~source_space ~target_space ~test_points =
    let source_norm = compute_norm source_space f test_points in
    let target_norm = compute_norm target_space f test_points in
    Tensor.(target_norm <= (Scalar.f 1.5) * source_norm)
end

let derivative_operator k alpha x y =
    let (module K: Kernel) = k in
    let d = ref (K.evaluate x y) in
    Array.iteri (fun i order ->
      for _ = 1 to order do
        d := Tensor.(select (K.gradient !d x) 0 i)
      done) alpha;
    !d

let kernel_matrix k points alphas =
    let n = Array.length points in
    let m = Array.length alphas in
    let mat = Tensor.zeros [n * m; n * m] in
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        for a = 0 to m-1 do
          for b = 0 to m-1 do
            let kij = derivative_operator k 
              (Array.append alphas.(a) alphas.(b))
              points.(i) points.(j) in
            Tensor.set mat [|i*m + a; j*m + b|] kij
          done
        done
      done
    done;
    mat

let feature_matrix k points values alphas =
    let n = Array.length points in
    let m = Array.length alphas in
    let mat = Tensor.zeros [n; m] in
    for i = 0 to n-1 do
      for j = 0 to m-1 do
        let deriv = derivative_operator k alphas.(j) points.(i) points.(i) in
        Tensor.set mat [|i; j|] deriv
      done
    done;
    mat

let apply_regularization mat lambda regtype =
    match regtype with
    | `Standard -> 
        Tensor.(mat + (eye (size mat.(0)) * (Scalar.f lambda)))
    | `Tikhonov ->
        let n = Tensor.size mat.(0) in
        let l = Tensor.(eye n * (Scalar.f lambda)) in
        let lap = Tensor.zeros [n; n] in
        for i = 1 to n-2 do
          Tensor.set lap [|i; i-1|] (Scalar.f (-1.));
          Tensor.set lap [|i; i|] (Scalar.f 2.);
          Tensor.set lap [|i; i+1|] (Scalar.f (-1.));
        done;
        Tensor.(mat + mm l lap)
    | `Adaptive delta ->
        let n = Tensor.size mat.(0) in
        let diag = Tensor.diagonal mat 0 in
        let scale = Tensor.(mean diag * (Scalar.f delta)) in
        Tensor.(mat + (eye n * scale))

let solve_regularized_system k points values alphas lambda =
    let gram = kernel_matrix k points alphas in
    let reg_gram = apply_regularization gram lambda `Standard in
    let phi = feature_matrix k points values alphas in
    let rhs = Tensor.(mm (transpose2 phi 0 1) values) in
    Tensor.(solve reg_gram rhs)

module OptimizationMethods = struct
  type optimization_method = 
    | GradientDescent of {step_size: float}
    | LBFGS of {m: int; max_line_search: int}
    | GaussNewton
    | TrustRegion of {radius: float; eta: float}

  let lbfgs_update s_list y_list grad =
    let m = List.length s_list in
    let q = ref grad in
    let alpha_list = ref [] in
    
    List.iter2 (fun s y ->
      let alpha = Tensor.(sum (s * !q)) /. Tensor.(sum (s * y)) in
      alpha_list := alpha :: !alpha_list;
      q := Tensor.(!q - (Scalar.f alpha * y))
    ) s_list y_list;
    
    let r = ref !q in
    List.iter2 (fun s y ->
      let beta = Tensor.(sum (y * !r)) /. Tensor.(sum (s * y)) in
      let alpha = List.hd !alpha_list in
      alpha_list := List.tl !alpha_list;
      r := Tensor.(!r + (s * (Scalar.f (alpha -. beta))))
    ) (List.rev s_list) (List.rev y_list);
    !r

  let gauss_newton_update jacobian residual =
    let jtj = Tensor.(mm (transpose2 jacobian 0 1) jacobian) in
    let jtr = Tensor.(mm (transpose2 jacobian 0 1) residual) in
    Tensor.solve jtj jtr

  let trust_region_update ~radius ~eta hessian grad =
    let n = Tensor.size grad.(0) in
    let identity = Tensor.eye n in
    
    let rec solve_subproblem lambda =
      let reg_hessian = Tensor.(hessian + (identity * (Scalar.f lambda))) in
      let step = Tensor.solve reg_hessian (Tensor.neg grad) in
      if Tensor.(norm step) <= radius then step
      else solve_subproblem (lambda *. 2.)
    in
    solve_subproblem eta
end

module RBFKernel = struct
  type rkhs = {
    centers: point array;
    coefficients: Tensor.t array;
    length_scale: float;
  }

  let length_scale = 1.0

  let evaluate x1 x2 =
    let diff = Tensor.(x1 - x2) in
    let squared_dist = Tensor.(sum (diff * diff) ~dim:[0] ~keepdim:true) in
    Tensor.(exp (neg squared_dist / (Scalar.f (2.0 * length_scale * length_scale))))

  let gradient x1 x2 =
    let diff = Tensor.(x2 - x1) in
    let k = evaluate x1 x2 in
    Tensor.(k * diff / (Scalar.f (length_scale * length_scale)))

  let partial_derivative alpha x1 x2 =
    let k = evaluate x1 x2 in
    let diff = Tensor.(x2 - x1) in
    let order = Array.fold_left (+) 0 alpha in
    match order with
    | 0 -> k
    | 1 -> Tensor.(k * diff / (Scalar.f (length_scale * length_scale)))
    | _ -> 
        let h = Tensor.(diff / (Scalar.f length_scale)) in
        let scale = Float.pow length_scale (float_of_int order) in
        Tensor.(k * (h ** (Scalar.i order)) / (Scalar.f scale))
end

module MaternKernel(P: sig val nu: float end) = struct
  type rkhs = {
    centers: point array;
    coefficients: Tensor.t array;
    length_scale: float;
  }

  let length_scale = 1.0

  let evaluate x1 x2 =
    let r = Tensor.(sqrt (sum ((x1 - x2) * (x1 - x2)) ~dim:[0] ~keepdim:true)) in
    let r_scale = Tensor.(r / (Scalar.f P.nu)) in
    match Float.to_int P.nu with
    | 1 -> Tensor.(exp (neg r_scale))
    | 3 -> Tensor.((1. + r_scale) * exp (neg r_scale))
    | 5 -> Tensor.((1. + r_scale + r_scale * r_scale / 3.) * exp (neg r_scale))
    | _ -> failwith "Unsupported Matérn smoothness parameter"

  let gradient x1 x2 =
    let r = Tensor.(sqrt (sum ((x1 - x2) * (x1 - x2)) ~dim:[0] ~keepdim:true)) in
    let r_scale = Tensor.(r / (Scalar.f P.nu)) in
    let diff = Tensor.(x2 - x1) in
    match Float.to_int P.nu with
    | 1 -> Tensor.(diff * exp (neg r_scale) / r)
    | 3 -> Tensor.(diff * ((neg r_scale - 1.) * exp (neg r_scale)) / r)
    | 5 -> 
        let term = Tensor.((neg r_scale * (r_scale + 3.) / 3.) * exp (neg r_scale)) in
        Tensor.(diff * term / r)
    | _ -> failwith "Unsupported Matérn smoothness parameter"

  let partial_derivative alpha x1 x2 =
    (* Matérn kernel *)
    let k = evaluate x1 x2 in
    let diff = Tensor.(x2 - x1) in
    let order = Array.fold_left (+) 0 alpha in
    match order with
    | 0 -> k
    | 1 -> Tensor.(k * diff / (Scalar.f (length_scale * length_scale)))
    | _ -> 
        let h = Tensor.(diff / (Scalar.f length_scale)) in
        let scale = Float.pow length_scale (float_of_int order) in
        Tensor.(k * (h ** (Scalar.i order)) / (Scalar.f scale))
end

module PolynomialKernel(P: sig val degree: int end) = struct
  type rkhs = {
    centers: point array;
    coefficients: Tensor.t array;
    length_scale: float;
  }

  let length_scale = 1.0

  let evaluate x1 x2 =
    let dot_prod = Tensor.(sum (x1 * x2) ~dim:[0] ~keepdim:true) in
    Tensor.((dot_prod + (Scalar.f 1.)) ** (Scalar.i P.degree))

  let gradient x1 x2 =
    let dot_prod = Tensor.(sum (x1 * x2) ~dim:[0] ~keepdim:true) in
    let base = Tensor.(dot_prod + (Scalar.f 1.)) in
    Tensor.(x2 * ((Scalar.i P.degree) * (base ** (Scalar.i (P.degree - 1)))))

  let partial_derivative alpha x1 x2 =
    (* Polynomial kernel specifics *)
    let k = evaluate x1 x2 in
    let diff = Tensor.(x2 - x1) in
    let order = Array.fold_left (+) 0 alpha in
    match order with
    | 0 -> k
    | 1 -> gradient x1 x2
    | _ -> 
        (* Higher order derivatives for polynomial kernel *)
        let base = Tensor.(sum (x1 * x2) ~dim:[0] ~keepdim:true + (Scalar.f 1.)) in
        Tensor.(k * (base ** (Scalar.i (P.degree - order))))
end

module NonlinearPDE = struct
  type nonlinear_term =
    | Polynomial of float array
    | Rational of float array * float array
    | Composition of (Tensor.t -> Tensor.t) * linear_operator
    | Product of nonlinear_term list
    | Sum of nonlinear_term list

  let rec evaluate_term term u x =
    match term with
    | Polynomial coeffs ->
        let result = ref (Tensor.zeros [1]) in
        Array.iteri (fun i c ->
          result := Tensor.(!result + (Scalar.f c * (u ** (Scalar.i i))))
        ) coeffs;
        !result
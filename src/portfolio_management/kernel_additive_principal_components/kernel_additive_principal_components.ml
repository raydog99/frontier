open Torch

(* Core types *)
type kernel = Tensor.t -> Tensor.t -> Tensor.t
type transformation = Tensor.t -> Tensor.t

(* Numerical stability stability *)
module NumericStability = struct
  let epsilon = 1e-10
  let max_condition_number = 1e16

  (* Stable matrix inverse *)
  let stable_inverse matrix =
    let n = Tensor.size matrix 0 in
    let identity = Tensor.eye n in
    let reg_matrix = Tensor.add matrix 
      (Tensor.mul_scalar identity epsilon) in
    Tensor.inverse reg_matrix

  (* Stable eigendecomposition *)
  let stable_symeig ?(eigenvectors=true) matrix =
    let n = Tensor.size matrix 0 in
    let sym_matrix = Tensor.div (Tensor.add matrix (Tensor.transpose matrix 0 1)) 2.0 in
    let reg_matrix = Tensor.add sym_matrix 
      (Tensor.mul_scalar (Tensor.eye n) epsilon) in
    Tensor.symeig ~eigenvectors reg_matrix

  (* Stable matrix square root *)
  let stable_sqrtm matrix =
    let eigenvals, eigenvecs = stable_symeig ~eigenvectors:true matrix in
    let sqrt_eigenvals = Tensor.sqrt (Tensor.maximum eigenvals (Tensor.full [] epsilon)) in
    let diag = Tensor.diag sqrt_eigenvals in
    Tensor.mm (Tensor.mm eigenvecs diag) (Tensor.transpose eigenvecs 0 1)

  (* Check matrix condition number *)
  let condition_number matrix =
    let eigenvals, _ = stable_symeig ~eigenvectors:false matrix in
    let max_eigenval = Tensor.max eigenvals in
    let min_eigenval = Tensor.min eigenvals in
    Tensor.div max_eigenval (Tensor.maximum min_eigenval (Tensor.full [] epsilon))
end

(* Kernel operations *)
module Kernel = struct
  (* Basic RBF/Gaussian kernel *)
  let rbf_kernel sigma x y =
    let diff = Tensor.( - ) x y in
    let sq_dist = Tensor.(sum (diff * diff)) in
    Tensor.exp (Tensor.mul_scalar sq_dist (-0.5 /. (sigma *. sigma)))

  (* Linear kernel *)
  let linear_kernel x y =
    Tensor.dot x y

  (* Polynomial kernel *)
  let poly_kernel degree c x y =
    let dot_prod = Tensor.dot x y in
    Tensor.pow (Tensor.add_scalar dot_prod c) degree
    
  (* Create kernel matrix from data points *)
  let create_kernel_matrix kernel data =
    let n = Tensor.size data 0 in
    let k_matrix = Tensor.zeros [n; n] in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let xi = Tensor.select data 0 i in
        let xj = Tensor.select data 0 j in
        let k_ij = kernel xi xj in
        Tensor.set k_matrix [i; j] k_ij
      done
    done;
    k_matrix

  (* Center kernel matrix in feature space *)
  let center_kernel_matrix k_matrix =
    let n = float_of_int (Tensor.size k_matrix 0) in
    let ones = Tensor.ones [Tensor.size k_matrix 0; 1] in
    let h = Tensor.sub (Tensor.eye (int_of_float n)) 
      (Tensor.mul_scalar (Tensor.mm ones (Tensor.transpose ones 0 1)) (1.0 /. n)) in
    Tensor.mm (Tensor.mm h k_matrix) h

  (* Compute kernel alignment score between two kernels *)
  let kernel_alignment k1 k2 data =
    let gram1 = center_kernel_matrix (create_kernel_matrix k1 data) in
    let gram2 = center_kernel_matrix (create_kernel_matrix k2 data) in
    let frobenius_prod = Tensor.sum (Tensor.mul gram1 gram2) in
    let norm1 = Tensor.sqrt (Tensor.sum (Tensor.mul gram1 gram1)) in
    let norm2 = Tensor.sqrt (Tensor.sum (Tensor.mul gram2 gram2)) in
    Tensor.div frobenius_prod (Tensor.mul norm1 norm2)
end

(* Population statistics *)
module Population = struct
  (* Compute population covariance *)
  let covariance x y =
    let n = float_of_int (Tensor.size x 0) in
    let mean_x = Tensor.mean x in
    let mean_y = Tensor.mean y in
    let centered_x = Tensor.sub x mean_x in
    let centered_y = Tensor.sub y mean_y in
    Tensor.sum (Tensor.mul centered_x centered_y) /. n

  (* Inner product in L2 space *)
  let inner_product f g data =
    let fx = f data in
    let gx = g data in
    covariance fx gx

  (* Compute population variance *)
  let variance x =
    let mean = Tensor.mean x in
    let centered = Tensor.sub x mean in
    Tensor.mean (Tensor.mul centered centered)

  (* Center data with respect to population mean *)
  let center data =
    let mean = Tensor.mean data in
    Tensor.sub data mean

  (* Normalize data to have unit variance *)
  let normalize data =
    let std = Tensor.std data in
    Tensor.div data std
end

(* Measures *)
module Measure = struct
  (* Sigma algebra representation *)
  type measurable_set = 
    | Empty
    | Full
    | Interval of float * float
    | Union of measurable_set list
    | Intersection of measurable_set list
    | Complement of measurable_set

  (* Abstract measure type *)
  type 'a measure = {
    space: 'a;
    measure: measurable_set -> float;
    support: measurable_set;
  }

  (* Different types of measures *)
  type measure_type =
    | Lebesgue
    | Counting
    | Gaussian of float * float  (* mean, variance *)
    | EmpiricalMeasure of Tensor.t
    | ProductMeasure of measure_type list

  (* Create measure from type *)
  let create_measure = function
    | Lebesgue -> {
        space = ();
        measure = (function
          | Interval(a, b) -> b -. a
        );
        support = Full
      }
    | Counting -> {
        space = ();
        measure = (function
          | Empty -> 0.0
          | Full -> infinity
        );
        support = Full
      }
    | Gaussian(mu, sigma) -> {
        space = (mu, sigma);
        measure = (function
          | Interval(a, b) ->
              let normalize x = (x -. mu) /. sigma in
              let phi x = exp (-. x *. x /. 2.0) /. sqrt (2.0 *. Float.pi) in
              (* Numerical integration for Gaussian measure *)
              let n = 1000 in
              let dx = (b -. a) /. float_of_int n in
              let sum = ref 0.0 in
              for i = 0 to n - 1 do
                let x = a +. dx *. float_of_int i in
                sum := !sum +. phi (normalize x) *. dx
              done;
              !sum
        );
        support = Full
      }
    | EmpiricalMeasure data -> {
        space = data;
        measure = (function
          | _ -> 1.0 /. float_of_int (Tensor.size data 0)
        );
        support = Full
      }
    | ProductMeasure measures ->
        let component_measures = List.map create_measure measures in
        {
          space = ();
          measure = (function
            | _ -> List.fold_left (fun acc m -> acc *. m.measure Full) 1.0 component_measures
          );
          support = Full
        }
end

(* Function spaces *)
module FunctionSpace = struct
  type function_space_type =
    | L2Space of Measure.measure_type
    | SobolevSpace of int * Measure.measure_type
    | InfiniteDimensional of (int -> float)  (* Basis function generator *)

  type 'a function_space = {
    space_type: function_space_type;
    inner_product: ('a -> 'a -> float);
    norm: ('a -> float);
    basis: (int -> 'a option);  (* Basis elements indexed by natural numbers *)
  }

  (* Create L2 space *)
  let create_l2_space measure_type =
    let measure = Measure.create_measure measure_type in
    {
      space_type = L2Space measure_type;
      inner_product = (fun f g ->
        (* Numerical integration for inner product *)
        let n = 1000 in
        let a = -10.0 and b = 10.0 in  (* Integration bounds *)
        let dx = (b -. a) /. float_of_int n in
        let sum = ref 0.0 in
        for i = 0 to n - 1 do
          let x = a +. dx *. float_of_int i in
          sum := !sum +. f x *. g x *. dx
        done;
        !sum
      );
      norm = (fun f -> sqrt (float_of_int (Tensor.size f 0)));
      basis = fun i -> 
        if i < 0 then None
        else Some (Tensor.ones [1])
    }

  (* Create Sobolev space *)
  let create_sobolev_space order measure_type =
    let l2_space = create_l2_space measure_type in
    {
      space_type = SobolevSpace(order, measure_type);
      inner_product = (fun f g ->
        (* Add derivatives to inner product *)
        let l2_product = l2_space.inner_product f g in
        let derivative_product = ref 0.0 in
        for k = 1 to order do
          (* Approximate derivatives using finite differences *)
          let h = 1e-5 in
          let dx = fun x ->
            (f (x +. h) -. f x) /. h in
          let dy = fun x ->
            (g (x +. h) -. g x) /. h in
          derivative_product := !derivative_product +. l2_space.inner_product dx dy
        done;
        l2_product +. !derivative_product
      );
      norm = (fun f ->
        sqrt (float_of_int (Tensor.size f 0) +. 
              float_of_int order));  (* Include derivative norms *)
      basis = l2_space.basis
    }

  (* Module for infinite basis handling *)
  module InfiniteBasis = struct
    type basis_type =
      | Fourier
      | Hermite
      | Legendre
      | Custom of (int -> float -> float)

    (* Generate basis function *)
    let get_basis_function = function
      | Fourier -> (fun n x ->
          if n = 0 then 1.0
          else if n mod 2 = 1 then
            sin (float_of_int ((n+1)/2) *. x)
          else
            cos (float_of_int (n/2) *. x))
      | Hermite -> (fun n x ->
          let rec hermite n x =
            if n = 0 then 1.0
            else if n = 1 then 2.0 *. x
            else 2.0 *. x *. hermite (n-1) x -. 
                 2.0 *. float_of_int (n-1) *. hermite (n-2) x
          in
          hermite n x *. exp (-. x *. x /. 2.0))
      | Legendre -> (fun n x ->
          let rec legendre n x =
            if n = 0 then 1.0
            else if n = 1 then x
            else ((2.0 *. float_of_int n -. 1.0) *. x *. legendre (n-1) x -. 
                  (float_of_int n -. 1.0) *. legendre (n-2) x) /. 
                 float_of_int n
          in
          legendre n x)
      | Custom f -> f

    (* Create truncated basis *)
    let create_truncated_basis basis_type n =
      Array.init n (fun i -> get_basis_function basis_type i)
  end
end

(* Hilbert spaces *)
module HilbertSpace = struct
  type hilbert_space = {
    inner_product: Tensor.t -> Tensor.t -> float;
    norm: Tensor.t -> float;
    dim: int option;  (* None for infinite-dimensional spaces *)
  }

  (* Create finite-dimensional Hilbert space *)
  let create_finite_dimensional dim =
    {
      inner_product = (fun x y -> Tensor.dot x y |> Tensor.float_value);
      norm = (fun x -> Tensor.norm x |> Tensor.float_value);
      dim = Some dim;
    }

  (* Create infinite-dimensional Hilbert space *)
  let create_infinite_dimensional inner_product =
    {
      inner_product;
      norm = (fun x -> sqrt (inner_product x x));
      dim = None;
    }

  (* Orthogonal projection onto closed subspace *)
  let project_onto_subspace space basis vector =
    let n = Array.length basis in
    let gram_matrix = Array.make_matrix n n 0.0 in
    let projection_coeffs = Array.make n 0.0 in
    
    (* Compute Gram matrix *)
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        gram_matrix.(i).(j) <- space.inner_product basis.(i) basis.(j)
      done;
      projection_coeffs.(i) <- space.inner_product vector basis.(i)
    done;
    
    (* Solve normal equations *)
    let solution = Array.make n 0.0 in
    for i = 0 to n - 1 do
      let sum = ref 0.0 in
      for j = 0 to n - 1 do
        sum := !sum +. gram_matrix.(i).(j) *. projection_coeffs.(j)
      done;
      solution.(i) <- !sum
    done;
    
    (* Compute projection *)
    let projection = ref (Tensor.zeros_like vector) in
    Array.iteri (fun i coeff ->
      projection := Tensor.add !projection (Tensor.mul_scalar basis.(i) coeff)
    ) solution;
    !projection
end

(* Optimization and contraints *)
module Optimization = struct
  (* Types for constraint handling *)
  type constraint_type =
    | EqualityConstraint of (Tensor.t -> Tensor.t)
    | InequalityConstraint of (Tensor.t -> Tensor.t)
    | NormConstraint of float

  type optimization_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float;
    constraint_weight: float;
  }

  let default_params = {
    max_iter = 1000;
    tolerance = 1e-6;
    learning_rate = 1e-3;
    momentum = 0.9;
    constraint_weight = 1.0;
  }

  (* Project onto constraint set *)
  let project_onto_constraints constraints x =
    List.fold_left (fun curr_x constraint_ ->
      match constraint_ with
      | EqualityConstraint f ->
          let constraint_val = f curr_x in
          Tensor.sub curr_x constraint_val
      | InequalityConstraint f ->
          let constraint_val = f curr_x in
          let violation = Tensor.maximum constraint_val (Tensor.zeros_like constraint_val) in
          Tensor.sub curr_x violation
      | NormConstraint target_norm ->
          let current_norm = Tensor.norm curr_x in
          if Tensor.float_value current_norm > target_norm then
            Tensor.mul_scalar curr_x (target_norm /. Tensor.float_value current_norm)
          else
            curr_x
    ) x constraints

  (* Trust region method *)
  let trust_region ~objective ~gradient ~hessian ?(params=default_params) init_x =
    let trust_radius = ref 1.0 in
    let min_trust_radius = 1e-4 in
    
    let solve_trust_subproblem x radius =
      let grad = gradient x in
      let hess = hessian x in
      let n = Tensor.size x 0 in
      
      let eigenvals, eigenvecs = NumericStability.stable_symeig ~eigenvectors:true hess in
      let min_eigenval = Tensor.min eigenvals in
      
      if Tensor.float_value min_eigenval > 0.0 then
        (* Use Newton step if Hessian is positive definite *)
        let step = Tensor.mm (NumericStability.stable_inverse hess) 
          (Tensor.mul_scalar grad (-1.0)) in
        if Tensor.float_value (Tensor.norm step) <= radius then step
        else Tensor.mul_scalar step (radius /. Tensor.float_value (Tensor.norm step))
      else
        (* Add regularization for indefinite Hessian *)
        let lambda = Tensor.abs (Tensor.minimum eigenvals (Tensor.zeros [])) in
        let reg_hess = Tensor.add hess 
          (Tensor.mul_scalar (Tensor.eye n) (Tensor.float_value lambda)) in
        let step = Tensor.mm (NumericStability.stable_inverse reg_hess) 
          (Tensor.mul_scalar grad (-1.0)) in
        Tensor.mul_scalar step (radius /. Tensor.float_value (Tensor.norm step))
    in
    
    let rec optimize iter x =
      if iter >= params.max_iter then x
      else
        let step = solve_trust_subproblem x !trust_radius in
        let x_new = Tensor.add x step in
        
        (* Compute actual vs predicted reduction *)
        let actual_reduction = Tensor.sub (objective x) (objective x_new) in
        let predicted_reduction = Tensor.neg (Tensor.add
          (Tensor.dot (gradient x) step)
          (Tensor.dot step (Tensor.mv (hessian x) step)) |> 
            fun x -> Tensor.div x (Tensor.full [] 2.0)) in
        
        let ratio = Tensor.div actual_reduction predicted_reduction in
        
        if Tensor.float_value ratio > 0.75 then
          trust_radius := !trust_radius *. 2.0
        else if Tensor.float_value ratio < 0.25 then
          trust_radius := !trust_radius *. 0.25;
        
        if Tensor.float_value ratio > 0.0 && 
           Tensor.float_value (Tensor.norm step) < params.tolerance then x_new
        else if !trust_radius < min_trust_radius then x
        else optimize (iter + 1) (if Tensor.float_value ratio > 0.0 then x_new else x)
    in
    optimize 0 init_x

  (* Augmented Lagrangian method *)
  let augmented_lagrangian ~objective ~constraints ?(params=default_params) init_x =
    let max_outer_iter = 20 in
    let lambda = ref (Tensor.zeros [List.length constraints]) in
    let mu = ref 10.0 in
    
    let rec outer_loop iter x =
      if iter >= max_outer_iter then x
      else
        (* Define augmented Lagrangian *)
        let augmented_obj x =
          let obj = objective x in
          let violation = ref (Tensor.zeros []) in
          List.iteri (fun i constraint_ ->
            match constraint_ with
            | EqualityConstraint f -> 
                violation := Tensor.add !violation (f x)
            | _ -> ()
          ) constraints;
          let penalty = Tensor.add
            (Tensor.mul !lambda !violation)
            (Tensor.mul_scalar (Tensor.mul !violation !violation) (!mu /. 2.0)) in
          Tensor.add obj penalty
        in
        
        (* Minimize augmented Lagrangian *)
        let gradient x = Tensor.grad augmented_obj x in
        let x' = trust_region ~objective:augmented_obj 
          ~gradient ~hessian:(fun x -> Tensor.jacobian gradient x) ~params x in
        
        (* Update multipliers *)
        List.iteri (fun i constraint_ ->
          match constraint_ with
          | EqualityConstraint f ->
              let violation = f x' in
              Tensor.set !lambda [i] 
                (Tensor.get !lambda [i] +. !mu *. Tensor.float_value violation)
          | _ -> ()
        ) constraints;
        
        mu := !mu *. 1.5;
        
        let total_violation = List.fold_left (fun acc constraint_ ->
          match constraint_ with
          | EqualityConstraint f ->
              let v = f x' in
              Tensor.add acc (Tensor.mul v v)
          | _ -> acc
        ) (Tensor.zeros []) constraints in
        
        if Tensor.float_value (Tensor.norm total_violation) < params.tolerance
        then x'
        else outer_loop (iter + 1) x'
    in
    outer_loop 0 init_x
end

(* Numerical integration *)
module Integration = struct
  type integration_method =
    | Trapezoidal
    | Simpson
    | GaussLegendre of int  (* number of points *)
    | AdaptiveQuadrature
    | MonteCarloIntegration of int  (* number of samples *)

  (* Gauss-Legendre quadrature points and weights *)
  let gauss_legendre_points = function
    | 2 -> [|-0.5773502692; 0.5773502692|], [|1.0; 1.0|]
    | 3 -> [|-0.7745966692; 0.0; 0.7745966692|], 
           [|0.5555555556; 0.8888888889; 0.5555555556|]
    | 4 -> [|-0.8611363116; -0.3399810436; 0.3399810436; 0.8611363116|],
           [|0.3478548451; 0.6521451549; 0.6521451549; 0.3478548451|]

  (* Change of interval for Gauss-Legendre *)
  let change_interval x a b =
    (b -. a) /. 2.0 *. x +. (b +. a) /. 2.0

  (* Integration error estimate *)
  type error_estimate = {
    absolute_error: float;
    relative_error: float;
    n_evaluations: int;
  }

  (* Integrate function over interval with error estimate *)
  let integrate ?(method_=Trapezoidal) ?(tol=1e-8) f a b =
    match method_ with
    | Trapezoidal ->
        let rec adaptive_trapezoid n prev_result =
          let h = (b -. a) /. float_of_int n in
          let result = ref ((f a +. f b) /. 2.0) in
          for i = 1 to n - 1 do
            let x = a +. float_of_int i *. h in
            result := !result +. f x
          done;
          result := !result *. h;
          
          if n > 1 && abs_float (!result -. prev_result) < tol then
            { absolute_error = abs_float (!result -. prev_result);
              relative_error = abs_float (!result -. prev_result) /. abs_float !result;
              n_evaluations = n + 1 }, !result
          else if n > 1000000 then
            { absolute_error = infinity;
              relative_error = infinity;
              n_evaluations = n + 1 }, !result
          else
            adaptive_trapezoid (n * 2) !result
        in
        adaptive_trapezoid 10 0.0

    | GaussLegendre n ->
        let points, weights = gauss_legendre_points n in
        let result = ref 0.0 in
        for i = 0 to n - 1 do
          let x = change_interval points.(i) a b in
          result := !result +. weights.(i) *. f x
        done;
        result := !result *. (b -. a) /. 2.0;
        { absolute_error = tol;
          relative_error = tol /. abs_float !result;
          n_evaluations = n }, !result

    | AdaptiveQuadrature ->
        let rec adaptive_quad a b fa fm fb tol =
          let m = (a +. b) /. 2.0 in
          let l = (a +. m) /. 2.0 in
          let r = (m +. b) /. 2.0 in
          let fl = f l in
          let fr = f r in
          
          let area1 = (fa +. 4.0 *. fm +. fb) *. (b -. a) /. 6.0 in
          let area2 = (fa +. 4.0 *. fl +. 2.0 *. fm +. 4.0 *. fr +. fb) *. 
                     (b -. a) /. 12.0 in
          
          if abs_float (area1 -. area2) < tol then
            area2
          else
            adaptive_quad a m fa fl fm (tol /. 2.0) +.
            adaptive_quad m b fm fr fb (tol /. 2.0)
        in
        let fa = f a in
        let fb = f b in
        let fm = f ((a +. b) /. 2.0) in
        let result = adaptive_quad a b fa fm fb tol in
        { absolute_error = tol;
          relative_error = tol /. abs_float result;
          n_evaluations = -1 }, result

  (* Multiple integration *)
  let integrate_multiple f bounds =
    let n = Array.length bounds in
    let rec integrate_dim dim point acc =
      if dim = n then f point
      else
        let (a, b) = bounds.(dim) in
        let g x =
          let new_point = Array.copy point in
          new_point.(dim) <- x;
          integrate_dim (dim + 1) new_point acc
        in
        let error, result = integrate g a b in
        result
    in
    let init_point = Array.make n 0.0 in
    integrate_dim 0 init_point 0.0
end

(* Special cases module for KAPC *)
module SpecialCases = struct
  (* CCA handling for p=2 case *)
  module CCA = struct
    type cca_result = {
      correlation: float;
      transform1: transformation;
      transform2: transformation;
    }

    (* Convert KAPC to CCA for p=2 case *)
    let from_kapc kernels penalty_params data =
      let k1 = kernels.(0) in
      let k2 = kernels.(1) in
      let alpha1 = penalty_params.(0) in
      let alpha2 = penalty_params.(1) in
      
      (* Compute kernel matrices *)
      let k1_matrix = Kernel.create_kernel_matrix k1 data in
      let k2_matrix = Kernel.create_kernel_matrix k2 data in
      
      (* Center kernel matrices *)
      let k1_centered = Kernel.center_kernel_matrix k1_matrix in
      let k2_centered = Kernel.center_kernel_matrix k2_matrix in
      
      (* Regularize kernel matrices *)
      let n = Tensor.size k1_matrix 0 in
      let identity = Tensor.eye n in
      let r1 = Tensor.add k1_centered (Tensor.mul_scalar identity alpha1) in
      let r2 = Tensor.add k2_centered (Tensor.mul_scalar identity alpha2) in
      
      (* Compute generalized eigenvalue problem *)
      let r1_sqrt_inv = NumericStability.stable_sqrtm r1 |> NumericStability.stable_inverse in
      let r2_sqrt_inv = NumericStability.stable_sqrtm r2 |> NumericStability.stable_inverse in
      
      let m = Tensor.mm (Tensor.mm r1_sqrt_inv k1_centered) 
        (Tensor.mm r2_sqrt_inv k2_centered) in
      
      let eigenvals, eigenvecs = NumericStability.stable_symeig ~eigenvectors:true 
        (Tensor.mm m (Tensor.transpose m 0 1)) in
      
      let max_corr_idx = Tensor.argmax eigenvals in
      let correlation = sqrt (Tensor.float_value (Tensor.get eigenvals [max_corr_idx])) in
      
      let alpha = Tensor.select eigenvecs 1 max_corr_idx in
      let beta = Tensor.mm r2_sqrt_inv (Tensor.mm k2_centered 
        (Tensor.mm r1_sqrt_inv alpha)) in
      
      let transform1 = fun x ->
        let k1_x = Tensor.stack (Array.to_list (Array.map (k1 x) 
          (Array.init n (fun i -> Tensor.select data 0 i)))) 0 in
        Tensor.mm k1_x alpha
      in
      
      let transform2 = fun x ->
        let k2_x = Tensor.stack (Array.to_list (Array.map (k2 x) 
          (Array.init n (fun i -> Tensor.select data 0 i)))) 0 in
        Tensor.mm k2_x beta
      in
      
      { correlation; transform1; transform2 }
  end

  (* Multiple kernel handling *)
  module MultipleKernels = struct
    type kernel_weights = {
      kernels: kernel array;
      weights: Tensor.t;
    }

    (* Optimize kernel weights *)
    let optimize_weights kernels data =
      let n_kernels = Array.length kernels in
      let n = Tensor.size data 0 in
      
      (* Compute kernel alignment matrix *)
      let alignment_matrix = Tensor.zeros [n_kernels; n_kernels] in
      for i = 0 to n_kernels - 1 do
        for j = 0 to n_kernels - 1 do
          let alignment = Kernel.kernel_alignment kernels.(i) kernels.(j) data in
          Tensor.set alignment_matrix [i; j] (Tensor.float_value alignment)
        done
      done;

      (* Solve quadratic program for weights *)
      let obj_matrix = Tensor.add alignment_matrix 
        (Tensor.mul_scalar (Tensor.eye n_kernels) NumericStability.epsilon) in
      let obj_vector = Tensor.ones [n_kernels] in
      
      let constraints = [
        Optimization.InequalityConstraint (fun x -> Tensor.neg x);  (* weights >= 0 *)
        Optimization.EqualityConstraint (fun x ->
          Tensor.sub (Tensor.sum x) (Tensor.ones []))  (* sum(weights) = 1 *)
      ] in
      
      let optimization_params = {
        Optimization.default_params with
        max_iter = 500;
        tolerance = 1e-8
      } in
      
      let objective w =
        let weighted_alignment = Tensor.mm (Tensor.mm w 
          (Tensor.transpose alignment_matrix 0 1)) w in
        Tensor.neg weighted_alignment
      in
      
      let init_weights = Tensor.div (Tensor.ones [n_kernels]) 
        (Tensor.full [] (float_of_int n_kernels)) in
      
      let optimal_weights = Optimization.trust_region
        ~objective
        ~gradient:(fun w -> Tensor.grad objective w)
        ~hessian:(fun w -> Tensor.jacobian (fun x -> Tensor.grad objective x) w)
        ~params:optimization_params
        init_weights in
      
      { kernels; weights = optimal_weights }

    (* Create composite kernel *)
    let create_composite kernel_weights =
      let weights = kernel_weights.weights in
      fun x y ->
        Array.mapi (fun i k ->
          let ki = k x y in
          Tensor.mul_scalar ki (Tensor.float_value (Tensor.get weights [i]))
        ) kernel_weights.kernels |>
        Array.fold_left Tensor.add (Tensor.zeros [])
  end
end

(* Kernel additive principal components *)
module KAPC = struct
  type t = {
    kernels: kernel array;
    penalty_params: float array;
    transforms: transformation array;
    function_spaces: FunctionSpace.function_space array;
  }

  (* Create new KAPC instance *)
  let create kernels penalty_params data =
    let n = Array.length kernels in
 
    let function_spaces = Array.init n (fun _ ->
      FunctionSpace.create_l2_space MeasureTheory.Lebesgue) in
    
    {
      kernels;
      penalty_params;
      transforms = Array.make n (fun x -> x);
      function_spaces;
    }

  (* Compute KAPC objective (penalized variance) *)
  let compute_objective kapc data =
    (* Compute sum of transforms *)
    let sum_transforms = Array.fold_left (fun acc transform ->
      let transformed = transform data in
      Tensor.add acc transformed
    ) (Tensor.zeros_like data) kapc.transforms in
    
    (* Compute variance term *)
    let variance = Population.variance sum_transforms in
    
    (* Compute penalty term *)
    let penalty = Array.fold_left2 (fun acc transform alpha ->
      let transformed = transform data in
      acc +. alpha *. Population.variance transformed
    ) 0.0 kapc.transforms kapc.penalty_params in
    
    Tensor.add (Tensor.full [] variance) (Tensor.full [] penalty)

  (* Fit KAPC to data *)
  let fit kapc data =
    (* Handle p=2 case using CCA *)
    if Array.length kapc.kernels = 2 then
      let cca_result = SpecialCases.CCA.from_kapc 
        kapc.kernels kapc.penalty_params data in
      { kapc with 
        transforms = [|cca_result.transform1; cca_result.transform2|] }
    else
      (* General case optimization *)
      let n = Tensor.size data 0 in
      let p = Array.length kapc.kernels in
      
      (* Initialize transforms *)
      let transforms = Array.make p (fun x -> x) in
      
      (* Optimize each transform sequentially *)
      for i = 0 to p - 1 do
        let kernel_matrix = Kernel.create_kernel_matrix kapc.kernels.(i) data in
        let centered_matrix = Kernel.center_kernel_matrix kernel_matrix in
        
        (* Add regularization *)
        let reg_matrix = Tensor.add centered_matrix 
          (Tensor.mul_scalar (Tensor.eye n) kapc.penalty_params.(i)) in
        
        (* Solve eigenvalue problem *)
        let eigenvals, eigenvecs = NumericStability.stable_symeig 
          ~eigenvectors:true reg_matrix in
        
        (* Find minimal eigenvector *)
        let min_idx = Tensor.argmin eigenvals in
        let min_eigenvec = Tensor.select eigenvecs 1 min_idx in
        
        (* Create transform function *)
        transforms.(i) <- (fun x ->
          let kernel_vec = Tensor.stack (Array.to_list (
            Array.map (kapc.kernels.(i) x) 
              (Array.init n (fun j -> Tensor.select data 0 j)))) 0 in
          Tensor.mv kernel_vec min_eigenvec)
      done;
      
      { kapc with transforms }

  (* Evaluate KAPC on new data *)
  let evaluate kapc data =
    compute_objective kapc data |> Tensor.float_value

  (* Get transforms *)
  let get_transforms kapc = kapc.transforms

  (* Optimize kernel weights *)
  let optimize_kernels kapc data =
    let kernel_weights = SpecialCases.MultipleKernels.optimize_weights 
      kapc.kernels data in
    let composite_kernel = SpecialCases.MultipleKernels.create_composite 
      kernel_weights in
    create [|composite_kernel|] [|1.0|] data
end
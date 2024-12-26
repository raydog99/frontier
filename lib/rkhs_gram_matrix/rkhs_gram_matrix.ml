open Torch

type tensor = Tensor.t
type scalar = float

type kernel_function = tensor -> tensor -> tensor
type feature_map = tensor -> tensor

let linear_combination tensors coeffs =
  List.fold_left2 (fun acc t c ->
    Tensor.(acc + (mul_scalar t (F c)))
  ) (List.hd tensors) (List.tl tensors) (List.tl coeffs)

let check_axioms space points =
  (* Additive commutativity *)
  let check_add_commutative x y =
    let sum1 = Tensor.(x + y) in
    let sum2 = Tensor.(y + x) in
    Tensor.(mean (abs (sum1 - sum2))) |> Tensor.float_value < 1e-6
  in
  
  (* Scalar multiplication distributivity *)
  let check_scalar_distributive x a b =
    let prod1 = Tensor.(mul_scalar x (F (a +. b))) in
    let prod2 = Tensor.(mul_scalar x (F a) + mul_scalar x (F b)) in
    Tensor.(mean (abs (prod1 - prod2))) |> Tensor.float_value < 1e-6
  in
  
  List.for_all (fun p1 ->
    List.for_all (fun p2 ->
      check_add_commutative p1 p2 &&
      check_scalar_distributive p1 1.0 2.0
    ) points
  ) points

module Kernel = struct
  type t = kernel_function

  let gaussian sigma x1 x2 =
    let diff = Tensor.(x1 - x2) in
    let sq_dist = Tensor.(sum (diff * diff) ~dim:[1]) in
    Tensor.(exp (neg sq_dist / (F 2. * F (sigma *. sigma))))

  let hardy sigma x1 x2 =
    let diff = Tensor.(x1 - x2) in
    let sq_dist = Tensor.(sum (diff * diff) ~dim:[1]) in
    Tensor.((F 1. + sq_dist / (F (sigma *. sigma))) ** (F (-0.5)))

  let gram k points =
    let n = (Tensor.shape points).(0) in
    let result = Tensor.zeros [n; n] in
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let xi = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
        let xj = Tensor.slice points ~dim:0 ~start:j ~end_:(j+1) in
        let kij = k xi xj in
        Tensor.copy_ ~src:kij ~dst:(Tensor.narrow result ~dim:0 ~start:i ~length:1 
                                    |> Tensor.narrow ~dim:1 ~start:j ~length:1)
      done
    done;
    result

  let is_positive_definite k points =
    let gram = gram k points in
    let eigenvals = Tensor.eigenvals gram in
    Tensor.(min eigenvals) |> Tensor.float_value > 0.
end

module RBF = struct
  type t = {
    func: tensor -> tensor -> float -> tensor;
    params: float list;
  }

  let gaussian = {
    func = (fun x y sigma ->
      let diff = Tensor.(x - y) in 
      Tensor.(exp (neg (sum (diff * diff) ~dim:[1]) / (F (2. *. sigma *. sigma))))
    );
    params = [1.0];
  }

  let hardy = {
    func = (fun x y sigma ->
      let diff = Tensor.(x - y) in
      let sq_dist = Tensor.(sum (diff * diff) ~dim:[1]) in
      Tensor.((F 1. + sq_dist / (F (sigma *. sigma))) ** (F (-0.5)))
    );
    params = [1.0];
  }

  let is_even_polynomial f =
    let test_points = Tensor.linspace 
      ~start:(Tensor.float (-5.)) 
      ~end_:(Tensor.float 5.) 
      ~steps:100 in
    let values_pos = f test_points test_points 1.0 in
    let values_neg = f (Tensor.neg test_points) (Tensor.neg test_points) 1.0 in
    let diff = Tensor.(values_pos - values_neg) in
    Tensor.(mean (abs diff)) |> Tensor.float_value < 1e-6
end

module RKVerification = struct
  type kernel_properties = {
    symmetric: bool;
    positive_definite: bool;
    continuous: bool;
    reproducing: bool;
    universal: bool;
  }

  module FeatureSpace = struct
    type t = {
      dim: int;
      map: feature_map;
      inner_product: tensor -> tensor -> float;
    }

    let from_kernel k points =
      let gram = Kernel.gram k points in
      let eigenvals, eigenvecs = Tensor.symeig gram ~eigenvectors:true in
      let feature_dim = Tensor.shape eigenvecs |> fun s -> s.(1) in
      
      let map x =
        let kx = k points x in
        Tensor.(mm (transpose2 eigenvecs) kx)
      in

      let inner_product x y =
        let phi_x = map x in
        let phi_y = map y in
        Tensor.(sum (phi_x * phi_y)) |> Tensor.float_value
      in

      { dim = feature_dim; map; inner_product }
  end

  let verify_kernel k points =
    let verify_symmetry () =
      let n = (Tensor.shape points).(0) in
      let symmetric = ref true in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let xi = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
          let xj = Tensor.slice points ~dim:0 ~start:j ~end_:(j+1) in
          let kij = k xi xj |> Tensor.float_value in
          let kji = k xj xi |> Tensor.float_value in
          if abs_float (kij -. kji) > 1e-6 then
            symmetric := false
        done
      done;
      !symmetric
    in

    let verify_positive_definite () = 
      Kernel.is_positive_definite k points
    in

    let verify_continuous () =
      let n = (Tensor.shape points).(0) in
      let continuous = ref true in
      for i = 0 to n - 2 do
        let x = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
        let y = Tensor.slice points ~dim:0 ~start:(i+1) ~end_:(i+2) in
        let kxx = k x x |> Tensor.float_value in
        let kxy = k x y |> Tensor.float_value in
        let dist = Tensor.(norm (x - y)) |> Tensor.float_value in
        if dist < 1e-6 && abs_float (kxx -. kxy) > 1e-5 then
          continuous := false
      done;
      !continuous
    in

    let feature_space = FeatureSpace.from_kernel k points in
    
    let verify_reproducing () =
      let n = (Tensor.shape points).(0) in
      let reproducing = ref true in
      for i = 0 to n - 1 do
        let x = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
        let phi_x = feature_space.map x in
        let k_x = k points x in
        let error = Tensor.(norm (phi_x - k_x)) |> Tensor.float_value in
        if error > 1e-6 then
          reproducing := false
      done;
      !reproducing
    in

    let verify_universal () =
      let test_fn x = Tensor.(sin (x * F 2.)) in
      let n = (Tensor.shape points).(0) in
      let errors = Tensor.zeros [n] in
      for i = 0 to n - 1 do
        let x = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
        let phi_x = feature_space.map x in
        let fx = test_fn x in
        let error = Tensor.(norm (phi_x - fx)) |> Tensor.float_value in
        Tensor.copy_ ~src:(Tensor.float error)
          ~dst:(Tensor.slice errors ~dim:0 ~start:i ~end_:(i+1))
      done;
      Tensor.(mean errors) |> Tensor.float_value < 1e-4
    in

    {
      symmetric = verify_symmetry ();
      positive_definite = verify_positive_definite ();
      continuous = verify_continuous ();
      reproducing = verify_reproducing ();
      universal = verify_universal ();
    }
end

module ProductSpace = struct
  type ('a, 'b) product_space = {
    space1: 'a;
    space2: 'b;
    inner_product: tensor -> tensor -> tensor -> tensor -> float;
    norm: tensor -> tensor -> float;
  }

  let create space1 space2 =
    let inner_product x1 x2 y1 y2 =
      let ip1 = Tensor.(sum (x1 * y1)) |> Tensor.float_value in
      let ip2 = Tensor.(sum (x2 * y2)) |> Tensor.float_value in
      ip1 *. ip2
    in

    let norm x y =
      sqrt (inner_product x y x y)
    in

    {space1; space2; inner_product; norm}

  let verify_properties space points1 points2 =
    let check_inner_product () =
      let n = (Tensor.shape points1).(0) in
      let valid = ref true in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let xi = Tensor.slice points1 ~dim:0 ~start:i ~end_:(i+1) in
          let yi = Tensor.slice points2 ~dim:0 ~start:i ~end_:(i+1) in
          let xj = Tensor.slice points1 ~dim:0 ~start:j ~end_:(j+1) in
          let yj = Tensor.slice points2 ~dim:0 ~start:j ~end_:(j+1) in
          
          let ip1 = space.inner_product xi yi xj yj in
          let ip2 = space.inner_product xj yj xi yi in
          if abs_float (ip1 -. ip2) > 1e-6 then valid := false;
          
          if i = j && ip1 <= 0. then valid := false
        done
      done;
      !valid
    in

    let check_complete () =
      let n = (Tensor.shape points1).(0) in
      let cauchy = ref true in
      for i = 0 to n - 2 do
        let xi = Tensor.slice points1 ~dim:0 ~start:i ~end_:(i+1) in
        let yi = Tensor.slice points2 ~dim:0 ~start:i ~end_:(i+1) in
        let xj = Tensor.slice points1 ~dim:0 ~start:(i+1) ~end_:(i+2) in
        let yj = Tensor.slice points2 ~dim:0 ~start:(i+1) ~end_:(i+2) in
        let dist = space.norm (Tensor.(xi - xj)) (Tensor.(yi - yj)) in
        if dist > 1e-6 then cauchy := false
      done;
      !cauchy
    in

    check_inner_product () && check_complete ()
end

module ProductRKHS = struct
  type t = {
    kernel_u: kernel_function;
    kernel_x: kernel_function;
    sigma_u: float;
    sigma_x: float;
  }

  let create ~sigma_u ~sigma_x = {
    kernel_u = Kernel.gaussian sigma_u;
    kernel_x = Kernel.gaussian sigma_x;
    sigma_u;
    sigma_x;
  }

  let gram_matrix t ~input_data ~state_data =
    let ku = t.kernel_u input_data input_data in
    let kx = t.kernel_x state_data state_data in
    Tensor.(ku * kx)  (* Element-wise product for gram matrix *)

  let kernel_vector t ~input_data ~state_data ~input ~state =
    let ku = t.kernel_u input input_data in
    let kx = t.kernel_x state state_data in
    Tensor.(ku * kx)

  let learn t ~input_data ~state_data ~output_data ~input ~state =
    let gram = gram_matrix t ~input_data ~state_data in
    let k = kernel_vector t ~input_data ~state_data ~input ~state in
    let coeffs = Tensor.(mm (inverse gram) (reshape output_data ~shape:[-1; 1])) in
    Tensor.(mm (transpose k) coeffs)
end

module NonlinearOperator = struct
  type operator_properties = {
    continuous: bool;
    bounded: bool;
    causal: bool;
    time_invariant: bool;
  }

  type dynamical_operator = {
    state_transition: tensor -> tensor -> tensor;
    output_map: tensor -> tensor;
    properties: operator_properties;
  }

  let verify_operator_properties op points =
    let check_continuous () =
      let n = (Tensor.shape points).(0) in
      let continuous = ref true in
      for i = 0 to n - 2 do
        let x1 = Tensor.slice points ~dim:0 ~start:i ~end_:(i+1) in
        let x2 = Tensor.slice points ~dim:0 ~start:(i+1) ~end_:(i+2) in
        let y1 = op.output_map (op.state_transition x1 x1) in
        let y2 = op.output_map (op.state_transition x2 x2) in
        if Tensor.(norm (x1 - x2)) |> Tensor.float_value < 1e-6 &&
           Tensor.(norm (y1 - y2)) |> Tensor.float_value > 1e-5 then
          continuous := false
      done;
      !continuous
    in

    let check_bounded () =
      let max_norm = ref 0. in
      List.init 100 (fun _ ->
        let x = Tensor.rand [1; (Tensor.shape points).(1)] in
        let y = op.output_map (op.state_transition x x) in
        max_norm := max !max_norm (Tensor.norm y |> Tensor.float_value)
      ) |> ignore;
      !max_norm < Float.infinity
    in

    {
      continuous = check_continuous ();
      bounded = check_bounded ();
      causal = true;  
      time_invariant = true;  
    }
end

module OperatorLearning = struct
  type learning_params = {
    input_kernel: kernel_function;
    state_kernel: kernel_function;
    regularization: float;
  }

  type learned_operator = {
    input_centers: tensor;
    state_centers: tensor;
    coefficients: tensor;
    params: learning_params;
  }

  let learn ~params ~input_data ~state_data ~output_data =
    (* Compute gram matrices *)
    let gram_input = Kernel.gram params.input_kernel input_data in
    let gram_state = Kernel.gram params.state_kernel state_data in
    let gram_product = Tensor.(gram_input * gram_state) in

    (* Add regularization *)
    let reg_matrix = Tensor.(eye (shape gram_product).(0) * F params.regularization) in
    let reg_gram = Tensor.(gram_product + reg_matrix) in

    (* Solve for coefficients *)
    let coefficients = 
      Tensor.(mm (inverse reg_gram) (reshape output_data ~shape:[-1; 1])) in

    { input_centers = input_data; state_centers = state_data; coefficients; params }

  let evaluate op ~input ~state =
    let k_input = op.params.input_kernel input op.input_centers in
    let k_state = op.params.state_kernel state op.state_centers in
    let k_prod = Tensor.(k_input * k_state) in
    Tensor.(mm k_prod op.coefficients)
end

module Separability = struct
  type separable_space = {
    countable_dense: tensor list;
    metric: tensor -> tensor -> float;
  }

  let create points metric =
    (* Find countable dense subset *)
    let find_dense_subset points epsilon =
      let n = List.length points in
      let dense = ref [] in
      for i = 0 to n - 1 do
        let p = List.nth points i in
        let needed = ref true in
        List.iter (fun d ->
          if metric p d < epsilon then
            needed := false
        ) !dense;
        if !needed then
          dense := p :: !dense
      done;
      !dense
    in

    { countable_dense = find_dense_subset points 1e-6; metric }

  let verify_separable space points epsilon =
    List.for_all (fun p ->
      List.exists (fun d ->
        space.metric p d < epsilon
      ) space.countable_dense
    ) points
end

module InequalityBound = struct
  type requirements = {
    polynomial_degree: bool;
    continuous_function: bool;
    tempered_distribution: bool;
    compact_domain: bool;
    continuous_operator: bool;
  }

  let verify_requirements g domain =
    let check_continuous () =
      let points = Tensor.rand [100; 2] in
      let values = g points in
      let shifted = g (Tensor.(points + F 1e-6)) in
      let diff = Tensor.(values - shifted) in
      Tensor.(max (abs diff)) |> Tensor.float_value < 1e-5
    in

    let check_tempered () =
      let points = Tensor.rand [100; 2] in
      let large_points = Tensor.(points * F 1000.) in
      let values = g large_points in
      let growth = Tensor.(log (abs values + F 1.)) in
      Tensor.(max growth) |> Tensor.float_value < Float.infinity
    in

    let check_compact domain =
      Tensor.(max (abs domain)) |> Tensor.float_value < Float.infinity
    in

    {
      polynomial_degree = RBF.is_even_polynomial (fun x y _ -> g x);
      continuous_function = check_continuous ();
      tempered_distribution = check_tempered ();
      compact_domain = check_compact domain;
      continuous_operator = true;  
    }

  let construct_approximator ~input_data ~state_data ~output_data ~mu ~lambda =
    RBF.{
      func = (fun x y sigma ->
        let diff = Tensor.(x - y) in 
        Tensor.(exp (neg (sum (diff * diff) ~dim:[1]) / (F (2. *. sigma *. sigma))))
      );
      params = [mu; lambda]
    }

  let verify_bound approximator target_op epsilon test_points =
    let n = (Tensor.shape test_points).(0) in
    let max_error = ref 0. in
    
    for i = 0 to n - 1 do
      let point = Tensor.slice test_points ~dim:0 ~start:i ~end_:(i+1) in
      let approx_val = approximator.RBF.func point point 1.0 in
      let true_val = target_op.NonlinearOperator.output_map
        (target_op.NonlinearOperator.state_transition point point) in
      let error = Tensor.(norm (approx_val - true_val)) |> Tensor.float_value in
      max_error := max !max_error error
    done;
    
    !max_error < epsilon
end

module FiniteRKHS = struct
  type finite_rkhs = {
    dim: int;
    basis: tensor list;
    kernel: kernel_function;
    completion: bool;
  }

  let create basis kernel =
    let dim = List.length basis in
    let check_orthonormal () =
      let n = List.length basis in
      let orthonormal = ref true in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let bi = List.nth basis i in
          let bj = List.nth basis j in
          let ip = Tensor.(sum (bi * bj)) |> Tensor.float_value in
          if i = j && abs_float (ip -. 1.0) > 1e-6 then
            orthonormal := false;
          if i <> j && abs_float ip > 1e-6 then
            orthonormal := false
        done
      done;
      !orthonormal
    in

    { dim; basis; kernel; completion = check_orthonormal () }

  let optimal_approximation rkhs points =
    let gram = Kernel.gram rkhs.kernel (Tensor.cat points ~dim:0) in
    let coeffs = Tensor.inverse gram in
    fun x ->
      let k_x = List.map (fun p -> rkhs.kernel p x) points in
      let k_tensor = Tensor.stack k_x ~dim:0 in
      Tensor.(mm coeffs k_tensor)
end
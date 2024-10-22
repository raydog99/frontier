open Torch

type riemannian_metric = {
  metric_tensor: Tensor.t -> Tensor.t;   (* G(θ) *)
  inverse_metric: Tensor.t -> Tensor.t;  (* G(θ)^{-1} *)
  det_metric: Tensor.t -> float;         (* det G(θ) *)
}

type connection = {
  christoffel: Tensor.t -> Tensor.t array array array;  (* Γijk *)
  dual_connection: Tensor.t -> Tensor.t array array array; (* Γ*ijk *)
  torsion: Tensor.t -> Tensor.t array array array;      (* T(X,Y) *)
}

type statistical_manifold = {
  dimension: int;
  metric: riemannian_metric;
  connection: connection;
  curvature: Tensor.t -> float;
}

let create_statistical_manifold dim =
  (* Fisher metric implementation *)
  let metric_tensor point =
    let hessian = Tensor.hessian point in
    Tensor.positive_definite_projection hessian
  in

  let inverse_metric point =
    let metric = metric_tensor point in
    Tensor.inverse metric
  in

  let det_metric point =
    let metric = metric_tensor point in
    Tensor.det metric |> Tensor.float_value
  in

  (* Connection computations *)
  let compute_christoffel point =
    let n = dim in
    let g = metric_tensor point in
    let g_inv = inverse_metric point in
    
    let gamma = Array.make_matrix n n [||] in
    let gamma_dual = Array.make_matrix n n [||] in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        gamma.(i).(j) <- Array.make n (Tensor.zeros [1]);
        gamma_dual.(i).(j) <- Array.make n (Tensor.zeros [1]);
        
        for k = 0 to n - 1 do
          (* Compute standard Christoffel symbols *)
          let std_symbol = ref 0. in
          for m = 0 to n - 1 do
            let g_im = Tensor.get g_inv [|i; m|] in
            let d_gjk = Tensor.partial_derivative g [|j; k|] m in
            std_symbol := !std_symbol +. 
              (Tensor.float_value g_im *. 
               Tensor.float_value d_gjk)
          done;
          
          gamma.(i).(j).(k) <- Tensor.scalar (!std_symbol /. 2.);
          
          (* Compute dual connection *)
          let dual_symbol = ref 0. in
          for m = 0 to n - 1 do
            let g_im = Tensor.get g_inv [|i; m|] in
            let d_gjk = Tensor.partial_derivative g [|k; j|] m in
            dual_symbol := !dual_symbol +. 
              (Tensor.float_value g_im *. 
               Tensor.float_value d_gjk)
          done;
          
          gamma_dual.(i).(j).(k) <- 
            Tensor.scalar (!dual_symbol /. 2.)
        done
      done
    done;
    
    gamma, gamma_dual
  in

  let christoffel point =
    let gamma, _ = compute_christoffel point in
    gamma
  in

  let dual_connection point =
    let _, gamma_dual = compute_christoffel point in
    gamma_dual
  in

  let torsion point =
    let gamma = christoffel point in
    let n = dim in
    let tor = Array.make_matrix n n [||] in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        tor.(i).(j) <- Array.make n (Tensor.zeros [1]);
        for k = 0 to n - 1 do
          let torsion_comp = 
            Tensor.sub
              (Tensor.add gamma.(i).(j).(k) gamma.(j).(k).(i))
              gamma.(k).(i).(j) in
          tor.(i).(j).(k) <- torsion_comp
        done
      done
    done;
    tor
  in

  (* Curvature computation *)
  let compute_curvature point =
    let gamma = christoffel point in
    let g = metric_tensor point in
    let n = dim in
    let riemann = ref 0. in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        for k = 0 to n - 1 do
          for l = 0 to n - 1 do
            let r_ijkl = ref 0. in
            
            (* Compute Riemann tensor components *)
            for m = 0 to n - 1 do
              r_ijkl := !r_ijkl +.
                Tensor.float_value (
                  Tensor.get gamma.(i).(j).(m)
                ) *.
                Tensor.float_value (
                  Tensor.get gamma.(m).(k).(l)
                )
            done;
            
            riemann := !riemann +. !r_ijkl *.
              Tensor.float_value (Tensor.get g [|i; j|]) *.
              Tensor.float_value (Tensor.get g [|k; l|])
          done
        done
      done
    done;
    
    !riemann /. (float_of_int (n * (n - 1)))
  in

  {
    dimension = dim;
    metric = {
      metric_tensor;
      inverse_metric;
      det_metric;
    };
    connection = {
      christoffel;
      dual_connection;
      torsion;
    };
    curvature = compute_curvature;
  }

(* Parallel transport along geodesics *)
let parallel_transport manifold start_point end_point vector =
  let steps = 100 in
  let dt = 1. /. float_of_int steps in
  
  let transported = ref vector in
  let current = ref start_point in
  
  for step = 0 to steps - 1 do
    let t = float_of_int step *. dt in
    
    (* Update position along geodesic *)
    let velocity = Tensor.sub end_point start_point in
    current := Tensor.add start_point 
      (Tensor.mul_scalar velocity t);
    
    (* Compute connection coefficients *)
    let gamma = manifold.connection.christoffel !current in
    
    (* Transport vector using connection *)
    let correction = Array.init manifold.dimension (fun i ->
      Array.init manifold.dimension (fun j ->
        Array.init manifold.dimension (fun k ->
          let coeff = gamma.(i).(j).(k) in
          Tensor.mul coeff 
            (Tensor.mul velocity !transported)
        ) |> Array.fold_left Tensor.add 
            (Tensor.zeros [manifold.dimension])
      ) |> Array.fold_left Tensor.add 
          (Tensor.zeros [manifold.dimension])
    ) |> Array.fold_left Tensor.add 
        (Tensor.zeros [manifold.dimension]) in
    
    transported := Tensor.sub !transported 
      (Tensor.mul_scalar correction dt)
  done;
  
  !transported
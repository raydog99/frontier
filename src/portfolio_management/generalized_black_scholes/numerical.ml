open Utils
open Torch

module FiniteDifference = struct
  type scheme_matrices = {
    a: Tensor.t;
    b: Tensor.t;
    psi: Tensor.t;
  }

  let build_matrices params grid measure =
    let n = Grid.size grid - 2 in
    let h = Grid.delta grid in
    let points = Grid.interior_points grid in
    
    let a = Tensor.zeros [n; n] in
    let psi = Tensor.zeros [n; n] in
    let b = Tensor.zeros [n; 1] in
    
    (* Build matrices *)
    for i = 0 to n-1 do
      let k = float_of_int (i + 1) in
      let x = points.(i) in
      
      (* A matrix - diagonal terms *)
      let diag = -.(params.sigma *. params.sigma *. k *. k) -. params.r in
      Tensor.set a [i; i] diag;
      
      (* A matrix - off-diagonal terms *)
      if i > 0 then
        let lower = params.sigma *. params.sigma *. k *. k /. 2.0 -. 
                   params.r *. k /. 2.0 in
        Tensor.set a [i; i-1] lower;
      
      if i < n-1 then
        let upper = params.sigma *. params.sigma *. k *. k /. 2.0 +. 
                   params.r *. k /. 2.0 in
        Tensor.set a [i; i+1] upper;
      
      (* Ψ matrix *)
      let psi_val = Measure.integrate_psi measure (i+1) k in
      Tensor.set psi [i; i] psi_val;
      
      (* B matrix *)
      Tensor.set b [i; 0] (float_of_int (1 lsl (i+1)) *. params.r)
    done;
    
    {a; b; psi}

  let solve params matrices initial_condition dt n_steps =
    let n = Grid.size params.grid in
    let solution = Tensor.zeros [n_steps; n] in
    
    (* Set initial condition *)
    for i = 0 to n-1 do
      Tensor.set solution [0; i] initial_condition.(i)
    done;
    
    (* Time stepping with implicit scheme *)
    for t = 1 to n_steps-1 do
      let prev = Tensor.slice solution [t-1; -1] in
      
      (* Full system solve *)
      let system = Tensor.add 
        (Tensor.eye (n-2))
        (Tensor.matmul
           (Tensor.matmul matrices.psi matrices.a)
           (Tensor.mul_scalar (Tensor.eye (n-2)) dt)) in
      
      let rhs = Tensor.add prev
        (Tensor.mul_scalar matrices.b dt) in
      
      let next = Tensor.solve system rhs in
      
      (* Update solution *)
      for i = 0 to n-1 do
        Tensor.set solution [t; i] (Tensor.get next [i])
      done
    done;
    solution
end

module Validation = struct
  type error_stats = {
    l2_error: float;
    max_error: float;
    convergence_rate: float;
  }

  let analyze_error solution exact_solution grid =
    let n = Grid.size grid in
    let h = Grid.delta grid in
    let l2_error = ref 0.0 in
    let max_error = ref 0.0 in
    
    for i = 0 to n-1 do
      let err = abs_float (Tensor.get solution [solution.shape.!(0)-1; i] -. 
                          exact_solution.(i)) in
      l2_error := !l2_error +. err *. err *. h;
      max_error := max !max_error err
    done;
    
    {
      l2_error = sqrt !l2_error;
      max_error = !max_error;
      convergence_rate = -. log !max_error /. log h;
    }

  let check_convergence stats tolerance =
    stats.l2_error < tolerance && stats.max_error < tolerance
end
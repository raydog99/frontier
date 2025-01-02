open Torch

module NumericalSchemes = struct
  type spatial_scheme = 
    | CentralDiff    (* Second-order central differences *)
    | UpwindDiff     (* First-order upwind *)
    | WENO of int    (* WENO scheme of specified order *)
    | ENO of int     (* ENO scheme of specified order *)

  type time_scheme =
    | ExplicitEuler
    | ImplicitEuler
    | CrankNicolson
    | RK4           (* Classical 4th order Runge-Kutta *)
    | IMEX of int   (* IMEX scheme of specified order *)

  type discretization_params = {
    spatial_scheme: spatial_scheme;
    time_scheme: time_scheme;
    dx: float array;     (* Spatial steps for each dimension *)
    dt: float;          (* Time step *)
    theta: float;       (* Parameter for theta-methods *)
  }

  let weno_reconstruction order flux_values =
    let k = (order + 1) / 2 in
    let n = Array.length flux_values in
    
    let beta = Array.make k 0.0 in
    for r = 0 to k-1 do
      let mut_sum = ref 0.0 in
      for l = 1 to k do
        let deriv = Array.init l (fun i ->
          let idx = r + i in
          if idx >= n then flux_values.(n-1)
          else flux_values.(idx)) in
        mut_sum := !mut_sum +. 
          (Float.pow (derivative_norm deriv) 2.0)
      done;
      beta.(r) <- !mut_sum
    done;
    
    let d = match order with
      | 5 -> [|0.1; 0.6; 0.3|]
      | 3 -> [|0.3; 0.7|]
      | _ -> [|1.0|] in
      
    let epsilon = 1e-6 in
    let alpha = Array.mapi (fun i b ->
      d.(i) /. ((b +. epsilon) ** 2.0)) beta in
    let alpha_sum = Array.fold_left (+.) 0.0 alpha in
    let omega = Array.map (fun a -> a /. alpha_sum) alpha in
    
    Array.fold_left2 (fun acc w f -> acc +. w *. f) 
      0.0 omega flux_values

  let eno_reconstruction order flux_values =
    let k = order + 1 in
    let n = Array.length flux_values in
    
    let divided_diff = Array.make_matrix k n 0.0 in
    for i = 0 to n-1 do
      divided_diff.(0).(i) <- flux_values.(i)
    done;
    
    for j = 1 to k-1 do
      for i = 0 to n-j-1 do
        divided_diff.(j).(i) <- 
          (divided_diff.(j-1).(i+1) -. divided_diff.(j-1).(i))
      done
    done;
    
    let rec choose_stencil r i =
      if r = k then i
      else
        let left = abs_float divided_diff.(r).(i-1) in
        let right = abs_float divided_diff.(r).(i) in
        if left < right then
          choose_stencil (r+1) (i-1)
        else
          choose_stencil (r+1) i
    in
    
    let i0 = choose_stencil 1 (k-1) in
    
    let mut_sum = ref divided_diff.(0).(i0) in
    let mut_prod = ref 1.0 in
    for j = 1 to k-1 do
      mut_prod := !mut_prod *. (flux_values.(k-1) -. 
        flux_values.(i0+j-1));
      mut_sum := !mut_sum +. divided_diff.(j).(i0) *. !mut_prod
    done;
    !mut_sum

  let imex_step params explicit_op implicit_op solution dt =
    match params.time_scheme with
    | IMEX order ->
        begin match order with
        | 1 -> (* First order IMEX *)
            let f_expl = explicit_op solution in
            let f_impl = implicit_op solution in
            
            (* Solve implicit part *)
            let solver = LinearAlgebra.BlockTridiagonal.create
              ~lower:(Array.make (Tensor.size solution 0 - 1) 
                (Tensor.eye 1))
              ~diag:(Array.make (Tensor.size solution 0)
                (Tensor.add (Tensor.eye 1)
                  (Tensor.mul f_impl (Tensor.of_float dt))))
              ~upper:(Array.make (Tensor.size solution 0 - 1)
                (Tensor.eye 1)) in
                
            let rhs = Tensor.add solution
              (Tensor.mul f_expl (Tensor.of_float dt)) in
              
            LinearAlgebra.BlockTridiagonal.solve solver rhs
            
        | 2 -> (* Second order IMEX *)
            let gamma = (2.0 -. sqrt 2.0) /. 2.0 in
            
            (* First stage *)
            let f_expl1 = explicit_op solution in
            let f_impl1 = implicit_op solution in
            
            let solver1 = LinearAlgebra.BlockTridiagonal.create
              ~lower:(Array.make (Tensor.size solution 0 - 1)
                (Tensor.eye 1))
              ~diag:(Array.make (Tensor.size solution 0)
                (Tensor.add (Tensor.eye 1)
                  (Tensor.mul f_impl1 
                    (Tensor.of_float (gamma *. dt)))))
              ~upper:(Array.make (Tensor.size solution 0 - 1)
                (Tensor.eye 1)) in
                
            let rhs1 = Tensor.add solution
              (Tensor.mul f_expl1 
                (Tensor.of_float (gamma *. dt))) in
                
            let u1 = LinearAlgebra.BlockTridiagonal.solve 
              solver1 rhs1 in
              
            (* Second stage *)
            let f_expl2 = explicit_op u1 in
            let f_impl2 = implicit_op u1 in
            
            let solver2 = LinearAlgebra.BlockTridiagonal.create
              ~lower:(Array.make (Tensor.size solution 0 - 1)
                (Tensor.eye 1))
              ~diag:(Array.make (Tensor.size solution 0)
                (Tensor.add (Tensor.eye 1)
                  (Tensor.mul f_impl2
                    (Tensor.of_float ((1.0 -. gamma) *. dt)))))
              ~upper:(Array.make (Tensor.size solution 0 - 1)
                (Tensor.eye 1)) in
                
            let rhs2 = Tensor.add u1
              (Tensor.mul f_expl2
                (Tensor.of_float ((1.0 -. gamma) *. dt))) in
                
            LinearAlgebra.BlockTridiagonal.solve solver2 rhs2
            
        | _ -> failwith "IMEX order not implemented"
        end
    | _ -> failwith "Not an IMEX scheme"

  let rk4_step params op solution dt =
    let k1 = op solution in
    
    let u2 = Tensor.add solution
      (Tensor.mul k1 (Tensor.of_float (dt /. 2.0))) in
    let k2 = op u2 in
    
    let u3 = Tensor.add solution
      (Tensor.mul k2 (Tensor.of_float (dt /. 2.0))) in
    let k3 = op u3 in
    
    let u4 = Tensor.add solution
      (Tensor.mul k3 (Tensor.of_float dt)) in
    let k4 = op u4 in
    
    Tensor.add solution
      (Tensor.mul
        (Tensor.add
          (Tensor.add k1 (Tensor.mul k2 (Tensor.of_float 2.0)))
          (Tensor.add 
            (Tensor.mul k3 (Tensor.of_float 2.0))
            k4))
        (Tensor.of_float (dt /. 6.0)))

  let solve params initial_condition t_final =
    let n_steps = int_of_float (t_final /. params.dt) in
    let solution = ref initial_condition in
    
    for _ = 1 to n_steps do
      solution := match params.time_scheme with
        | ExplicitEuler ->
            PDEFormulation.time_step `Explicit params !solution
        | ImplicitEuler ->
            PDEFormulation.time_step `Implicit params !solution
        | CrankNicolson ->
            PDEFormulation.time_step `CrankNicolson params !solution
        | RK4 ->
            rk4_step params 
              (PDEFormulation.apply_operator params) !solution params.dt
        | IMEX order ->
            let explicit_op = PDEFormulation.build_explicit_operator params in
            let implicit_op = PDEFormulation.build_implicit_operator params in
            imex_step params explicit_op implicit_op !solution params.dt
    done;
    
    !solution
end

module BoundaryHandler = struct
  type boundary_type =
    | Dirichlet of float  (* Fixed value *)
    | Neumann of float    (* Fixed derivative *)
    | Robin of {          (* Robin boundary condition *)
        alpha: float;
        beta: float;
        gamma: float;
      }
    | Periodic            (* Periodic boundary *)
    | Artificial of {     (* Artificial boundary *)
        order: int;
        extrapolation: bool;
      }

  let apply_boundary solution params boundary rates dim =
    let n = Tensor.size solution dim in
    
    match boundary with
    | Dirichlet value ->
        Tensor.narrow_copy_
          solution dim 0 1
          (Tensor.ones [1] |> Tensor.mul_scalar value);
        Tensor.narrow_copy_
          solution dim (n-1) 1
          (Tensor.ones [1] |> Tensor.mul_scalar value)
          
    | Neumann slope ->
        let dx = rates.(1) -. rates.(0) in
        Tensor.narrow_copy_
          solution dim 0 1
          (Tensor.sub
            (Tensor.narrow solution dim 1 1)
            (Tensor.mul_scalar (Tensor.ones [1]) (slope *. dx)));
            
        let dx = rates.(n-1) -. rates.(n-2) in
        Tensor.narrow_copy_
          solution dim (n-1) 1
          (Tensor.add
            (Tensor.narrow solution dim (n-2) 1)
            (Tensor.mul_scalar (Tensor.ones [1]) (slope *. dx)))
            
    | Robin {alpha; beta; gamma} ->
        (* Robin condition: alpha*u + beta*du/dx = gamma *)
        let dx = rates.(1) -. rates.(0) in
        Tensor.narrow_copy_
          solution dim 0 1
          (Tensor.div
            (Tensor.sub
              (Tensor.mul_scalar (Tensor.ones [1]) gamma)
              (Tensor.mul_scalar 
                (Tensor.narrow solution dim 1 1)
                (beta /. dx)))
            (Tensor.mul_scalar (Tensor.ones [1])
              (alpha -. beta /. dx)));
              
        let dx = rates.(n-1) -. rates.(n-2) in
        Tensor.narrow_copy_
          solution dim (n-1) 1
          (Tensor.div
            (Tensor.sub
              (Tensor.mul_scalar (Tensor.ones [1]) gamma)
              (Tensor.mul_scalar
                (Tensor.narrow solution dim (n-2) 1)
                (beta /. dx)))
            (Tensor.mul_scalar (Tensor.ones [1])
              (alpha +. beta /. dx)))
              
    | Periodic ->
        (* Enforce periodicity *)
        Tensor.narrow_copy_
          solution dim 0 1
          (Tensor.narrow solution dim (n-2) 1);
        Tensor.narrow_copy_
          solution dim (n-1) 1
          (Tensor.narrow solution dim 1 1)
          
    | Artificial {order; extrapolation} ->
        if extrapolation then
          (* High-order extrapolation *)
          let coeffs = match order with
            | 1 -> [|1.; -1.|]
            | 2 -> [|2.; -3.; 1.|]
            | 3 -> [|3.; -6.; 4.; -1.|]
            | _ -> failwith "Unsupported extrapolation order"
          in
          
          (* Lower boundary *)
          let mut_sum = ref 0.0 in
          for i = 1 to Array.length coeffs - 1 do
            mut_sum := !mut_sum +. coeffs.(i) *.
              Tensor.float_value
                (Tensor.narrow solution dim i 1)
          done;
          Tensor.narrow_copy_
            solution dim 0 1
            (Tensor.of_float (-. !mut_sum /. coeffs.(0)));
            
          (* Upper boundary *)
          let mut_sum = ref 0.0 in
          for i = 1 to Array.length coeffs - 1 do
            mut_sum := !mut_sum +. coeffs.(i) *.
              Tensor.float_value
                (Tensor.narrow solution dim (n-i-1) 1)
          done;
          Tensor.narrow_copy_
            solution dim (n-1) 1
            (Tensor.of_float (-. !mut_sum /. coeffs.(0)))
        else
          (* Use one-sided differences *)
          let stencil = match order with
            | 1 -> [|-1.; 1.|]
            | 2 -> [|-3./2.; 2.; -1./2.|]
            | 3 -> [|-11./6.; 3.; -3./2.; 1./3.|]
            | _ -> failwith "Unsupported difference order"
          in
          
          (* Apply to boundaries *)
          let apply_stencil idx =
            let mut_sum = ref 0.0 in
            Array.iteri (fun i coef ->
              mut_sum := !mut_sum +. coef *.
                Tensor.float_value
                  (Tensor.narrow solution dim (idx+i) 1)
            ) stencil;
            !mut_sum
          in
          
          Tensor.narrow_copy_
            solution dim 0 1
            (Tensor.of_float (apply_stencil 0));
          Tensor.narrow_copy_
            solution dim (n-1) 1
            (Tensor.of_float (apply_stencil (n-Array.length stencil)))

end

module MixedDerivatives = struct
  type mixed_approx =
    | CentralDiff       (* Standard central difference *)
    | FourthOrder      (* Fourth-order accurate *)
    | CompactStencil   (* Compact 9-point stencil *)
    | Upwind           (* Direction-biased stencil *)

  let compute_mixed_derivative approx_type solution rates i j =
    let ni = Tensor.size solution 0 in
    let nj = Tensor.size solution 1 in
    
    match approx_type with
    | CentralDiff ->
        let dxi = rates.(i+1) -. rates.(i-1) in
        let dxj = rates.(j+1) -. rates.(j-1) in
        
        let stencil = Array.make_matrix 3 3 0.0 in
        stencil.(0).(0) <- 1.0 /. (4.0 *. dxi *. dxj);
        stencil.(2).(2) <- 1.0 /. (4.0 *. dxi *. dxj);
        stencil.(0).(2) <- -1.0 /. (4.0 *. dxi *. dxj);
        stencil.(2).(0) <- -1.0 /. (4.0 *. dxi *. dxj);
        
        apply_stencil stencil solution i j

    | FourthOrder ->
        let dxi = rates.(i+1) -. rates.(i-1) in
        let dxj = rates.(j+1) -. rates.(j-1) in
        
        let stencil = Array.make_matrix 5 5 0.0 in
        
        stencil.(2).(2) <- 0.0;
        
        for di = -2 to 2 do
          for dj = -2 to 2 do
            if abs di + abs dj = 2 then
              stencil.(di+2).(dj+2) <- 
                (float_of_int (di * dj)) /. 
                  (12.0 *. dxi *. dxj)
          done
        done;
        
        apply_stencil stencil solution i j
        
    | CompactStencil ->
        let dxi = rates.(i+1) -. rates.(i) in
        let dxj = rates.(j+1) -. rates.(j) in
        
        let stencil = Array.make_matrix 3 3 0.0 in
        
        (* Center point *)
        stencil.(1).(1) <- 4.0 /. (dxi *. dxj);
        
        (* Edge points *)
        stencil.(0).(1) <- -1.0 /. (dxi *. dxj);
        stencil.(2).(1) <- -1.0 /. (dxi *. dxj);
        stencil.(1).(0) <- -1.0 /. (dxi *. dxj);
        stencil.(1).(2) <- -1.0 /. (dxi *. dxj);
        
        (* Corner points *)
        stencil.(0).(0) <- 0.25 /. (dxi *. dxj);
        stencil.(0).(2) <- 0.25 /. (dxi *. dxj);
        stencil.(2).(0) <- 0.25 /. (dxi *. dxj);
        stencil.(2).(2) <- 0.25 /. (dxi *. dxj);
        
        apply_stencil stencil solution i j
        
    | Upwind ->
        let dxi = rates.(i+1) -. rates.(i) in
        let dxj = rates.(j+1) -. rates.(j) in
        
        (* Get flow direction from coefficients *)
        let vi = Tensor.float_value 
          (Tensor.select (Tensor.select solution i) j) in
        let vj = Tensor.float_value
          (Tensor.select (Tensor.select solution j) i) in
          
        (* Select upwind or downwind differences *)
        let stencil = Array.make_matrix 3 3 0.0 in
        if vi >= 0.0 && vj >= 0.0 then begin
          stencil.(1).(1) <- 1.0 /. (dxi *. dxj);
          stencil.(0).(1) <- -1.0 /. (dxi *. dxj);
          stencil.(1).(0) <- -1.0 /. (dxi *. dxj);
          stencil.(0).(0) <- 1.0 /. (dxi *. dxj)
        end else if vi >= 0.0 && vj < 0.0 then begin
          stencil.(1).(1) <- 1.0 /. (dxi *. dxj);
          stencil.(0).(1) <- -1.0 /. (dxi *. dxj);
          stencil.(1).(2) <- -1.0 /. (dxi *. dxj);
          stencil.(0).(2) <- 1.0 /. (dxi *. dxj)
        end else if vi < 0.0 && vj >= 0.0 then begin
          stencil.(1).(1) <- 1.0 /. (dxi *. dxj);
          stencil.(2).(1) <- -1.0 /. (dxi *. dxj);
          stencil.(1).(0) <- -1.0 /. (dxi *. dxj);
          stencil.(2).(0) <- 1.0 /. (dxi *. dxj)
        end else begin
          stencil.(1).(1) <- 1.0 /. (dxi *. dxj);
          stencil.(2).(1) <- -1.0 /. (dxi *. dxj);
          stencil.(1).(2) <- -1.0 /. (dxi *. dxj);
          stencil.(2).(2) <- 1.0 /. (dxi *. dxj)
        end;
        
        apply_stencil stencil solution i j

  let apply_stencil stencil solution i j =
    let ni = Tensor.size solution 0 in
    let nj = Tensor.size solution 1 in
    let result = ref 0.0 in
    
    let si = Array.length stencil in
    let sj = Array.length stencil.(0) in
    let offset_i = si / 2 in
    let offset_j = sj / 2 in
    
    for di = 0 to si-1 do
      for dj = 0 to sj-1 do
        let ii = i + (di - offset_i) in
        let jj = j + (dj - offset_j) in
        
        if ii >= 0 && ii < ni && jj >= 0 && jj < nj then
          result := !result +. stencil.(di).(dj) *.
            Tensor.float_value
              (Tensor.select (Tensor.select solution ii) jj)
      done
    done;
    
    Tensor.of_float !result
    
  let build_operator params rates i j =
    let n = Array.length rates in
    let op = Tensor.zeros [n; n] in
    
    (* Interior points *)
    for k = 1 to n-2 do
      for l = 1 to n-2 do
        let stencil = match params.spatial_scheme with
          | CentralDiff -> [|[|1.; 0.; -1.|]; [|0.; 0.; 0.|]; [|-1.; 0.; 1.|]|]
          | _ -> failwith "Unsupported scheme for operator matrix"
        in
        
        let si = Array.length stencil in
        let sj = Array.length stencil.(0) in
        for di = 0 to si-1 do
          for dj = 0 to sj-1 do
            let ii = k + (di - 1) in
            let jj = l + (dj - 1) in
            if ii >= 0 && ii < n && jj >= 0 && jj < n then
              Tensor.copy_
                (Tensor.select (Tensor.select op k) l)
                (Tensor.add
                  (Tensor.select (Tensor.select op k) l)
                  (Tensor.of_float (stencil.(di).(dj) /. 
                    (4.0 *. (rates.(i+1) -. rates.(i)) *. 
                           (rates.(j+1) -. rates.(j))))))
          done
        done
      done
    done;
    
    op
end

module StabilityAnalysis = struct
  type stability_method =
    | VonNeumann         (* von Neumann stability analysis *)
    | MatrixStability    (* Matrix-based stability analysis *)
    | EnergyMethod       (* Energy method stability *)
    | CFLCondition       (* CFL condition checking *)

  type stability_params = {
    method_type: stability_method;
    max_time: float;
    space_steps: int array;
    time_steps: int;
    safety_factor: float;
  }

  let analyze_stability params solution =
    match params.method_type with
    | VonNeumann ->
        (* Fourier analysis of discretization *)
        let max_wavenumber = Array.map (fun n ->
          3.14159 /. float_of_int n) params.space_steps in
          
        let amplification_factors = ref [] in
        Array.iteri (fun dim n ->
          let dx = 1.0 /. float_of_int n in
          let dt = params.max_time /. float_of_int params.time_steps in
          
          (* Sample wavenumbers *)
          for k = 1 to 10 do
            let xi = float_of_int k *. max_wavenumber.(dim) /. 10.0 in
            
            (* Compute amplification factor *)
            let g = match params.spatial_scheme with
              | CentralDiff ->
                  1.0 -. 4.0 *. (dt /. (dx *. dx)) *. 
                    (sin (xi *. dx /. 2.0) ** 2.0)
              | _ -> failwith "Unsupported scheme for von Neumann analysis"
            in
            
            amplification_factors := g :: !amplification_factors
          done
        ) params.space_steps;
        
        (* Check max amplification factor *)
        let max_factor = List.fold_left max 
          (List.hd !amplification_factors)
          (List.tl !amplification_factors) in
          
        max_factor < 1.0 +. params.safety_factor
        
    | MatrixStability ->
        (* Analyze eigenvalues of discretization matrix *)
        let op = PDEFormulation.build_system_matrix 
          params solution in
          
        let eigenvals = Tensor.eig op in
        let max_eigenval = Tensor.max eigenvals in
        
        Tensor.float_value max_eigenval < 
          1.0 /. params.max_time +. params.safety_factor
        
    | EnergyMethod ->
        (* Energy-based stability estimate *)
        let energy = ref 0.0 in
        let prev_energy = ref 0.0 in
        
        let dt = params.max_time /. 
          float_of_int params.time_steps in
          
        (* Compute solution energy *)
        for t = 0 to params.time_steps do
          prev_energy := !energy;
          energy := 0.0;
          
          let sol_t = Tensor.narrow solution 0 t 1 in
          energy := Tensor.float_value
            (Tensor.dot sol_t sol_t)
        done;
        
        (* Check energy growth *)
        !energy <= (1.0 +. params.safety_factor) *. 
          !prev_energy
          
    | CFLCondition ->
        (* Check CFL condition for each dimension *)
        let stable = ref true in
        Array.iteri (fun dim n ->
          let dx = 1.0 /. float_of_int n in
          let dt = params.max_time /. 
            float_of_int params.time_steps in
            
          (* Get maximum wave speed *)
          let max_speed = ref 0.0 in
          let sol_t = Tensor.narrow solution 0 0 1 in
          for i = 0 to Tensor.size sol_t 0 - 1 do
            let speed = abs_float (Tensor.float_value
              (Tensor.narrow sol_t 0 i 1)) in
            max_speed := max !max_speed speed
          done;
          
          (* Check CFL number *)
          let cfl = !max_speed *. dt /. dx in
          stable := !stable && 
            (cfl <= 1.0 +. params.safety_factor)
        ) params.space_steps;
        
        !stable

  let suggest_parameters params solution =
    let stable = ref false in
    let dt = ref (params.max_time /. 
      float_of_int params.time_steps) in
    let dx = ref (Array.map (fun n ->
      1.0 /. float_of_int n) params.space_steps) in
      
    while not !stable do
      (* Try reducing timestep *)
      dt := !dt /. 2.0;
      
      (* Check stability *)
      stable := analyze_stability
        {params with 
          time_steps = int_of_float (params.max_time /. !dt)}
        solution;
        
      (* If still unstable, refine spatial grid *)
      if not !stable then
        dx := Array.map (fun d -> d /. 2.0) !dx
    done;
    
    !dt, !dx
end

module ForwardRateProperties = struct
  let check_martingale params rate_index measure t =
    let rate = Tensor.narrow params.initial_rates 0 rate_index 1 in
    let dt = 0.01 in  (* Small time step *)
    let n_steps = 100 in
    let paths = Array.init n_steps (fun _ -> Tensor.clone rate) in
    
    (* Simulate paths *)
    for i = 1 to n_steps-1 do
      let prev = paths.(i-1) in
      let drift, vol = Measures.transform_dynamics
        params (float_of_int i *. dt) prev measure Measures.RiskNeutral in
      paths.(i) <- match measure with
        | Measures.TForward maturity ->
            (* Should be martingale under T-forward measure *)
            let dW = Tensor.randn [1] in
            Tensor.add prev
              (Tensor.mul vol 
                (Tensor.mul dW (Tensor.of_float (sqrt dt))))
        | _ -> 
            (* Include drift under other measures *)
            let dW = Tensor.randn [1] in
            Tensor.add prev
              (Tensor.add
                (Tensor.mul drift (Tensor.of_float dt))
                (Tensor.mul vol 
                  (Tensor.mul dW (Tensor.of_float (sqrt dt)))))
    done;
    
    (* Check martingale property *)
    let final_mean = Array.fold_left
      (fun acc path -> acc +. Tensor.float_value path)
      0.0 paths in
    let initial = Tensor.float_value rate in
    abs_float (final_mean /. float_of_int n_steps -. initial) < 1e-3

  (* Forward measure change *)  
  let change_measure_dynamics params rate t from_measure to_measure =
    let drift, vol = Measures.transform_dynamics
      params t rate from_measure to_measure in
      
    match to_measure with
    | Measures.TForward maturity when maturity > t ->
        (* Forward measure drift adjustment *)
        let adj = Tensor.div vol
          (Tensor.add (Tensor.ones [1])
            (Tensor.mul (Tensor.of_float (maturity -. t)) rate)) in
        Tensor.add drift adj
    | _ -> drift
end

module CorrelationStructure = struct
  type correlation_config = {
    base_correlation: float;
    time_decay: float option;
    tenor_decay: float option;
    custom_correlation: (int -> int -> float) option;
  }

  let build_correlation config n_rates =
    let matrix = Tensor.zeros [n_rates; n_rates] in
    
    for i = 0 to n_rates-1 do
      for j = 0 to n_rates-1 do
        if i = j then
          Tensor.copy_
            (Tensor.select (Tensor.select matrix i) j)
            (Tensor.ones [1])
        else
          let corr = match config.custom_correlation with
            | Some f -> f i j
            | None ->
                let base = config.base_correlation in
                let time_factor = match config.time_decay with
                  | Some decay -> exp(-. decay *. float_of_int (abs (i - j)))
                  | None -> 1.0 in
                let tenor_factor = match config.tenor_decay with
                  | Some decay -> exp(-. decay *. float_of_int (abs (i - j)))
                  | None -> 1.0 in
                base *. time_factor *. tenor_factor
          in
          Tensor.copy_
            (Tensor.select (Tensor.select matrix i) j)
            (Tensor.of_float corr)
      done
    done;
    
    ensure_positive_definite matrix

  let ensure_positive_definite matrix =
    let n = Tensor.size matrix 0 in
    
    (* Compute eigendecomposition *)
    let e, v = Tensor.symeig matrix ~eigenvectors:true in
    
    (* Replace negative eigenvalues with small positive values *)
    let min_eigenval = 1e-6 in
    let e' = Tensor.max e 
      (Tensor.ones [n] |> Tensor.mul_scalar min_eigenval) in
      
    (* Reconstruct matrix *)
    let v_t = Tensor.transpose v 0 1 in
    Tensor.matmul
      (Tensor.matmul v (Tensor.diag e'))
      v_t

  (* Generate correlated increments *)
  let generate_increments matrix n_paths =
    let n = Tensor.size matrix 0 in
    
    (* Generate independent normal increments *)
    let z = Tensor.randn [n_paths; n] in
    
    (* Compute Cholesky decomposition *)
    let l = Tensor.cholesky matrix Lower in
    
    (* Generate correlated increments *)
    Tensor.matmul z l
end

module CompletePDESystem = struct
  (* Index function η(t) *)
  let compute_eta t tenors =
    let rec find_index idx =
      if idx >= Array.length tenors then
        Array.length tenors - 1
      else if tenors.(idx) >= t then
        idx
      else
        find_index (idx + 1)
    in
    find_index 0

  (* Complete drift computation *)  
  let compute_drift params rates t =
    let n = Array.length rates in
    let eta = compute_eta t params.tenors in
    let drift = Tensor.zeros [n] in
    
    for k = 0 to n-1 do
      let vk = Tensor.select params.volatilities k in
      let gamma_k = SDE.volatility_decay t 
        params.tenors.(k) (params.tenors.(k) +. params.tenors.(0)) in
        
      (* Sum over appropriate indices based on η(t) *)
      let mut_sum = ref (Tensor.zeros [1]) in
      for i = eta to k do
        let rho_ki = Tensor.select 
          (Tensor.select params.correlations k) i in
          
        let vi = Tensor.select params.volatilities i in
        let gamma_i = SDE.volatility_decay t
          params.tenors.(i) (params.tenors.(i) +. params.tenors.(0)) in
          
        let tau_i = params.tenors.(i) -. params.tenors.(i-1) in
        let rate_i = Tensor.select rates i in
        
        let term = Tensor.mul
          (Tensor.mul 
            (Tensor.mul vi
              (Tensor.of_float (gamma_i *. Tensor.float_value rho_ki)))
            (Tensor.of_float tau_i))
          (Tensor.div rate_i
            (Tensor.add (Tensor.ones [1])
              (Tensor.mul rate_i 
                (Tensor.of_float tau_i)))) in
                
        mut_sum := Tensor.add !mut_sum term
      done;
      
      Tensor.copy_
        (Tensor.select drift k)
        (Tensor.mul vk
          (Tensor.mul (Tensor.of_float gamma_k) !mut_sum))
    done;
    
    drift

  (* Complete PDE system matrix assembly *)
  let build_system_matrix params rates t =
    let n = Array.length rates in
    let eta = compute_eta t params.tenors in
    
    (* Initialize block matrices *)
    let system = Tensor.zeros [n * n; n * n] in
    
    (* Build blocks *)
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let block_i = i * n in
        let block_j = j * n in
        
        (* Drift terms *)
        if i = j then
          let drift = compute_drift params rates t in
          for k = 0 to n-1 do
            Tensor.copy_
              (Tensor.select 
                (Tensor.select system (block_i + k))
                (block_j + k))
              (Tensor.select drift k)
          done;
          
        (* Diffusion terms *)
        if i >= eta && j >= eta then begin
          let vi = Tensor.select params.volatilities i in
          let vj = Tensor.select params.volatilities j in
          let rho_ij = Tensor.select
            (Tensor.select params.correlations i) j in
          let gamma_i = SDE.volatility_decay t 
            params.tenors.(i) (params.tenors.(i) +. params.tenors.(0)) in
          let gamma_j = SDE.volatility_decay t
            params.tenors.(j) (params.tenors.(j) +. params.tenors.(0)) in
            
          (* Second derivative coefficients *)
          let coeff = Tensor.mul
            (Tensor.mul 
              (Tensor.mul vi vj)
              (Tensor.of_float (gamma_i *. gamma_j)))
            (Tensor.mul rho_ij
              (Tensor.of_float 0.5)) in
              
          for k = 0 to n-1 do
            for l = 0 to n-1 do
              if abs (k - l) <= 1 then
                Tensor.copy_
                  (Tensor.select
                    (Tensor.select system (block_i + k))
                    (block_j + l))
                  coeff
            done
          done
        end
      done
    done;
    
    system

  (* Complete relative price PDE *)
  let relative_price_pde params rates t =
    (* Build system matrix *)
    let system = build_system_matrix params rates t in
    
    (* Handle time derivative term *)
    let dt = params.dt in
    let identity = Tensor.eye (Tensor.size system 0) in
    
    let lhs = Tensor.add identity
      (Tensor.mul system (Tensor.of_float dt)) in
      
    (* Set up boundary conditions *)
    let n = Array.length rates in
    for i = 0 to n-1 do
      (* Lower boundary *)
      Tensor.copy_
        (Tensor.select (Tensor.select lhs (i * n)) (i * n))
        (Tensor.ones [1]);
        
      (* Upper boundary *)
      Tensor.copy_
        (Tensor.select 
          (Tensor.select lhs ((i + 1) * n - 1))
          ((i + 1) * n - 1))
        (Tensor.ones [1])
    done;
    
    lhs, system
end

module CompletePDE = struct
  type pde_operator = {
    time_derivative: Tensor.t -> Tensor.t;      (* ∂Π/∂t *)
    space_derivatives: Tensor.t -> Tensor.t;    (* First order terms *)
    mixed_derivatives: Tensor.t -> Tensor.t;    (* Second order terms *)
    boundary_conditions: Tensor.t -> Tensor.t;  (* Boundary handling *)
  }

  let build_operator params rates t =
    let n = Array.length rates in
    let eta = CompletePDESystem.compute_eta t params.tenors in
    
    (* Time derivative operator *)
    let time_derivative solution =
      let dt = params.dt in
      Tensor.div 
        (Tensor.sub solution
          (Tensor.narrow solution 0 0 (Tensor.size solution 0 - 1)))
        (Tensor.of_float dt)
    in
    
    (* First order spatial derivatives  *)
    let space_derivatives solution =
      let drift = CompletePDESystem.compute_drift params rates t in
      let result = Tensor.zeros_like solution in
      
      for k = eta to n-1 do
        let d_dx = PDEDiscretization.first_derivative solution k in
        Tensor.add_ result 
          (Tensor.mul (Tensor.select drift k) d_dx)
      done;
      
      result
    in
    
    (* Second order mixed derivatives *)
    let mixed_derivatives solution =
      let result = Tensor.zeros_like solution in
      
      for k = eta to n-1 do
        for l = eta to n-1 do
          let vk = Tensor.select params.volatilities k in
          let vl = Tensor.select params.volatilities l in
          let rho_kl = Tensor.select 
            (Tensor.select params.correlations k) l in
          
          let gamma_k = SDE.volatility_decay t
            params.tenors.(k) (params.tenors.(k) +. params.tenors.(0)) in
          let gamma_l = SDE.volatility_decay t  
            params.tenors.(l) (params.tenors.(l) +. params.tenors.(0)) in
            
          let coeff = Tensor.mul
            (Tensor.mul 
              (Tensor.mul vk vl)
              (Tensor.of_float (gamma_k *. gamma_l)))
            (Tensor.mul rho_kl
              (Tensor.of_float 0.5)) in
              
          let d2_dxdy = PDEDiscretization.mixed_derivative solution k l in
          Tensor.add_ result (Tensor.mul coeff d2_dxdy)
        done
      done;
      
      result
    in
    
    let boundary_conditions solution =
      (* Lower boundary R = 0 *)
      Tensor.copy_
        (Tensor.narrow solution 0 0 1)
        (Tensor.zeros [1]);
        
      (* Upper boundary - zero second derivative *)
      let n = Tensor.size solution 0 in
      let d2 = PDEDiscretization.second_derivative 
        (Tensor.narrow solution 0 (n-2) 2) 0 in
      Tensor.copy_
        (Tensor.narrow solution 0 (n-1) 1)
        (Tensor.zeros_like d2);
        
      solution
    in
    
    {
      time_derivative;
      space_derivatives;
      mixed_derivatives;
      boundary_conditions;
    }

  let solve params payoff =
    let n = Array.length params.tenors in
    
    let solution = Tensor.zeros [n] in
    let terminal_values = payoff params.initial_rates in
    let discount = Bonds.zero_coupon_bond
      params.initial_rates 
      params.tenors.(n-1)
      params.tenors.(0) in
    
    Tensor.copy_ solution
      (Tensor.div terminal_values discount);
      
    let n_steps = int_of_float (params.tenors.(n-1) /. params.dt) in
    for step = n_steps - 1 downto 0 do
      let t = float_of_int step *. params.dt in
      
      let operator = build_operator params 
        (Tensor.to_float_array solution) t in
        
      solution <- Tensor.add
        (Tensor.add
          (operator.time_derivative solution)
          (operator.space_derivatives solution))
        (operator.mixed_derivatives solution);
        
      solution <- operator.boundary_conditions solution
    done;
    
    solution
end

module PDEDiscretization = struct
  type fd_coefficients = {
    dx: float;
    dt: float;
    theta: float;  (* For theta-methods *)
  }

  let first_derivative solution dim =
    let n = Tensor.size solution dim in
    let result = Tensor.zeros_like solution in
    
    for i = 1 to n-2 do
      Tensor.copy_
        (Tensor.narrow result dim i 1)
        (Tensor.div
          (Tensor.sub
            (Tensor.narrow solution dim (i+1) 1)
            (Tensor.narrow solution dim (i-1) 1))
          (Tensor.of_float (2.0 *. params.dx)))
    done;
    
    result

  let second_derivative solution dim =
    let n = Tensor.size solution dim in
    let result = Tensor.zeros_like solution in
    
    for i = 1 to n-2 do
      Tensor.copy_
        (Tensor.narrow result dim i 1)
        (Tensor.div
          (Tensor.add
            (Tensor.sub
              (Tensor.add
                (Tensor.narrow solution dim (i+1) 1)
                (Tensor.narrow solution dim (i-1) 1))
              (Tensor.mul
                (Tensor.narrow solution dim i 1)
                (Tensor.of_float 2.0)))
            (Tensor.of_float (params.dx *. params.dx))))
    done;
    
    result

  let mixed_derivative solution dim1 dim2 =
    let n1 = Tensor.size solution dim1 in
    let n2 = Tensor.size solution dim2 in
    let result = Tensor.zeros_like solution in
    
    for i = 1 to n1-2 do
      for j = 1 to n2-2 do
        let d2_dxdy = Tensor.div
          (Tensor.sub
            (Tensor.sub
              (Tensor.add
                (Tensor.narrow 
                  (Tensor.narrow solution dim1 (i+1) 1)
                  dim2 (j+1) 1)
                (Tensor.narrow
                  (Tensor.narrow solution dim1 (i-1) 1)
                  dim2 (j-1) 1))
              (Tensor.narrow
                (Tensor.narrow solution dim1 (i+1) 1)
                dim2 (j-1) 1))
            (Tensor.narrow
              (Tensor.narrow solution dim1 (i-1) 1)
              dim2 (j+1) 1))
          (Tensor.of_float 
            (4.0 *. params.dx *. params.dx)) in
            
        Tensor.copy_
          (Tensor.narrow
            (Tensor.narrow result dim1 i 1)
            dim2 j 1)
          d2_dxdy
      done
    done;
    
    result
end

module CompleteSwaptionMethods = struct
  let compute_irs_value params rates t_a t_b strike =
    let n = Array.length rates in
    
    (* Find tenor indices *)
    let a = ref 0 in
    let b = ref 0 in
    for i = 0 to n-1 do
      if params.tenors.(i) >= t_a then a := i;
      if params.tenors.(i) <= t_b then b := i
    done;
    
    (* Compute IRS value *)
    let mut_sum = ref 0.0 in
    for i = !a + 1 to !b do
      (* Payment period *)
      let tau_i = params.tenors.(i) -. params.tenors.(i-1) in
      
      (* Discount factor *)
      let p_ti = Tensor.float_value
        (Bonds.zero_coupon_bond params.initial_rates t_a params.tenors.(i)) in
        
      (* Rate difference *)
      let r_i = Tensor.float_value (Tensor.select rates (i-1)) in
      
      mut_sum := !mut_sum +. p_ti *. tau_i *. (r_i -. strike)
    done;
    
    !mut_sum

  type tenor_grid = {
    payment_dates: float array;
    fixing_dates: float array;
    observation_dates: float array;
    year_fractions: float array;
  }

  let build_tenor_grid params start_date end_date =
    let n_periods = int_of_float ((end_date -. start_date) /. 0.25) in
    
    (* Payment dates - quarterly *)
    let payment_dates = Array.init (n_periods + 1) (fun i ->
      start_date +. float_of_int i *. 0.25) in
      
    (* Fixing dates - 2 business days before *)
    let fixing_dates = Array.map (fun d -> d -. 2.0 /. 360.0) payment_dates in
    
    (* Daily observation dates for compounding *)
    let obs_dates = ref [] in
    Array.iter2 (fun fix_date pay_date ->
      let days = int_of_float ((pay_date -. fix_date) *. 360.0) in
      for d = 0 to days-1 do
        obs_dates := (fix_date +. float_of_int d /. 360.0) :: !obs_dates
      done) fixing_dates payment_dates;
      
    (* Year fractions - actual/360 *)
    let year_fractions = Array.init (n_periods) (fun i ->
      (payment_dates.(i+1) -. payment_dates.(i)) *. 360.0 /. 360.0) in
      
    {
      payment_dates;
      fixing_dates;
      observation_dates = Array.of_list (List.rev !obs_dates);
      year_fractions;
    }

  (* Complete swaption valuation *)
  let price_swaption params rates tenor_grid strike =
    (* Terminal payoff function *)
    let payoff final_rates =
      let irs_value = compute_irs_value params final_rates
        tenor_grid.payment_dates.(0) 
        tenor_grid.payment_dates.(Array.length tenor_grid.payment_dates - 1)
        strike in
      max 0.0 irs_value in
    
    (* Solve PDE system *)
    let solution = CompletePDE.solve params payoff in
    
    (* Apply discounting *)
    let discount = Bonds.zero_coupon_bond
      params.initial_rates 0.0
      tensor_grid.payment_dates.(0) in
      
    Tensor.mul solution discount

  (* Forward-starting swaption handling *)
  let price_forward_starting_swaption params rates tenor_grid strike forward_start =
    (* Adjust tenor grid *)
    let shifted_grid = {tenor_grid with
      payment_dates = Array.map (fun d -> d +. forward_start) 
        tenor_grid.payment_dates;
      fixing_dates = Array.map (fun d -> d +. forward_start)
        tenor_grid.fixing_dates;
      observation_dates = Array.map (fun d -> d +. forward_start)
        tenor_grid.observation_dates} in
        
    price_swaption params rates shifted_grid strike
    
  (* Compute swaption sensitivities *)
  let compute_greeks params rates tenor_grid strike =
    let base_price = price_swaption params rates tenor_grid strike in
    let n = Array.length rates in
    
    (* Delta - rate sensitivity *)
    let delta = Array.init n (fun i ->
      let bump = 0.0001 in
      let bumped_rates = Array.copy rates in
      bumped_rates.(i) <- rates.(i) +. bump;
      
      let bumped_price = price_swaption params bumped_rates tenor_grid strike in
      (Tensor.float_value bumped_price -. Tensor.float_value base_price) /. bump) in
      
    (* Gamma - second-order rate sensitivity *)
    let gamma = Array.init n (fun i ->
      let bump = 0.0001 in
      let up_rates = Array.copy rates in
      let down_rates = Array.copy rates in
      up_rates.(i) <- rates.(i) +. bump;
      down_rates.(i) <- rates.(i) -. bump;
      
      let up_price = price_swaption params up_rates tenor_grid strike in
      let down_price = price_swaption params down_rates tenor_grid strike in
      
      (Tensor.float_value up_price +. Tensor.float_value down_price -. 
       2.0 *. Tensor.float_value base_price) /. (bump *. bump)) in
       
    (* Vega - volatility sensitivity *)
    let vega = Array.init n (fun i ->
      let bump = 0.0001 in
      let bumped_vols = Tensor.clone params.volatilities in
      Tensor.add_ (Tensor.select bumped_vols i) (Tensor.of_float bump);
      
      let bumped_price = price_swaption 
        {params with volatilities = bumped_vols}
        rates tenor_grid strike in
        
      (Tensor.float_value bumped_price -. Tensor.float_value base_price) /. bump) in
      
    {delta; gamma; vega}
end

module CompleteMonteCarloMethods = struct
  type mc_params = {
    n_paths: int;
    time_steps: int;
    dt: float;
    variance_reduction: bool;
    antithetic: bool;
    seed: int option;
  }

  (* Evolution with volatility decay *)
  let evolve_rates params rates dt t =
    let n = Array.length params.tenors in
    let new_rates = Tensor.zeros_like rates in
    
    (* Generate correlated increments *)
    let dw = Tensor.randn [n] in
    let corr_dw = CorrelationStructure.generate_increments 
      params.correlations dw in
      
    (* Compute coefficients *)
    let drift = CompletePDESystem.compute_drift params 
      (Tensor.to_float_array rates) t in
      
    (* Update each rate *)
    for k = 0 to n-1 do
      let gamma_k = SDE.volatility_decay t 
        params.tenors.(k) (params.tenors.(k) +. params.tenors.(0)) in
      
      (* Volatility term *)  
      let vol_k = Tensor.select params.volatilities k in
      let sigma_term = Tensor.mul vol_k
        (Tensor.mul 
          (Tensor.select corr_dw k)
          (Tensor.of_float (gamma_k *. sqrt dt))) in
          
      (* Full update *)
      let increment = Tensor.add
        (Tensor.mul (Tensor.select drift k) (Tensor.of_float dt))
        sigma_term in
        
      Tensor.copy_
        (Tensor.select new_rates k)
        (Tensor.add (Tensor.select rates k) increment)
    done;
    
    new_rates

  (* Generate complete path with measure changes *)
  let generate_path params initial_rates mc_params =
    let n_steps = mc_params.time_steps in
    let path = Tensor.zeros [n_steps + 1; Array.length initial_rates] in
    
    (* Set initial rates *)
    Tensor.copy_
      (Tensor.select path 0)
      (Tensor.of_float_array initial_rates);
      
    (* Evolve path *)
    let curr_rates = ref (Tensor.select path 0) in
    for i = 1 to n_steps do
      let t = float_of_int (i-1) *. mc_params.dt in
      curr_rates := evolve_rates params !curr_rates mc_params.dt t;
      Tensor.copy_
        (Tensor.select path i)
        !curr_rates
    done;
    
    path

  (* Generate multiple paths *)
  let simulate params mc_params =
    Option.iter manual_seed mc_params.seed;
    
    let n_paths = if mc_params.antithetic then
      mc_params.n_paths * 2 else mc_params.n_paths in
      
    let paths = Tensor.zeros [n_paths; mc_params.time_steps + 1; 
      Array.length params.tenors] in
      
    (* Generate main paths *)
    for i = 0 to mc_params.n_paths - 1 do
      let path = generate_path params 
        (Tensor.to_float_array params.initial_rates) mc_params in
      Tensor.copy_
        (Tensor.select paths i)
        path
    done;
    
    (* Generate antithetic paths if requested *)
    if mc_params.antithetic then
      for i = 0 to mc_params.n_paths - 1 do
        let anti_path = Tensor.neg (Tensor.select paths i) in
        Tensor.copy_
          (Tensor.select paths (i + mc_params.n_paths))
          anti_path
      done;
      
    paths

  (* Swaption pricing with Monte Carlo *)
  let price_swaption params tenor_grid strike mc_params =
    (* Generate paths *)
    let paths = simulate params mc_params in
    
    (* Compute payoffs *)
    let payoffs = Tensor.zeros [Tensor.size paths 0] in
    
    for i = 0 to Tensor.size paths 0 - 1 do
      let terminal_rates = Tensor.select
        (Tensor.select paths i) mc_params.time_steps in
        
      let irs_value = CompleteSwaptionMethods.compute_irs_value
        params (Tensor.to_float_array terminal_rates)
        tenor_grid.payment_dates.(0)
        tenor_grid.payment_dates.(Array.length tenor_grid.payment_dates - 1)
        strike in
        
      Tensor.copy_
        (Tensor.select payoffs i)
        (Tensor.of_float (max 0.0 irs_value))
    done;
    
    (* Apply variance reduction if requested *)
    let final_payoffs = 
      if mc_params.variance_reduction then
        (* Control variate using analytical approximation *)
        let control = SwaptionAnalytics.black_approximation
          params strike tenor_grid in
        let beta = Statistics.optimal_control_variate_beta 
          payoffs control in
        Tensor.sub payoffs
          (Tensor.mul control (Tensor.of_float beta))
      else
        payoffs in
        
    (* Compute price with confidence interval *)
    let mean = Tensor.mean final_payoffs in
    let std = Tensor.std final_payoffs ~unbiased:true in
    let conf_interval = Statistics.normal_confidence_interval 
      mean std 0.95 (float_of_int (Tensor.size final_payoffs 0)) in
      
    {
      price = Tensor.float_value mean;
      confidence_interval = conf_interval;
      std_error = Tensor.float_value std /. 
        sqrt (float_of_int (Tensor.size final_payoffs 0))
    }

  module ErrorAnalysis = struct
    let analyze_convergence params tenor_grid strike =
      let base_paths = 1000 in
      let max_paths = 100000 in
      let steps = 5 in
      
      let results = Array.init steps (fun i ->
        let n_paths = base_paths * Int.shift_left 1 i in
        let mc_params = {
          n_paths;
          time_steps = 100;
          dt = tenor_grid.payment_dates.(0) /. 100.0;
          variance_reduction = false;
          antithetic = false;
          seed = Some 42
        } in
        
        price_swaption params tenor_grid strike mc_params) in
        
      (* Compute convergence rate *)
      let rates = Array.init (steps-1) (fun i ->
        log (abs_float (results.(i+1).price -. results.(i).price)) /.
        log (float_of_int (base_paths * Int.shift_left 1 i))) in
        
      results, rates, Statistics.extrapolate_limit results
  end
end

module MultiDimPDESolver = struct
  (* Spatial discretization parameters *)
  type spatial_params = {
    points: int array;        (* Grid points per dimension *)
    min_rates: float array;   (* Minimum rate values *)
    max_rates: float array;   (* Maximum rate values *)
    transform_type: [
      | `Uniform             (* Uniform grid *)
      | `Sinh of float       (* Sinh transformation with concentration param *)
      | `Adaptive           (* Adaptive grid points *)
    ]
  }

  let build_grid params spatial_params =
    let n = Array.length params.tenors in
    let grids = Array.make_matrix n 
      (Array.fold_left max 0 spatial_params.points) 0.0 in
      
    for dim = 0 to n-1 do
      let points = spatial_params.points.(dim) in
      let min_r = spatial_params.min_rates.(dim) in
      let max_r = spatial_params.max_rates.(dim) in
      
      match spatial_params.transform_type with
      | `Uniform ->
          (* Uniform grid spacing *)
          let dx = (max_r -. min_r) /. float_of_int (points - 1) in
          for i = 0 to points-1 do
            grids.(dim).(i) <- min_r +. float_of_int i *. dx
          done
          
      | `Sinh concentration ->
          (* Sinh transformation for grid concentration *)
          for i = 0 to points-1 do
            let xi = -1.0 +. 2.0 *. float_of_int i /. 
              float_of_int (points - 1) in
            let x = min_r +. (max_r -. min_r) *.
              (1.0 +. tanh (concentration *. xi)) /. 2.0 in
            grids.(dim).(i) <- x
          done
          
      | `Adaptive ->
          (* Concentrate points based on solution gradient *)
          let base_grid = Array.init points (fun i ->
            min_r +. (max_r -. min_r) *. 
              float_of_int i /. float_of_int (points - 1)) in
              
          let monitor = AdaptiveMesh.compute_monitor_function
            params base_grid dim in
          let new_grid = AdaptiveMesh.redistribute_points
            base_grid monitor in
            
          Array.blit new_grid 0 grids.(dim) 0 points
    done;
    
    grids

  module ADI = struct
    (* Operator splitting for multi-dimensions *)
    let split_operators coeffs =
      let n = Array.length coeffs.PDEFormulation.drift in
      
      (* Build operators for each dimension *)
      Array.init n (fun dim ->
        let op_size = Array.length coeffs.PDEFormulation.diffusion.(dim) in
        let op = Tensor.zeros [op_size; op_size] in
        
        (* Add diffusion terms *)
        let diff_coeff = coeffs.PDEFormulation.diffusion.(dim) in
        for i = 1 to op_size-2 do
          (* Central difference stencil *)
          Tensor.copy_
            (Tensor.select (Tensor.select op i) (i-1))
            (Tensor.div diff_coeff 
              (Tensor.of_float params.dx.(dim) ** 2.0));
          Tensor.copy_
            (Tensor.select (Tensor.select op i) i)
            (Tensor.mul diff_coeff
              (Tensor.of_float (-2.0 /. params.dx.(dim) ** 2.0)));
          Tensor.copy_
            (Tensor.select (Tensor.select op i) (i+1))
            (Tensor.div diff_coeff 
              (Tensor.of_float params.dx.(dim) ** 2.0))
        done;
        
        (* Add drift terms *)
        let drift_coeff = coeffs.PDEFormulation.drift.(dim) in
        for i = 1 to op_size-2 do
          (* Upwind difference based on drift sign *)
          if Tensor.float_value drift_coeff > 0.0 then begin
            Tensor.add_
              (Tensor.select (Tensor.select op i) (i-1))
              (Tensor.div drift_coeff
                (Tensor.of_float params.dx.(dim)));
            Tensor.add_
              (Tensor.select (Tensor.select op i) i)
              (Tensor.div drift_coeff
                (Tensor.of_float (-.params.dx.(dim))))
          end else begin
            Tensor.add_
              (Tensor.select (Tensor.select op i) i)
              (Tensor.div drift_coeff
                (Tensor.of_float params.dx.(dim)));
            Tensor.add_
              (Tensor.select (Tensor.select op i) (i+1))
              (Tensor.div drift_coeff
                (Tensor.of_float (-.params.dx.(dim))))
          end
        done;
        
        op)

    (* Douglas splitting scheme *)  
    let douglas_step coeffs dt solution theta =
      let n = Array.length coeffs.PDEFormulation.drift in
      let operators = split_operators coeffs in
      
      (* First step - explicit predictor *)
      let predicted = ref solution in
      for dim = 0 to n-1 do
        predicted := Tensor.add !predicted
          (Tensor.mul
            (Tensor.matmul operators.(dim) solution)
            (Tensor.of_float dt))
      done;
      
      (* Correction steps *)
      let result = ref !predicted in
      for dim = 0 to n-1 do
        let correction = Tensor.sub
          (Tensor.matmul operators.(dim) !result)
          (Tensor.matmul operators.(dim) solution) in
          
        let identity = Tensor.eye (Tensor.size operators.(dim) 0) in
        let lhs = Tensor.add identity
          (Tensor.mul operators.(dim)
            (Tensor.of_float (theta *. dt))) in
            
        result := LinearAlgebra.solve lhs
          (Tensor.add !result
            (Tensor.mul correction
              (Tensor.of_float (-. theta))))
      done;
      
      !result
  end

  let solve params spatial_params payoff =
    (* Build spatial grid *)
    let grid = build_grid params spatial_params in
    
    (* Initialize solution with terminal condition *)
    let n_points = Array.map Array.length grid in
    let solution = Tensor.zeros n_points in
    
    (* Set terminal condition *)
    let terminal_rates = Array.map (fun g -> g.(0)) grid in
    Tensor.copy_ solution
      (Tensor.of_float (payoff terminal_rates));
      
    (* Time stepping *)
    let t_steps = Array.length params.tenors in
    let dt = params.dt in
    
    for step = t_steps-1 downto 0 do
      let t = float_of_int step *. dt in
      
      (* Get coefficients *)
      let coeffs = PDEFormulation.compute_coefficients
        params terminal_rates t in
        
      (* ADI step *)
      solution <- ADI.douglas_step coeffs dt solution 0.5
    done;
    
    solution
end

module AdaptiveMesh = struct
  (* Monitor function for mesh adaptation *)
  let compute_monitor_function params grid dim =
    let n = Array.length grid in
    let monitor = Array.make n 0.0 in
    
    (* Compute solution gradient *)
    for i = 1 to n-2 do
      let dx = grid.(i+1) -. grid.(i-1) in
      let gradient = abs_float (
        params.initial_rates.(dim+1) -. 
        params.initial_rates.(dim-1)) /. dx in
      monitor.(i) <- sqrt (1.0 +. gradient *. gradient)
    done;
    
    (* Boundary values *)
    monitor.(0) <- monitor.(1);
    monitor.(n-1) <- monitor.(n-2);
    
    monitor

  (* Equidistribute points based on monitor function *)
  let redistribute_points grid monitor =
    let n = Array.length grid in
    let new_grid = Array.make n 0.0 in
    new_grid.(0) <- grid.(0);
    new_grid.(n-1) <- grid.(n-1);
    
    (* Integrate monitor function *)
    let integral = Array.make n 0.0 in
    integral.(0) <- 0.0;
    for i = 1 to n-1 do
      integral.(i) <- integral.(i-1) +.
        0.5 *. (monitor.(i) +. monitor.(i-1)) *.
        (grid.(i) -. grid.(i-1))
    done;
    
    (* Distribute points *)
    let total = integral.(n-1) in
    for i = 1 to n-2 do
      let target = total *. float_of_int i /. float_of_int (n-1) in
      
      (* Find interval containing target *)
      let rec find_interval j =
        if j >= n-1 || integral.(j) >= target then j
        else find_interval (j+1)
      in
      let j = find_interval 0 in
      
      (* Linear interpolation *)
      let t = (target -. integral.(j-1)) /.
        (integral.(j) -. integral.(j-1)) in
      new_grid.(i) <- grid.(j-1) +. t *. (grid.(j) -. grid.(j-1))
    done;
    
    new_grid

  (* Error estimation for adaptation *)
  let estimate_error solution grid =
    let n = Array.length grid in
    let error = Array.make (n-1) 0.0 in
    
    for i = 0 to n-2 do
      (* Second derivative approximation *)
      if i > 0 && i < n-2 then
        let h1 = grid.(i) -. grid.(i-1) in
        let h2 = grid.(i+1) -. grid.(i) in
        let d2 = (solution.(i+1) -. 2.0 *. solution.(i) +.
          solution.(i-1)) /. (h1 *. h2) in
        error.(i) <- abs_float d2
      else
        error.(i) <- 0.0
    done;
    
    error

  (* Full mesh adaptation cycle *)
  let adapt_mesh params solution grid threshold =
    let error = estimate_error solution grid in
    let need_refinement = ref false in
    
    (* Check if refinement needed *)
    Array.iter (fun e ->
      if e > threshold then need_refinement := true) error;
      
    if !need_refinement then
      let monitor = compute_monitor_function params grid 0 in
      let new_grid = redistribute_points grid monitor in
      
      (* Interpolate solution to new grid *)
      let new_solution = Array.make (Array.length new_grid) 0.0 in
      for i = 0 to Array.length new_grid - 1 do
        (* Find interval in old grid *)
        let x = new_grid.(i) in
        let rec find_interval j =
          if j >= Array.length grid - 1 || grid.(j+1) > x then j
          else find_interval (j+1)
        in
        let j = find_interval 0 in
        
        (* Linear interpolation *)
        let t = (x -. grid.(j)) /. (grid.(j+1) -. grid.(j)) in
        new_solution.(i) <- solution.(j) *. (1.0 -. t) +.
          solution.(j+1) *. t
      done;
      
      Some (new_grid, new_solution)
    else
      None
end

module DiscountFactors = struct
  (* Compute P(Ti,Tj) *)
  let compute_discount_factor params rates t_i t_j =
    let i = CompletePDESystem.compute_eta t_i params.tenors in
    let j = CompletePDESystem.compute_eta t_j params.tenors in
    
    if t_i = t_j then
      Tensor.ones [1]
    else if t_i < t_j then
      (* Forward discount factor *)
      let mut_prod = ref (Tensor.ones [1]) in
      for k = i+1 to j do
        let r_k = Tensor.select rates (k-1) in
        let tau_k = params.tenors.(k) -. params.tenors.(k-1) in
        mut_prod := Tensor.div !mut_prod
          (Tensor.add (Tensor.ones [1])
            (Tensor.mul r_k (Tensor.of_float tau_k)))
      done;
      !mut_prod
    else
      (* Backward discount factor *)
      let mut_prod = ref (Tensor.ones [1]) in
      for k = j+1 to i do
        let r_k = Tensor.select rates (k-1) in
        let tau_k = params.tenors.(k) -. params.tenors.(k-1) in
        mut_prod := Tensor.mul !mut_prod
          (Tensor.add (Tensor.ones [1])
            (Tensor.mul r_k (Tensor.of_float tau_k)))
      done;
      !mut_prod

  (* Bank account following equation (5) special case j=0 *)
  let compute_bank_account params rates t =
    let idx = CompletePDESystem.compute_eta t params.tenors in
    let mut_prod = ref (Tensor.ones [1]) in
    
    for k = 1 to idx do
      let r_k = Tensor.select rates (k-1) in
      let tau_k = params.tenors.(k) -. params.tenors.(k-1) in
      mut_prod := Tensor.mul !mut_prod
        (Tensor.add (Tensor.ones [1])
          (Tensor.mul r_k (Tensor.of_float tau_k)))
    done;
    
    !mut_prod
end

module GeneralizedFMM = struct
  (* Complete volatility decay *)
  let compute_volatility_decay t t_k_prev t_k =
    if t <= t_k_prev then
      1.0  (* Full volatility before fixing *)
    else if t >= t_k then
      0.0  (* Zero volatility after fixing *)
    else
      (* Linear decay during fixing period *)
      (t_k -. t) /. (t_k -. t_k_prev)

  (* Complete drift computation under different measures *)
  let compute_drift params rates t measure =
    let n = Array.length rates in
    let eta = CompletePDESystem.compute_eta t params.tenors in
    let drift = Tensor.zeros [n] in
    
    for k = 0 to n-1 do
      let vk = Tensor.select params.volatilities k in
      let gamma_k = compute_volatility_decay t 
        params.tenors.(k) (params.tenors.(k) +. params.tenors.(0)) in
        
      (* Sum over appropriate indices based on η(t) *)
      let mut_sum = ref (Tensor.zeros [1]) in
      
      for i = eta to k do
        (* Get correlation *)
        let rho_ki = Tensor.select
          (Tensor.select params.correlations k) i in
          
        (* Get volatility and decay for rate i *)
        let vi = Tensor.select params.volatilities i in
        let gamma_i = compute_volatility_decay t
          params.tenors.(i) (params.tenors.(i) +. params.tenors.(0)) in
          
        (* Year fraction *)
        let tau_i = params.tenors.(i) -. params.tenors.(i-1) in
        let rate_i = Tensor.select rates i in
        
        (* Core drift term *)
        let term = Tensor.mul
          (Tensor.mul 
            (Tensor.mul vi
              (Tensor.of_float (gamma_i *. Tensor.float_value rho_ki)))
            (Tensor.of_float tau_i))
          (Tensor.div rate_i
            (Tensor.add (Tensor.ones [1])
              (Tensor.mul rate_i 
                (Tensor.of_float tau_i)))) in
                
        mut_sum := Tensor.add !mut_sum term
      done;
      
      (* Adjust drift based on measure *)
      let measure_adjustment = match measure with
        | `RiskNeutral -> Tensor.ones [1]  (* No adjustment *)
        | `TForward maturity ->
            (* Add numeraire term *)
            let bond = Bonds.zero_coupon_bond
              params.initial_rates t maturity in
            Tensor.add (Tensor.ones [1])
              (Tensor.div bond
                (Tensor.mul vk
                  (Tensor.of_float gamma_k)))
        | `SpotMeasure ->
            (* Add spot LIBOR adjustment *)
            let spot_adjust = DiscountFactors.compute_bank_account
              params rates t in
            Tensor.mul spot_adjust
              (Tensor.of_float gamma_k)
      in
      
      Tensor.copy_
        (Tensor.select drift k)
        (Tensor.mul vk
          (Tensor.mul
            (Tensor.of_float gamma_k)
            (Tensor.mul !mut_sum measure_adjustment)))
    done;
    
    drift

  (* Complete model evolution accounting for volatility decay *)
  let evolve_rates params rates dt t measure =
    let n = Array.length rates in
    let new_rates = Tensor.zeros_like rates in
    
    (* Compute drift *)
    let drift = compute_drift params rates t measure in
    
    (* Generate correlated increments *)
    let dw = Tensor.randn [n] in
    let corr_dw = CorrelationStructure.generate_increments
      params.correlations dw in
      
    (* Update each rate *)
    for k = 0 to n-1 do
      let gamma_k = compute_volatility_decay t
        params.tenors.(k) (params.tenors.(k) +. params.tenors.(0)) in
        
      (* Volatility term *)
      let vol_k = Tensor.select params.volatilities k in
      let sigma_term = Tensor.mul vol_k
        (Tensor.mul
          (Tensor.select corr_dw k)
          (Tensor.of_float (gamma_k *. sqrt dt))) in
          
      (* Full update *)
      let increment = Tensor.add
        (Tensor.mul (Tensor.select drift k) 
          (Tensor.of_float dt))
        sigma_term in
        
      Tensor.copy_
        (Tensor.select new_rates k)
        (Tensor.add (Tensor.select rates k) increment)
    done;
    
    new_rates
end
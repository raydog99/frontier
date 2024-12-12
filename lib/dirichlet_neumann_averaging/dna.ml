open Torch

let pi = Float.pi
  
let rec gamma = function
  | x when x = 0.5 -> sqrt pi
  | x when x = 1.0 -> 1.0
  | x when x = 2.0 -> 1.0
  | x when x > 1.0 -> (x -. 1.0) *. gamma (x -. 1.0)
  | x when x > 0.0 -> gamma (x +. 1.0) /. x
  | _ -> failwith "Gamma function undefined for this input"

let fourier_transform f dim y =
  let integrand x =
    let dot_prod = Array.map2 ( *. ) x y |> Array.fold_left ( +. ) 0. in
    f x *. exp(-2. *. pi *. Complex.i *. dot_prod)
  in
  (* Numerical integration with adaptive Simpson's rule *)
  let rec adaptive_simpson f a b tol =
    let h = (b -. a) /. 6. in
    let fa = f a and fb = f b and fc = f ((a +. b) /. 2.) in
    let s1 = h *. (fa +. 4. *. fc +. fb) in
    let s2 = h /. 2. *. (fa +. 4. *. f (a +. h) +. 2. *. fc +. 
                       4. *. f (b -. h) +. fb) in
    if abs_float (s1 -. s2) < tol then s2
    else
      adaptive_simpson f a ((a +. b) /. 2.) (tol /. 2.) +.
      adaptive_simpson f ((a +. b) /. 2.) b (tol /. 2.)
  in
  adaptive_simpson integrand (-10.) 10. 1e-6

let bessel_j nu x =
  let rec series_sum k acc term =
    if abs_float term < 1e-15 *. abs_float acc then acc
    else
      let new_term = term *. (-. x *. x /. 
        (4. *. float (k * (k + int_of_float nu)))) in
      series_sum (k + 1) (acc +. new_term) new_term
  in
  if x = 0. then (if nu = 0. then 1. else 0.)
  else
    let first_term = (x /. 2.) ** nu /. gamma (nu +. 1.) in
    first_term *. series_sum 1 1. 1.

let bessel_k nu x =
  if x <= 0. then invalid_arg "x must be positive"
  else if nu = 0. then
    pi /. 2. *. exp(-. x) *. 
    (1. +. 1. /. (8. *. x) +. 9. /. (128. *. x *. x))
  else if nu = 0.5 then
    sqrt(pi /. (2. *. x)) *. exp(-. x)
  else
    let pi_over_2x = pi /. (2. *. x) in
    sqrt(pi_over_2x) *. exp(-. x) *. 
    (1. +. (4. *. nu *. nu -. 1.) /. (8. *. x))

let dct1 x =
  let n = Array.length x in
  Array.init n (fun k ->
    let sum = ref 0. in
    for i = 0 to n-1 do
      sum := !sum +. x.(i) *. cos(pi *. float k *. float i /. float (n-1))
    done;
    if k = 0 || k = n-1 then !sum else 2. *. !sum
  )

let dst1 x =
  let n = Array.length x in
  Array.init (n+1) (fun k ->
    if k = 0 || k = n then 0.
    else
      let sum = ref 0. in
      for i = 1 to n-1 do
        sum := !sum +. x.(i) *. sin(pi *. float k *. float i /. float n)
      done;
      2. *. !sum
  )

let ks_test samples theoretical_cdf =
  let n = Array.length samples in
  Array.sort compare samples;
  
  let max_diff = ref 0. in
  for i = 0 to n - 1 do
    let empirical = float (i + 1) /. float n in
    let theoretical = theoretical_cdf samples.(i) in
    max_diff := max !max_diff (abs_float (empirical -. theoretical))
  done;
  
  let critical = 1.36 /. sqrt (float n) in
  !max_diff < critical

let chi_square_test samples bins theoretical_pdf =
  let n = Array.length samples in
  let min_val = Array.fold_left min infinity samples in
  let max_val = Array.fold_left max neg_infinity samples in
  let bin_width = (max_val -. min_val) /. float bins in
  
  let observed = Array.make bins 0 in
  Array.iter (fun x ->
    let bin = int_of_float ((x -. min_val) /. bin_width) in
    let bin = min (bins - 1) (max 0 bin) in
    observed.(bin) <- observed.(bin) + 1
  ) samples;
  
  let expected = Array.init bins (fun i ->
    let x = min_val +. (float i +. 0.5) *. bin_width in
    float n *. bin_width *. theoretical_pdf x
  ) in
  
  let chi_sq = ref 0. in
  Array.iteri (fun i obs ->
    let exp = expected.(i) in
    if exp > 5. then
      chi_sq := !chi_sq +. (float obs -. exp) ** 2. /. exp
  ) observed;
  
  let critical = float (bins - 1) +. 2. *. sqrt(float (bins - 1)) in
  !chi_sq < critical

let anderson_darling_test samples theoretical_cdf =
  let n = float (Array.length samples) in
  Array.sort compare samples;
  
  let sum = ref 0. in
  for i = 0 to int_of_float n - 1 do
    let z = theoretical_cdf samples.(i) in
    if z > 0. && z < 1. then
      sum := !sum +. (2. *. float (i + 1) -. 1.) *. 
             (log z +. log(1. -. theoretical_cdf 
               samples.(Array.length samples - 1 - i)))
  done;
  
  let a_sq = -. n -. !sum /. n in
  a_sq < 2.492

let covariance_test cov points n_samples =
  let samples = Array.init n_samples (fun _ ->
    Array.map cov points) in
  
  let check_stationarity () =
    let diffs = ref [] in
    for i = 0 to Array.length points - 2 do
      for j = i + 1 to Array.length points - 1 do
        let d1 = Array.map2 (-..) points.(i) points.(j) in
        let d2 = Array.map2 (-..) points.(j) points.(i) in
        diffs := (abs_float (cov d1 -. cov d2)) :: !diffs
      done
    done;
    List.fold_left max 0. !diffs < 1e-10
  in
  
  let check_positive_definite () =
    let n = Array.length points in
    let matrix = Array.make_matrix n n 0. in
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        matrix.(i).(j) <- cov (Array.map2 (-..) 
          points.(i) points.(j))
      done
    done;
    
    let min_eig = ref infinity in
    for i = 0 to n-1 do
      let sum = Array.fold_left (+.) 0. matrix.(i) in
      min_eig := min !min_eig (sum /. float n)
    done;
    !min_eig > -1e-10
  in
  
  check_stationarity () && check_positive_definite ()

module MaternCovariance = struct
  type t = {
    nu: float;
    ell: float;
    sigma: float;
  }

  let create ~nu ~ell ~sigma = { nu; ell; sigma }

  let eval cov x r =
    let d = float (Array.length x) in
    let norm_x = sqrt (Array.fold_left (fun acc xi -> 
      acc +. xi *. xi) 0. x) in
    let term1 = 2. ** (1. -. cov.nu) /. gamma cov.nu in
    let scaled_dist = sqrt(2. *. cov.nu) *. norm_x /. cov.ell in
    if scaled_dist = 0. then cov.sigma
    else 
      cov.sigma *. term1 *. (scaled_dist ** cov.nu) *. 
      bessel_k cov.nu scaled_dist

  let fourier_transform cov xi k =
    let d = float (Array.length xi) in
    let norm_xi = sqrt (Array.fold_left (fun acc x -> 
      acc +. x *. x) 0. xi) in
    let c_nu = (4. *. pi) ** (d/.4.) *. 
               gamma(cov.nu +. d/.2.) /. 
               gamma(cov.nu) in
    let term = (2. *. cov.nu) +. norm_xi ** 2. in
    c_nu *. (2. *. pi) ** (-d/.2.) *. 
    term ** (-.(cov.nu +. d/.2.))
end

module Lattice = struct
  type multiindex = int array
  type hyperoctant = {
    q: int array;
    dim: int;
  }

  let create_hyperoctant dim signs =
    if Array.length signs <> dim then
      failwith "Signs array must match dimension"
    else
      {q = signs; dim}

  let in_hyperoctant oct mu =
    Array.mapi (fun i m -> 
      (oct.q.(i) = 1 && m >= 0) || (oct.q.(i) = -1 && m < 0)
    ) mu
    |> Array.for_all (fun x -> x)

  let truncated_lattice n dim =
    let rec cartesian_power n d acc =
      if d = 0 then [acc]
      else
        List.concat (List.init (2*n+1) (fun i ->
          cartesian_power n (d-1) ((i-n) :: acc)))
    in
    cartesian_power n dim []
    |> List.map Array.of_list
    |> List.filter (fun mu -> 
      Array.fold_left (fun acc x -> max acc (abs x)) 0 mu <= n)
end

module FunctionSpace = struct
  type function_type = Continuous | L2 | H1 | Sobolev of int
  type domain = {
    dim: int;
    bounds: (float * float) array;
  }

  let create_domain dim bounds = {dim; bounds}

  let norm space f domain =
    match space with
    | L2 ->
        let integrand x = f x ** 2. in
        (* Monte Carlo integration *)
        let n_samples = 1000 in
        let samples = Array.init n_samples (fun _ ->
          Array.init domain.dim (fun i ->
            let (a, b) = domain.bounds.(i) in
            a +. Random.float (b -. a))
        ) in
        let sum = Array.fold_left (fun acc x ->
          acc +. integrand x
        ) 0. samples in
        sqrt (sum /. float n_samples)
    | _ -> failwith "Not implemented for this space"
end

module GRF = struct
  type t = {
    covar: MaternCovariance.t;
    alpha: float;
    domain_dim: int;
  }

  let create ~covar ~alpha ~domain_dim = 
    {covar; alpha; domain_dim}

  let sample grf n points =
    let dim = Array.length (Array.get points 0) in
    let lattice = Lattice.truncated_lattice n dim in
    
    Array.map (fun x ->
      List.fold_left (fun acc mu ->
        let phase = Array.fold_left2 (fun s mi xi ->
          s +. float mi *. pi *. xi /. grf.alpha
        ) 0. mu x in
        let xi = Torch.randn [1] ~device:Torch.Cpu in
        let amplitude = sqrt(MaternCovariance.fourier_transform 
          grf.covar mu grf.alpha) in
        acc +. Torch.to_float0_exn xi *. amplitude *. cos phase
      ) 0. lattice
    ) points

  let compute_dna_coeffs grf n =
    let dim = grf.domain_dim in
    let indices = Lattice.truncated_lattice n dim in
    List.map (fun mu ->
      let lambda = MaternCovariance.fourier_transform 
        grf.covar mu ((2. *. grf.alpha) ** (-1.)) in
      (mu, sqrt lambda)
    ) indices

  let enhanced_sample grf n points =
    let coeffs = compute_dna_coeffs grf n in
    let boundary_conditions = 
      List.init (1 lsl grf.domain_dim) (fun i ->
        Array.init grf.domain_dim (fun j -> (i lsr j) land 1))
    in
    
    Array.map (fun x ->
      2. ** (-. float grf.domain_dim /. 2.) *.
      List.fold_left (fun acc b ->
        let u_b = List.fold_left (fun sum (mu, lambda) ->
          let xi = Torch.randn [1] ~device:Torch.Cpu in
          let xi_val = Torch.to_float0_exn xi in
          let phase = Array.fold_left2 (fun s mi xi ->
            s +. float mi *. pi *. xi /. grf.alpha
          ) 0. mu x in
          sum +. xi_val *. lambda *. cos phase
        ) 0. coeffs in
        acc +. u_b
      ) 0. boundary_conditions
    ) points
end

module FEM = struct
  type element = {
    nodes: float array array;
    basis: float array;
    weights: float array;
  }
  
  type mesh = {
    elements: element array;
    h: float;
  }

  let create_uniform_mesh dim n =
    let h = 1. /. float n in
    let nodes = Array.make_matrix (n+1) dim 0. in
    for i = 0 to n do
      for j = 0 to dim-1 do
        nodes.(i).(j) <- float i *. h
      done
    done;
    
    let elements = Array.init n (fun i ->
      let el_nodes = Array.init 2 (fun j ->
        nodes.(i+j)
      ) in
      let basis = Array.make 2 1. in  (* Linear basis functions *)
      let weights = Array.make 2 (h/.2.) in  (* Trapezoidal rule *)
      {nodes=el_nodes; basis; weights}
    ) in
    {elements; h}

  let assemble_stiffness mesh params =
    let n = Array.length mesh.elements in
    let stiffness = Array.make_matrix (n+1) (n+1) 0. in
    Array.iteri (fun i el ->
      let h = mesh.h in
      (* Local stiffness matrix for 1D linear elements *)
      stiffness.(i).(i) <- stiffness.(i).(i) +. 
        (1./.h +. params.SPDE.kappa**2. *. h/.3.);
      stiffness.(i).(i+1) <- stiffness.(i).(i+1) -. 
        (1./.h -. params.SPDE.kappa**2. *. h/.6.);
      stiffness.(i+1).(i) <- stiffness.(i+1).(i) -. 
        (1./.h -. params.SPDE.kappa**2. *. h/.6.);
      stiffness.(i+1).(i+1) <- stiffness.(i+1).(i+1) +. 
        (1./.h +. params.SPDE.kappa**2. *. h/.3.);
    ) mesh.elements;
    stiffness
end

module SPDE = struct
  type params = {
    kappa: float;
    beta: float;
    dim: int;
  }

  let create ~nu ~dim = 
    let beta = nu/.2. +. float dim/.4. in
    let kappa = sqrt(2. *. nu) in
    {kappa; beta; dim}

  let apply_operator params u x =
    let laplacian = Array.fold_left (fun acc xi ->
      acc -. (u x ** 2.)) 0. x in
    params.kappa ** 2. *. u x +. laplacian

  let white_noise dim n =
    Array.init dim (fun _ -> Array.init n (fun _ ->
      let xi = Torch.randn [1] ~device:Torch.Cpu in
      Torch.to_float0_exn xi))
end

let spde_test spde solution mesh =
  let check_regularity () =
    let h = mesh.h in
    let differences = Array.make (Array.length solution - 1) 0. in
    for i = 0 to Array.length solution - 2 do
      differences.(i) <- abs_float (
        (solution.(i+1) -. solution.(i)) /. h)
    done;
    Array.fold_left max 0. differences < infinity
  in
  
  let check_weak_form () =
    let n = Array.length solution in
    let residual = ref 0. in
    for i = 1 to n-2 do
      let h = mesh.h in
      let laplacian = (solution.(i+1) -. 2. *. solution.(i) +. 
                      solution.(i-1)) /. (h *. h) in
      residual := max !residual (abs_float (
        spde.kappa *. solution.(i) +. laplacian))
    done;
    !residual < 1e-8
  in
  
  check_regularity () && check_weak_form ()

module BoundaryConditions = struct
  type bc_type =
    | Dirichlet of float
    | Neumann of float
    | Robin of float * float
    | Periodic
    | Mixed of bc_type array

  type boundary = {
    bc_type: bc_type;
    boundary_index: int;
    dimension: int;
  }

  let apply_bc mesh bc stiffness rhs =
    match bc.bc_type with
    | Dirichlet value ->
        let n = Array.length stiffness in
        let idx = bc.boundary_index in
        Array.fill stiffness.(idx) 0 n 0.;
        stiffness.(idx).(idx) <- 1.;
        rhs.(idx) <- value
    | Neumann flux ->
        let n = Array.length stiffness in
        let idx = bc.boundary_index in
        let h = mesh.h in
        stiffness.(idx).(idx) <- 1. /. h;
        if idx > 0 then stiffness.(idx).(idx-1) <- -1. /. h;
        rhs.(idx) <- flux
    | Robin (alpha, beta) ->
        let n = Array.length stiffness in
        let idx = bc.boundary_index in
        let h = mesh.h in
        stiffness.(idx).(idx) <- alpha +. beta /. h;
        if idx > 0 then stiffness.(idx).(idx-1) <- -.beta /. h
    | Periodic ->
        let n = Array.length stiffness in
        stiffness.(0).(n-1) <- stiffness.(1).(0);
        stiffness.(n-1).(0) <- stiffness.(0).(1)
    | Mixed bcs ->
        Array.iteri (fun i bc' ->
          apply_bc mesh 
            {bc_type=bc'; boundary_index=i; dimension=bc.dimension}
            stiffness rhs
        ) bcs

  let verify_bc solution bc tol =
    let n = Array.length solution in
    match bc.bc_type with
    | Dirichlet value ->
        abs_float (solution.(bc.boundary_index) -. value) < tol
    | Neumann flux ->
        let h = 1. /. float n in
        let idx = bc.boundary_index in
        let deriv = if idx = 0 then
          (solution.(1) -. solution.(0)) /. h
        else
          (solution.(idx) -. solution.(idx-1)) /. h
        in
        abs_float (deriv -. flux) < tol
    | Robin (alpha, beta) ->
        let h = 1. /. float n in
        let idx = bc.boundary_index in
        let deriv = (solution.(idx) -. solution.(idx-1)) /. h in
        abs_float (alpha *. solution.(idx) +. beta *. deriv) < tol
    | Periodic ->
        abs_float (solution.(0) -. solution.(n-1)) < tol
    | Mixed bcs ->
        Array.fold_left (fun acc bc' ->
          acc && verify_bc solution 
            {bc_type=bc'; boundary_index=bc.boundary_index; 
             dimension=bc.dimension} 
            tol
        ) true bcs
end

module DNAIntegration = struct
  type boundary_config = {
    dirichlet_weight: float;
    neumann_weight: float;
    boundary_conditions: BoundaryConditions.boundary array;
  }

  let sample_with_boundaries grf config points =
    let dim = Array.length (Array.get points 0) in
    
    let dirichlet_sample = GRF.enhanced_sample grf 10 points in
    let neumann_sample = GRF.enhanced_sample grf 10 points in
    
    Array.map2 (fun d n ->
      let weighted_sum = config.dirichlet_weight *. d +. 
                        config.neumann_weight *. n in
      Array.fold_left (fun acc bc ->
        let mesh = FEM.create_uniform_mesh dim 50 in
        let stiffness = Array.make_matrix 51 51 0. in
        let rhs = Array.make 51 weighted_sum in
        BoundaryConditions.apply_bc mesh bc stiffness rhs;
        weighted_sum
      ) weighted_sum config.boundary_conditions
    ) dirichlet_sample neumann_sample

  let verify_equivalence grf spde points =
    let test_cases = [
      (`Dirichlet, 1.0, 0.0);
      (`Neumann, 0.0, 1.0);
      (`Mixed, 0.5, 0.5);
    ] in
    
    List.map (fun (bc_type, d_weight, n_weight) ->
      let config = {
        dirichlet_weight = d_weight;
        neumann_weight = n_weight;
        boundary_conditions = [|{
          BoundaryConditions.bc_type = 
            (match bc_type with
             | `Dirichlet -> Dirichlet 0.0
             | `Neumann -> Neumann 0.0
             | `Mixed -> Mixed [|Dirichlet 0.0; Neumann 0.0|]);
          boundary_index = 0;
          dimension = Array.length (Array.get points 0)
        }|]
      } in
      
      let dna_sample = sample_with_boundaries grf config points in
      let spde_sample = 
        let mesh = FEM.create_uniform_mesh 
          (Array.length (Array.get points 0)) 50 in
        let stiffness = FEM.assemble_stiffness mesh spde in
        let rhs = SPDE.white_noise spde.dim 
          (Array.length stiffness) in
        let solution = Array.make (Array.length stiffness) 0. in
        (* Simple Gaussian elimination *)
        for i = 0 to Array.length stiffness - 2 do
          for j = i + 1 to Array.length stiffness - 1 do
            let factor = stiffness.(j).(i) /. stiffness.(i).(i) in
            for k = i to Array.length stiffness - 1 do
              stiffness.(j).(k) <- stiffness.(j).(k) -. 
                factor *. stiffness.(i).(k)
            done;
            rhs.(0).(j) <- rhs.(0).(j) -. factor *. rhs.(0).(i)
          done
        done;
        
        for i = Array.length stiffness - 1 downto 0 do
          let sum = ref 0. in
          for j = i + 1 to Array.length stiffness - 1 do
            sum := !sum +. stiffness.(i).(j) *. solution.(j)
          done;
          solution.(i) <- (rhs.(0).(i) -. !sum) /. stiffness.(i).(i)
        done;
        solution in
      
      let correlation = Array.map2 ( *. ) dna_sample spde_sample
        |> Array.fold_left ( +. ) 0.
        |> fun sum -> sum /. float (Array.length dna_sample) in
      
      (bc_type, correlation)
    ) test_cases
end
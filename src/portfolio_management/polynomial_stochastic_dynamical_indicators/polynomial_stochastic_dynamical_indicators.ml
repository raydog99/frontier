open Torch

(* Compute the factorial of n *)
let rec factorial n =
  if n <= 1 then 1 else n * factorial (n - 1)
  
(* Compute the binomial coefficient *)
let binomial n k =
  let rec aux n k acc =
    if k = 0 then acc
    else aux (n - 1) (k - 1) (acc * n / k)
  in
  if k > n then 0 else aux n k 1
  
(* Multinomial coefficient *)
let multinomial ns =
  let sum = Array.fold_left (+) 0 ns in
  let num = factorial sum in
  let den = Array.fold_left (fun acc n -> acc * factorial n) 1 ns in
  num / den
  
(* Natural logarithm *)
let ln x = Stdlib.log x
  
(* Get the maximum value and its index from a tensor *)
let tensor_max tensor =
  let flat = Tensor.reshape tensor [|-1|] in
  let max_val = Tensor.max flat [-1] false in
  let max_idx = Tensor.argmax flat (-1) false in
  max_val, max_idx
  
(* Eigenvalues of a symmetric matrix *)
let eigvals_symmetric tensor =
  let n = Tensor.shape tensor |> Array.get 0 in
  let e, v = Tensor.symeig tensor ~eigenvectors:false in
  e

(* Orthogonal polynomial basis functions *)
module OrthogonalPolynomials = struct
  type basis_type = Chebyshev | Legendre | Hermite
  
  (* Compute the value of a 1D orthogonal polynomial at a point *)
  let eval_1d btype degree x =
    match btype with
    | Chebyshev ->
        (* Chebyshev polynomials of the second kind *)
        let rec aux n x =
          if n = 0 then 1.0
          else if n = 1 then 2.0 *. x
          else 2.0 *. x *. aux (n-1) x -. aux (n-2) x
        in
        aux degree x
    | Legendre ->
        (* Legendre polynomials *)
        let rec aux n x =
          if n = 0 then 1.0
          else if n = 1 then x
          else ((2.0 *. float_of_int n -. 1.0) *. x *. aux (n-1) x -. 
                (float_of_int n -. 1.0) *. aux (n-2) x) /. float_of_int n
        in
        aux degree x
    | Hermite ->
        (* Probabilists' Hermite polynomials *)
        let rec aux n x =
          if n = 0 then 1.0
          else if n = 1 then x
          else x *. aux (n-1) x -. float_of_int (n-1) *. aux (n-2) x
        in
        aux degree x
        
  (* Compute the weight function value at a point *)
  let weight_fun btype x =
    match btype with
    | Chebyshev -> sqrt (1.0 -. x *. x) (* Chebyshev weight function *)
    | Legendre -> 1.0 (* Uniform weight function *)
    | Hermite -> exp (-. x *. x /. 2.0) (* Gaussian weight function *)
    
  (* Compute the inner product of two basis functions *)
  let inner_product btype i j =
    match btype with
    | Chebyshev ->
        if i <> j then 0.0
        else if i = 0 then Float.pi
        else Float.pi /. 2.0
    | Legendre ->
        if i <> j then 0.0
        else 2.0 /. (2.0 *. float_of_int i +. 1.0)
    | Hermite ->
        if i <> j then 0.0
        else sqrt (2.0 *. Float.pi) *. float_of_int (factorial i)
    
  (* Generate the coefficients for the three-term recurrence relation *)
  let recurrence_coeffs btype i =
    match btype with
    | Chebyshev ->
        (* For Chebyshev polynomials: Pn+1 = 2x*Pn - Pn-1 *)
        2.0, 0.0, 1.0
    | Legendre ->
        (* For Legendre polynomials *)
        let n = float_of_int i in
        (2.0 *. n +. 1.0) /. (n +. 1.0), 0.0, n /. (n +. 1.0)
    | Hermite ->
        (* For Hermite polynomials *)
        1.0, 0.0, float_of_int i
        
  (* Create tensor of quadrature points and weights *)
  let quadrature_rule btype n =
    match btype with
    | Chebyshev ->
        (* Gauss-Chebyshev quadrature points and weights *)
        let points = Tensor.zeros [|n|] in
        let weights = Tensor.zeros [|n|] in
        for i = 0 to n - 1 do
          let x = cos (Float.pi *. float_of_int (2*i + 1) /. float_of_int (2*n)) in
          let w = Float.pi /. float_of_int n in
          Tensor.set points [|i|] x;
          Tensor.set weights [|i|] w
        done;
        points, weights
    | Legendre ->
        (* Gauss-Legendre quadrature *)
        let points = Tensor.zeros [|n|] in
        let weights = Tensor.zeros [|n|] in
        for i = 0 to n - 1 do
          let x = cos (Float.pi *. (float_of_int i +. 0.5) /. float_of_int n) in
          (* Approximate weights for demo purposes *)
          let w = Float.pi /. float_of_int n in
          Tensor.set points [|i|] x;
          Tensor.set weights [|i|] w
        done;
        points, weights
    | Hermite ->
        (* Gauss-Hermite quadrature *)
        let points = Tensor.zeros [|n|] in
        let weights = Tensor.zeros [|n|] in
        for i = 0 to n - 1 do
          let x = float_of_int (2*i - n + 1) *. sqrt (2.0 /. float_of_int n) in
          let w = 1.0 /. float_of_int n in
          Tensor.set points [|i|] x;
          Tensor.set weights [|i|] w
        done;
        points, weights
        
  (* Create a multivariate basis function *)
  let eval_mv btype degrees xs =
    let n = Array.length degrees in
    Array.mapi (fun i d -> eval_1d btype d (Array.get xs i)) degrees
    |> Array.fold_left ( *. ) 1.0
end

(* Polynomial Chaos Expansion *)
module PolynomialChaos = struct
  (* Polynomial chaos expansion model 
     z(t,p) ≈ Σᵢ₌₀ᵐ cᵢ(t)Ψᵢ(p) 
     where cᵢ are the coefficients and Ψᵢ are orthogonal polynomials *)
  type t = {
    coefs: Tensor.t;           (* Coefficients of the polynomial expansion [dim_state, num_terms] *)
    basis_type: OrthogonalPolynomials.basis_type;
    max_degree: int;           (* Maximum polynomial degree m *)
    dim_state: int;            (* Dimension of the state vector z *)
    dim_params: int;           (* Dimension of the parameter vector p *)
    num_terms: int;            (* Total number of terms in the expansion *)
  }
  
  (* Count the number of terms in a polynomial chaos expansion *)
  let count_terms max_degree dim =
    binomial (max_degree + dim) dim
  
  (* Create a new PCE model *)
  let create basis_type max_degree dim_state dim_params =
    let num_terms = count_terms max_degree dim_params in
    let coefs = Tensor.zeros [|dim_state; num_terms|] in
    { coefs; basis_type; max_degree; dim_state; dim_params; num_terms }
    
  (* Convert multi-index to flat index *)
  let multi_to_flat_index multi_idx max_degree =
    let dim = Array.length multi_idx in
    let sum = Array.fold_left (+) 0 multi_idx in
    if sum > max_degree then -1 (* Invalid index *)
    else
      let rec aux curr_sum curr_dim acc =
        if curr_dim = dim - 1 then
          acc + multi_idx.(curr_dim)
        else
          let max_remaining = max_degree - curr_sum in
          let term = binomial (max_remaining + dim - curr_dim - 1) (dim - curr_dim - 1) in
          aux (curr_sum + multi_idx.(curr_dim)) (curr_dim + 1) (acc + multi_idx.(curr_dim) * term)
      in
      aux 0 0 0
      
  (* Convert flat index to multi-index *)
  let flat_to_multi_index flat_idx max_degree dim =
    let multi_idx = Array.make dim 0 in
    let rec aux remain_idx curr_dim remain_degree =
      if curr_dim = dim - 1 then
        multi_idx.(curr_dim) <- remain_degree
      else
        let rec find_k k =
          let term = binomial (remain_degree - k + dim - curr_dim - 1) (dim - curr_dim) in
          if remain_idx < term then k
          else find_k (k + 1)
        in
        let k = find_k 0 in
        multi_idx.(curr_dim) <- k;
        aux (remain_idx - binomial (remain_degree - k + dim - curr_dim - 1) (dim - curr_dim))
            (curr_dim + 1) (remain_degree - k)
    in
    aux flat_idx 0 max_degree;
    multi_idx
    
  (* Evaluate a PCE at a specific parameter point *)
  let evaluate pce params =
    let result = Tensor.zeros [|pce.dim_state|] in
    for i = 0 to pce.num_terms - 1 do
      let multi_idx = flat_to_multi_index i pce.max_degree pce.dim_params in
      let basis_val = OrthogonalPolynomials.eval_mv pce.basis_type multi_idx params in
      let term = Tensor.mul_scalar (Tensor.select pce.coefs 1 (Scalar.int i)) basis_val in
      Tensor.add_ result term;
    done;
    result
    
  (* Calculate PCE coefficients using non-intrusive spectral projection *)
  (* ĉₖ(t) = ⟨z(t,p),Ψₖ(p)⟩ / ⟨Ψₖ(p),Ψₖ(p)⟩ 
     where ⟨f,g⟩ is the inner product with respect to the weight function w(p) *)
  let compute_coefficients_nisp pce dynamic_fn params_samples state_samples =
    let n_samples = Tensor.shape params_samples |> Array.get 0 in
    let new_coefs = Tensor.zeros [|pce.dim_state; pce.num_terms|] in
    
    (* For each coefficient in the expansion *)
    for i = 0 to pce.num_terms - 1 do
      let multi_idx = flat_to_multi_index i pce.max_degree pce.dim_params in
      let inner_product_sum = Tensor.zeros [|pce.dim_state|] in
      
      (* Approximate the integral using Monte Carlo integration with samples *)
      for j = 0 to n_samples - 1 do
        let param = Tensor.select params_samples 0 (Scalar.int j) in
        let state = Tensor.select state_samples 0 (Scalar.int j) in
        let basis_val = OrthogonalPolynomials.eval_mv pce.basis_type multi_idx 
                        (Tensor.to_float1 param |> Array.of_list) in
        let weighted_state = Tensor.mul_scalar state basis_val in
        Tensor.add_ inner_product_sum weighted_state;
      done;
      
      (* Divide by the norm of the basis function *)
      let norm = 
        Array.mapi (fun dim degree -> 
          OrthogonalPolynomials.inner_product pce.basis_type degree degree)
          multi_idx
        |> Array.fold_left ( *. ) 1.0
      in
      
      (* Divide by number of samples for Monte Carlo integration and by norm of basis *)
      let coef = Tensor.div_scalar inner_product_sum (float_of_int n_samples *. norm) in
      for j = 0 to pce.dim_state - 1 do
        Tensor.set new_coefs [|j; i|] (Tensor.get coef [|j|]);
      done;
    done;
    
    { pce with coefs = new_coefs }
    
  (* Compute PCE coefficients using intrusive Galerkin method *)
  (* dz/dt = ∑ᵢc̄ᵢ(t)Ψᵢ(p) = g(t,p,z)
     Leading to:
     ċₖ(t) = ⟨g(t,p,z),Ψₖ(p)⟩ / ⟨Ψₖ(p),Ψₖ(p)⟩ *)
  let compute_coefficients_galerkin pce dynamic_fn initial_state initial_params time_final n_steps =
    (* Initialize coefficients for initial state *)
    let coefs = Tensor.zeros [|pce.dim_state; pce.num_terms|] in
    
    (* If initial state is deterministic, only the zeroth coefficient is non-zero *)
    for i = 0 to pce.dim_state - 1 do
      Tensor.set coefs [|i; 0|] (Tensor.get initial_state [|i|]);
    done;
    
    let dt = time_final /. float_of_int n_steps in
    
    (* Setup quadrature rule for numerical integration *)
    let quad_order = 2 * pce.max_degree + 1 in (* Rule of thumb *)
    let points, weights = OrthogonalPolynomials.quadrature_rule pce.basis_type quad_order in
    
    (* Time stepping loop - integrating the system of ODEs for the coefficients *)
    let curr_coefs = ref coefs in
    for step = 0 to n_steps - 1 do
      let time = float_of_int step *. dt in
      let next_coefs = Tensor.zeros [|pce.dim_state; pce.num_terms|] in
      
      (* For each coefficient in the expansion *)
      for k = 0 to pce.num_terms - 1 do
        (* RHS of the ODE for coefficient k *)
        let rhs = Tensor.zeros [|pce.dim_state|] in
        
        (* Approximate the Galerkin projection integral using quadrature *)
        for q = 0 to quad_order - 1 do
          let quad_point = Tensor.get points [|q|] in
          let quad_weight = Tensor.get weights [|q|] in
          
          (* Evaluate PCE at quadrature point to get z(p) *)
          let param = Tensor.ones [|pce.dim_params|] |> Tensor.mul_scalar quad_point in
          let state = evaluate { pce with coefs = !curr_coefs } (Tensor.to_float1 param |> Array.of_list) in
          
          (* Evaluate dynamic function g(t,p,z) *)
          let f_val = dynamic_fn time (Tensor.to_float1 param |> Array.of_list) 
                      (Tensor.to_float1 state |> Array.of_list) in
          let f_tensor = Tensor.of_float1 f_val in
          
          (* Multiply by basis function and weight for quadrature integration *)
          let multi_idx = flat_to_multi_index k pce.max_degree pce.dim_params in
          let basis_val = OrthogonalPolynomials.eval_mv pce.basis_type multi_idx 
                          (Tensor.to_float1 param |> Array.of_list) in
          
          let weighted_term = Tensor.mul_scalar f_tensor (quad_weight *. basis_val) in
          Tensor.add_ rhs weighted_term;
        done;
        
        (* Divide by the norm of the basis function *)
        let multi_idx = flat_to_multi_index k pce.max_degree pce.dim_params in
        let norm = 
          Array.mapi (fun dim degree -> 
            OrthogonalPolynomials.inner_product pce.basis_type degree degree)
            multi_idx
          |> Array.fold_left ( *. ) 1.0
        in
        
        let coef_deriv = Tensor.div_scalar rhs norm in
        
        (* Forward Euler time integration step *)
        for i = 0 to pce.dim_state - 1 do
          let curr_val = Tensor.get !curr_coefs [|i; k|] in
          let new_val = curr_val +. dt *. Tensor.get coef_deriv [|i|] in
          Tensor.set next_coefs [|i; k|] new_val;
        done;
      done;
      
      curr_coefs := next_coefs;
    done;
    
    { pce with coefs = !curr_coefs }
    
  (* Compute the mean of the state from a PCE *)
  let mean pce =
    (* Mean is just the first coefficient (index 0) *)
    Tensor.select pce.coefs 1 (Scalar.int 0)
    
  (* Compute the variance of the state from a PCE *)
  let variance pce =
    let var = Tensor.zeros [|pce.dim_state|] in
    
    (* Variance is the sum of squares of the remaining coefficients, each 
       weighted by the norm of the corresponding basis function *)
    for i = 1 to pce.num_terms - 1 do
      let multi_idx = flat_to_multi_index i pce.max_degree pce.dim_params in
      let norm = 
        Array.mapi (fun dim degree -> 
          OrthogonalPolynomials.inner_product pce.basis_type degree degree)
          multi_idx
        |> Array.fold_left ( *. ) 1.0
      in
      
      let coef = Tensor.select pce.coefs 1 (Scalar.int i) in
      let squared_coef = Tensor.mul coef coef in
      let weighted_squared_coef = Tensor.mul_scalar squared_coef norm in
      Tensor.add_ var weighted_squared_coef;
    done;
    
    var
    
  (* Compute the covariance matrix from a PCE *)
  let covariance pce =
    let cov = Tensor.zeros [|pce.dim_state; pce.dim_state|] in
    
    (* Covariance matrix is the sum of outer products of the coefficients, 
       each weighted by the norm of the corresponding basis function *)
    for i = 1 to pce.num_terms - 1 do
      let multi_idx = flat_to_multi_index i pce.max_degree pce.dim_params in
      let norm = 
        Array.mapi (fun dim degree -> 
          OrthogonalPolynomials.inner_product pce.basis_type degree degree)
          multi_idx
        |> Array.fold_left ( *. ) 1.0
      in
      
      let coef = Tensor.select pce.coefs 1 (Scalar.int i) in
      let outer_product = Tensor.outer coef coef in
      let weighted_outer_product = Tensor.mul_scalar outer_product norm in
      Tensor.add_ cov weighted_outer_product;
    done;
    
    cov
end

(* Finite-Time Lyapunov Exponents *)
module FTLE = struct
  (* Compute the FTLE for a deterministic system *)
  let compute dynamic_fn initial_state time_final dt =
    let n_steps = int_of_float (time_final /. dt) in
    let dim = Tensor.shape initial_state |> Array.get 0 in
    
    (* Apply a small perturbation to the initial state *)
    let epsilon = 1e-6 in
    let states = Array.make (dim + 1) initial_state in
    for i = 0 to dim - 1 do
      let perturbed = Tensor.copy initial_state in
      let pert_val = Tensor.get perturbed [|i|] +. epsilon in
      Tensor.set perturbed [|i|] pert_val;
      states.(i+1) <- perturbed;
    done;
    
    (* Integrate each state forward in time *)
    let final_states = Array.map (fun state ->
      let curr_state = ref state in
      for step = 0 to n_steps - 1 do
        let time = float_of_int step *. dt in
        let curr_state_float = Tensor.to_float1 !curr_state |> Array.of_list in
        (* Pass empty parameter array as we're computing deterministic FTLE *)
        let derivative = dynamic_fn time [||] curr_state_float in
        let derivative_tensor = Tensor.of_float1 derivative in
        let next_state = Tensor.add !curr_state (Tensor.mul_scalar derivative_tensor dt) in
        curr_state := next_state;
      done;
      !curr_state
    ) states in
    
    (* Compute the deformation gradient (Jacobian) state transition matrix Φ *)
    let deform_gradient = Tensor.zeros [|dim; dim|] in
    for i = 0 to dim - 1 do
      let dx = Tensor.sub final_states.(i+1) final_states.(0) in
      for j = 0 to dim - 1 do
        Tensor.set deform_gradient [|j; i|] (Tensor.get dx [|j|] /. epsilon);
      done;
    done;
    
    (* Compute Cauchy-Green strain tensor (Δ = Φ^T Φ) and its eigenvalues *)
    let cg_tensor = Tensor.matmul deform_gradient 
                    (Tensor.transpose deform_gradient ~dim0:0 ~dim1:1) in
    let eigenvalues = eigvals_symmetric cg_tensor in
    
    (* FTLE is the natural log of the square root of the largest eigenvalue, 
       divided by the integration time *)
    let max_eigenval = Tensor.max eigenvalues [|-1|] false in
    let ftle = Tensor.log (Tensor.sqrt max_eigenval) |> Tensor.div_scalar (time_final) in
    
    Tensor.to_float0 ftle
end

(* Stochastic Finite-Time Lyapunov Exponents *)
module SFTLE = struct
  (* Statistical moments of FTLE due to parameter uncertainty *)
  type sftle1 = {
    mean: float;        (* First moment - mean (α₁¹) *)
    variance: float;    (* Second moment - variance (α₁²) *)
    skewness: float;    (* Third moment - skewness (α₁³) *)
    kurtosis: float;    (* Fourth moment - kurtosis (α₁⁴) *)
  }
  
  (* Measure of divergence between polynomial expansion *)
  type sftle2 = {
    coefficients: float array;  (* SFTLE2 value for each coefficient (α₂ⁱ) *)
    max_value: float;           (* Maximum SFTLE2 value across all coefficients *)
  }
  
  (* Compute SFTLE using PCE of the FTLE *)
  let compute_type1 dynamic_fn initial_state params_samples time_final dt basis_type max_degree =
    let dim_state = Tensor.shape initial_state |> Array.get 0 in
    let dim_params = Tensor.shape params_samples |> Array.get 1 in
    let n_samples = Tensor.shape params_samples |> Array.get 0 in
    
    (* For each parameter sample, compute the FTLE *)
    let ftle_samples = Tensor.zeros [|n_samples|] in
    for i = 0 to n_samples - 1 do
      let params = Tensor.select params_samples 0 (Scalar.int i) |> Tensor.to_float1 |> Array.of_list in
      let ftle_val = FTLE.compute 
        (fun t _ state -> dynamic_fn t params state) 
        initial_state time_final dt in
      Tensor.set ftle_samples [|i|] ftle_val;
    done;
    
    (* Create a PCE for the FTLE *)
    let pce = PolynomialChaos.create basis_type max_degree 1 dim_params in
    
    (* Compute PCE coefficients *)
    let ftle_tensor_samples = Tensor.unsqueeze ftle_samples ~dim:1 in
    let pce_fitted = PolynomialChaos.compute_coefficients_nisp 
                     pce (fun _ _ _ -> [|0.|]) params_samples ftle_tensor_samples in
    
    (* Extract moments from the PCE *)
    let mean = Tensor.get (PolynomialChaos.mean pce_fitted) [|0|] in
    let variance = Tensor.get (PolynomialChaos.variance pce_fitted) [|0|] in
    
    (* For higher moments, compute expectation of (FTLE - mean)^n *)
    let centered_samples = Tensor.sub ftle_samples (Tensor.ones_like ftle_samples |> Tensor.mul_scalar mean) in
    let cubed_samples = Tensor.pow centered_samples (Tensor.ones_like centered_samples |> Tensor.mul_scalar 3.) in
    let fourth_power_samples = Tensor.pow centered_samples (Tensor.ones_like centered_samples |> Tensor.mul_scalar 4.) in
    
    let skewness = Tensor.mean cubed_samples [0] false |> Tensor.div_scalar (variance ** 1.5) |> Tensor.to_float0 in
    let kurtosis = Tensor.mean fourth_power_samples [0] false |> Tensor.div_scalar (variance ** 2.0) |> Tensor.to_float0 in
    
    { mean; variance; skewness; kurtosis }
    
   to dim_state - 1 do
          let perturbed_coef = Tensor.get final_pce_models.(j+1).coefs [|i; k|] in
          Tensor.set perturbed_coefs [|j|] (perturbed_coef -. base_coef);
        done;
        
        (* Set the i-th row of the deformation tensor, computing the Jacobian ∂c_i/∂z_0 *)
        for j = 0 to dim_state - 1 do
          Tensor.set coef_deformation [|i; j|] (Tensor.get perturbed_coefs [|j|] /. epsilon);
        done;
      done;
      
      (* Compute Cauchy-Green strain tensor and its eigenvalues *)
      let cg_tensor = Tensor.matmul coef_deformation 
                      (Tensor.transpose coef_deformation ~dim0:0 ~dim1:1) in
      let eigenvalues = eigvals_symmetric cg_tensor in
      
      (* SFTLE2 is the natural log of the square root of the largest eigenvalue, 
         divided by the integration time *)
      let max_eigenval = Tensor.max eigenvalues [|-1|] false |> Tensor.to_float0 in
      let sftle2_val = log (sqrt max_eigenval) /. time_final in
      
      sftle2_values.(k) <- sftle2_val;
    done;
    
    let max_sftle2 = Array.fold_left max neg_infinity sftle2_values in
    { coefficients = sftle2_values; max_value = max_sftle2 }
end

(* Pseudo-Diffusion Exponent *)
module PseudoDiffusion = struct
  (* Psuedo-diffusion exponent  *)
  type t = {
    exponent: float;             (* The pseudo-diffusion exponent α̃ *)
    component_exponents: float array;  (* Component-wise exponents α̃ⱼ *)
  }
  
  (* Compute the pseudo-diffusion exponent from a PCE at a given time *)
  let compute pce time =
    let dim_state = pce.PolynomialChaos.dim_state in
    
    (* Compute component-wise variance:
       κ₂ = ⟨z-z₀,z-z₀⟩ = ⟨(Σᵢcᵢψᵢ-c₀)²⟩ = Σᵢsᵢcᵢ² *)
    let component_variances = Array.make dim_state 0. in
    
    for i = 0 to dim_state - 1 do
      let variance_sum = ref 0. in
      
      (* Sum up the contributions from all non-mean coefficients *)
      for j = 1 to pce.PolynomialChaos.num_terms - 1 do
        let multi_idx = PolynomialChaos.flat_to_multi_index j pce.PolynomialChaos.max_degree 
                                                           pce.PolynomialChaos.dim_params in
        (* Calculate sᵢ = ⟨ψᵢ,ψᵢ⟩ *)
        let s_i = 
          Array.mapi (fun dim degree -> 
            OrthogonalPolynomials.inner_product pce.PolynomialChaos.basis_type degree degree)
            multi_idx
          |> Array.fold_left ( *. ) 1.0
        in
        
        let coef = Tensor.get pce.PolynomialChaos.coefs [|i; j|] in
        (* Calculate sᵢcᵢ² term for the sum *)
        variance_sum := !variance_sum +. s_i *. coef *. coef;
      done;
      
      component_variances.(i) <- !variance_sum;
    done;
    
    (* Compute the full covariance matrix *)
    let cov = PolynomialChaos.covariance pce in
    
    (* Get eigenvalues of the covariance matrix *)
    let eigenvalues = eigvals_symmetric cov in
    let max_eigenval = Tensor.max eigenvalues [|-1|] false |> Tensor.to_float0 in
    
    (* Compute the pseudo-diffusion exponent:
       α̃ = log(√(max_i λ_i(c(t))) + 1) / log(t) *)
    let alpha = log (sqrt max_eigenval +. 1.) /. log time in
    
    (* Compute component-wise exponents following the same approach for each state variable *)
    let component_exponents = Array.map (fun var -> 
      log (sqrt var +. 1.) /. log time
    ) component_variances in
    
    { exponent = alpha; component_exponents }
end

(* Dynamical systems with uncertain parameters *)
module UncertainSystem = struct
  (* Uncertain dynamical system:
     dz/dt = g(t,p,z) with initial conditions z(t=t₀) = z₀
     where p ∈ Ω is a vector of uncertain model parameters *)
  type t = {
    dynamic_fn: float -> float array -> float array -> float array;  (* t, p, z -> dz/dt, represents g in equation (1) *)
    dim_state: int;      (* Dimension n of the state vector z *)
    dim_params: int;     (* Dimension n_p of the parameter vector p *)
    param_ranges: float array array;  (* Parameter ranges defining Ω: [|[|min1; max1|]; [|min2; max2|]; ...|] *)
  }
  
  (* Create a new uncertain dynamical system *)
  let create dynamic_fn dim_state dim_params param_ranges =
    { dynamic_fn; dim_state; dim_params; param_ranges }
    
  (* Generate samples of the parameter space *)
  let sample_parameters system n_samples =
    let samples = Tensor.zeros [|n_samples; system.dim_params|] in
    
    for i = 0 to n_samples - 1 do
      for j = 0 to system.dim_params - 1 do
        let min_val = system.param_ranges.(j).(0) in
        let max_val = system.param_ranges.(j).(1) in
        let sample = min_val +. Random.float (max_val -. min_val) in
        Tensor.set samples [|i; j|] sample;
      done;
    done;
    
    samples
    
  (* Propagate the system with a specific parameter vector *)
  let propagate system initial_state params time_final dt =
    let n_steps = int_of_float (time_final /. dt) in
    let curr_state = ref (Tensor.of_float1 initial_state) in
    
    for step = 0 to n_steps - 1 do
      let time = float_of_int step *. dt in
      let curr_state_float = Tensor.to_float1 !curr_state |> Array.of_list in
      let derivative = system.dynamic_fn time params curr_state_float in
      let derivative_tensor = Tensor.of_float1 derivative in
      let next_state = Tensor.add !curr_state (Tensor.mul_scalar derivative_tensor dt) in
      curr_state := next_state;
    done;
    
    Tensor.to_float1 !curr_state |> Array.of_list
    
  (* Propagate ensemble of parameter samples *)
  let propagate_ensemble system initial_state param_samples time_final dt =
    let n_samples = Tensor.shape param_samples |> Array.get 0 in
    let final_states = Tensor.zeros [|n_samples; system.dim_state|] in
    
    for i = 0 to n_samples - 1 do
      let params = Tensor.select param_samples 0 (Scalar.int i) |> Tensor.to_float1 |> Array.of_list in
      let final_state = propagate system initial_state params time_final dt in
      for j = 0 to system.dim_state - 1 do
        Tensor.set final_states [|i; j|] final_state.(j);
      done;
    done;
    
    final_states
    
  (* Propagate PCE for the system *)
  let propagate_pce system initial_state basis_type max_degree time_final dt =
    let pce = PolynomialChaos.create basis_type max_degree system.dim_state system.dim_params in
    
    (* Set the initial condition (only the mean coefficient is non-zero initially) *)
    for i = 0 to system.dim_state - 1 do
      Tensor.set pce.PolynomialChaos.coefs [|i; 0|] initial_state.(i);
    done;
    
    (* Propagate using Galerkin projection *)
    PolynomialChaos.compute_coefficients_galerkin 
      pce system.dynamic_fn (Tensor.of_float1 initial_state) 
      (Array.make system.dim_params 0.) time_final (int_of_float (time_final /. dt))
end
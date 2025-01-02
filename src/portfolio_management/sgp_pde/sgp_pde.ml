open Torch

type kernel_stats = {
  lengthscale: float;
  signal_variance: float;
  spectral_norm: float option;
}

type kernel = {
  eval: Tensor.t -> Tensor.t -> Tensor.t;
  grad: Tensor.t -> Tensor.t -> Tensor.t;
  stats: kernel_stats;
  name: string;
}

type linear_operator = {
  eval: Tensor.t -> Tensor.t;
  name: string 
}

type diff_operator = {
  order: int array;
  coefficients: Tensor.t;
}

type boundary_type = 
  | Dirichlet
  | Neumann 
  | Robin of float

type boundary_segment = {
  type_: boundary_type;
  value: Tensor.t -> Tensor.t;
  normal: Tensor.t -> Tensor.t * Tensor.t;
}

type pde_coeffs = {
  diffusion: Tensor.t;
  advection: Tensor.t;
  reaction: Tensor.t;
  forcing: Tensor.t;
}

type boundary_condition = {
  dirichlet: Tensor.t -> Tensor.t;
  neumann: Tensor.t -> Tensor.t;
}

type pde_operator = {
  interior_op: Tensor.t -> Tensor.t;
  boundary_op: Tensor.t -> Tensor.t;
}

type boundary_spec = {
  condition: boundary_condition;
  segments: boundary_segment list;
  points: Tensor.t;
}

type covariance_op = {
  forward: Tensor.t -> Tensor.t -> Tensor.t;  (* K: U* -> U *)
  adjoint: Tensor.t -> Tensor.t -> Tensor.t;  (* K*: U -> U* *)
}

type system_vars = {
  u: Tensor.t;          (* Solution *)
  z1: Tensor.t;         (* Interior values *)
  z2: Tensor.t;         (* Derivative values *)
  z3: Tensor.t;         (* Interior constraint values *)
  z_boundary: Tensor.t; (* Boundary values *)
}

let rbf_kernel params x y =
  let diff = Tensor.(x - y) in
  let sq_dist = Tensor.(sum (diff * diff) ~dim:[1]) in
  Tensor.(
    F params.signal_variance * 
    exp (neg sq_dist / F (2.0 *. params.lengthscale *. params.lengthscale))
  )

let matern_kernel params x y =
  let diff = Tensor.(x - y) in
  let dist = Tensor.(sqrt (sum (diff * diff) ~dim:[1])) in
  let scaled_dist = Tensor.(dist / F params.lengthscale) in
  Tensor.(
    F params.signal_variance * 
    (F 1.0 + scaled_dist) * 
    exp (neg scaled_dist)
  )

let periodic_kernel ~lengthscale ~period x y =
  let diff = Tensor.(x - y) in
  let scaled_dist = Tensor.(
    sin (F Float.pi * diff / F period) / F lengthscale
  ) in
  let sq_dist = Tensor.(sum (scaled_dist * scaled_dist) ~dim:[1]) in
  Tensor.(exp (neg sq_dist))

let stable_cholesky matrix ~min_eig =
  let n = Tensor.size matrix 0 in
  let diag_reg = Tensor.(eye n * F min_eig) in
  
  let rec attempt_cholesky reg_scale =
    try
      let reg_matrix = Tensor.(matrix + diag_reg * F reg_scale) in
      Tensor.cholesky reg_matrix, reg_scale
    with _ ->
      attempt_cholesky (reg_scale *. 10.)
  in
  attempt_cholesky 1e-6

let woodbury_inverse ~k_psi_phi ~k_phi_phi ~gamma =
  let l, reg_scale = stable_cholesky k_phi_phi ~min_eig:1e-6 in
  
  let l_inv_k = Tensor.triangular_solve l 
    ~upper:false ~unitriangular:false 
    ~b:k_psi_phi 
  in
  
  let a = Tensor.(l_inv_k / F (sqrt gamma)) in
  let i = Tensor.(eye (size a 0)) in
  let aat = Tensor.(mm a (transpose a 0 1)) in
  let aat_reg = Tensor.(aat + i * F (reg_scale *. 1e-6)) in
  
  let id = Tensor.(eye (size k_psi_phi 1)) in
  Tensor.(
    (id / F gamma) - 
    mm (transpose a 0 1) (mm (inverse (i + aat_reg)) a) / F gamma
  )

module RKHS = struct
  type t = {
    kernel: kernel;
    inner_product: Tensor.t -> Tensor.t -> Tensor.t;
    norm: Tensor.t -> Tensor.t;
  }

  let create ~kernel =
    let inner_product u v =
      let n = Tensor.size u 0 in
      let m = Tensor.size v 0 in
      let gram = Tensor.zeros [n; m] in
      
      for i = 0 to n-1 do
        for j = 0 to m-1 do
          let ui = Tensor.select u 0 i in
          let vj = Tensor.select v 0 j in
          let kij = kernel.eval ui vj in
          Tensor.set gram [i; j] kij
        done
      done;
      
      gram
    in

    let norm u =
      let ip = inner_product u u in
      Tensor.sqrt (Tensor.sum ip)
    in

    { kernel; inner_product; norm }
end

let fd_coeffs = function
  | 1 -> [|-0.5; 0.0; 0.5|]          
  | 2 -> [|1.0; -2.0; 1.0|]          
  | n -> failwith "Unsupported order"

let gradient f x =
  let y = f x in
  Tensor.grad x y

let partial_derivative f x ~dim ~order =
  let dx = 1e-4 in  (* Step size *)
  let coeffs = fd_coeffs order in
  let n = Array.length coeffs in
  let half = n / 2 in
  
  let result = Tensor.zeros_like x in
  for i = half to (Tensor.size x 0) - half - 1 do
    let mut_val = ref (Tensor.zeros [1]) in
    for j = 0 to n-1 do
      let xj = Tensor.select x 0 (i-half+j) in
      let fj = f xj in
      mut_val := Tensor.(!mut_val + fj * F coeffs.(j))
    done;
    Tensor.set result [i] !mut_val
  done;
  Tensor.(result / F dx)

let laplacian f x =
  let dx = 1e-4 in
  let coeffs = [|1.0; -2.0; 1.0|] in
  let result = Tensor.zeros_like x in
  
  for dim = 0 to 1 do  (* 2D Laplacian *)
    for i = 1 to (Tensor.size x 0) - 2 do
      let mut_val = ref (Tensor.zeros [1]) in
      for j = 0 to 2 do
        let xj = Tensor.select x 0 (i-1+j) in
        let fj = f xj in
        mut_val := Tensor.(!mut_val + fj * F coeffs.(j))
      done;
      let lap_dim = Tensor.(!mut_val / F (dx *. dx)) in
      Tensor.set result [i] Tensor.(get result [i] + lap_dim)
    done
  done;
  result

module SparseSolver = struct
  type t = {
    rkhs: RKHS.t;
    inducing_points: Tensor.t;
    sample_points: Tensor.t;
    gamma: float;
    pde: pde_operator;
  }

  let create ~kernel ~n_inducing ~domain ~gamma ~pde =
    let xmin, xmax, ymin, ymax = domain in
    
    let n_interior = n_inducing * 2 in
    let n_boundary = n_inducing / 2 in
    
    let interior_points = Tensor.(
      rand [n_interior; 2] * 
      F (xmax -. xmin) +
      F xmin
    ) in
    
    let boundary_points = Tensor.(
      rand [n_boundary; 2] * 
      F (xmax -. xmin) +
      F xmin
    ) in
    
    let sample_points = Tensor.cat [interior_points; boundary_points] 0 in
    
    let indices = Tensor.randperm (n_interior + n_boundary) in
    let inducing_indices = Tensor.narrow indices 0 0 n_inducing in
    let inducing_points = Tensor.index_select sample_points 0 inducing_indices in
    
    let rkhs = RKHS.create ~kernel in
    
    { rkhs; inducing_points; sample_points; gamma; pde }

  let solve t ~f ~g =
    let k_phi_phi = t.rkhs.kernel.eval t.inducing_points t.inducing_points in
    let k_psi_phi = t.rkhs.kernel.eval t.sample_points t.inducing_points in
    
    let cov_inv = woodbury_inverse 
      ~k_psi_phi 
      ~k_phi_phi 
      ~gamma:t.gamma
    in

    let opt_objective z =
      let n_interior = Tensor.size t.sample_points 0 in
      
      let u_interior = Tensor.narrow z 0 0 n_interior in
      let interior_res = t.pde.interior_op u_interior in
      
      let u_boundary = Tensor.narrow z 0 n_interior 
        (Tensor.size z 0 - n_interior) in
      let boundary_res = t.pde.boundary_op u_boundary in
      
      let residual = Tensor.cat [interior_res; boundary_res] 0 in
      Tensor.(
        mean (residual * residual) +
        F t.gamma * mean (mm z (mm cov_inv (transpose z 0 1)))
      )
    in

    let rec optimize iter z =
      if iter > 100 then z
      else
        let loss = opt_objective z in
        let grad = Tensor.grad z loss in
        let z' = Tensor.(z - F 0.01 * grad) in
        optimize (iter + 1) z'
    in

    let init_z = Tensor.zeros [Tensor.size t.sample_points 0] in
    optimize 0 init_z
end

module ErrorAnalysis = struct
  type eigen_decomp = {
    eigvals: Tensor.t;
    eigvecs: Tensor.t;
    condition_number: float;
  }

  let stable_eigen_decomp matrix =
    let scale = Tensor.max matrix in
    let scaled_matrix = Tensor.(matrix / scale) in
    
    let eigvals, eigvecs = 
      Tensor.symeig scaled_matrix ~eigenvectors:true 
    in
    
    let true_eigvals = Tensor.(eigvals * scale) in
    let cond = Tensor.(
      max eigvals / (min eigvals + F 1e-10)
      |> float_value
    ) in
    
    {eigvals=true_eigvals; eigvecs; condition_number=cond}

  let nystrom_error_bound ~kernel ~sample_points ~inducing_points ~r ~delta =
    let n_omega = Tensor.size sample_points 0 in
    let n_inducing = Tensor.size inducing_points 0 in

    (* Compute kernel matrices *)
    let k_ss = kernel.eval sample_points sample_points in
    let k_si = kernel.eval sample_points inducing_points in
    let k_ii = kernel.eval inducing_points inducing_points in

    (* Get eigenvalues *)
    let {eigvals; _} = stable_eigen_decomp k_ss in
    let sorted_eigvals, _ = Tensor.sort eigvals ~descending:true in

    (* Compute bound terms *)
    let c = float_of_int n_omega in
    let bound_term1 = 
      c *. (float_of_int n_omega) ** 2.0 *. 
      (log (2.0 /. delta)) ** 2.0 /.
      (float_of_int n_inducing)
    in

    let lambda_r = Tensor.get sorted_eigvals r in
    let bound_term2 = 4.0 *. Tensor.float_value lambda_r in

    bound_term1 +. bound_term2
end

module AdaptiveSampling = struct
  type sampling_criterion = 
    | PredictiveVariance
    | IntegratedVariance
    | ActiveLearning

  let compute_scores criterion ~model ~candidates =
    match criterion with
    | PredictiveVariance ->
        let k_star = model.SparseSolver.rkhs.kernel.eval candidates model.SparseSolver.inducing_points in
        let k_star_star = model.SparseSolver.rkhs.kernel.eval candidates candidates in
        let woodbury = woodbury_inverse 
          ~k_psi_phi:k_star
          ~k_phi_phi:(model.SparseSolver.rkhs.inner_product model.SparseSolver.inducing_points model.SparseSolver.inducing_points)
          ~gamma:model.SparseSolver.gamma
        in
        Tensor.(
          diag k_star_star - 
          diag (mm k_star (mm woodbury (transpose k_star 0 1)))
        )
        
    | IntegratedVariance ->
        let n_mc = 100 in
        let xmin, xmax, _, _ = model.SparseSolver.pde.domain in
        let mc_points = Tensor.(
          rand [n_mc; 2] * 
          F (xmax -. xmin) +
          F xmin
        ) in
        let k_mc = model.SparseSolver.rkhs.kernel.eval mc_points model.SparseSolver.inducing_points in
        let k_mc_mc = model.SparseSolver.rkhs.kernel.eval mc_points mc_points in
        let woodbury = woodbury_inverse
          ~k_psi_phi:k_mc
          ~k_phi_phi:(model.SparseSolver.rkhs.inner_product model.SparseSolver.inducing_points model.SparseSolver.inducing_points)
          ~gamma:model.SparseSolver.gamma
        in
        Tensor.(
          mean (diag k_mc_mc - 
                diag (mm k_mc (mm woodbury (transpose k_mc 0 1))))
        )
        
    | ActiveLearning ->
        let k_star = model.SparseSolver.rkhs.kernel.eval candidates model.SparseSolver.inducing_points in
        let grad_k = model.SparseSolver.rkhs.kernel.grad candidates model.SparseSolver.inducing_points in
        Tensor.(
          sum (abs (mm grad_k (inverse (
            model.SparseSolver.rkhs.inner_product 
              model.SparseSolver.inducing_points 
              model.SparseSolver.inducing_points
          )))) ~dim:[1]
        )

  let select_points ~model ~candidates ~n_points ~criterion =
    let scores = compute_scores criterion ~model ~candidates in
    let _, indices = Tensor.topk scores n_points ~largest:true in
    Tensor.index_select candidates 0 indices
end

module PDEOptimizer = struct
  type t = {
    radius: float ref;
    min_radius: float;
    max_radius: float;
    eta: float;
  }

  let create ~init_radius ~min_radius ~max_radius ~eta = {
    radius = ref init_radius;
    min_radius;
    max_radius;
    eta;
  }

  let solve_trust_region_subproblem ~grad ~hess ~radius =
    let n = Tensor.size grad 0 in
    let lambda = ref 0.0 in
    
    let rec newton_iter lambda_k iter =
      if iter > 20 then lambda_k
      else
        let r = Tensor.(inverse (hess + F lambda_k * eye n)) in
        let p = Tensor.(neg (mm r grad)) in
        let p_norm = Tensor.norm p |> float_value in
        
        if abs_float (p_norm -. radius) < 1e-6 then
          lambda_k
        else
          let lambda_new = lambda_k +. (p_norm -. radius) /. radius in
          newton_iter lambda_new (iter + 1)
    in
    
    let final_lambda = newton_iter !lambda 0 in
    let r = Tensor.(inverse (hess + F final_lambda * eye n)) in
    Tensor.(neg (mm r grad))

  let optimize ~objective ~gradient ~hessian ~init_x ~optimizer:tr =
    let rec optimize x iter =
      if iter > 1000 then x
      else begin
        let fx = objective x in
        let grad = gradient x in
        let hess = hessian x in
        
        let delta = solve_trust_region_subproblem 
          ~grad ~hess ~radius:!(tr.radius)
        in
        
        let x_new = Tensor.(x + delta) in
        let fx_new = objective x_new in
        
        let actual_red = Tensor.(fx - fx_new) in
        let pred_red = Tensor.(
          neg (sum (grad * delta) + 
              sum (mm delta (mm hess delta)) / F 2.0)
        ) in
        
        let rho = Tensor.(actual_red / pred_red |> float_value) in
        
        if rho < 0.25 then
          tr.radius := !(tr.radius) /. 4.0
        else if rho > 0.75 && Tensor.norm delta |> float_value >= !(tr.radius) then
          tr.radius := min (!(tr.radius) *. 2.0) tr.max_radius;
        
        if rho > tr.eta then
          optimize x_new (iter + 1)
        else
          optimize x iter
      end
    in
    optimize init_x 0
end
open Torch

module LinearAlgebra = struct
  let pseudo_inverse tensor =
    let u, s, v = Tensor.svd tensor ~some:false in
    let epsilon = 1e-10 *. (Tensor.max s |> Tensor.to_float0_exn) in
    let s_inv = Tensor.where_scalar s 
      ~condition:(Tensor.gt_scalar s epsilon)
      ~x:(Tensor.reciprocal (Tensor.where_scalar s 
        ~condition:(Tensor.gt_scalar s epsilon) 
        ~x:s 
        ~y:(Tensor.ones_like s)))
      ~y:(Tensor.zeros_like s) in
    let s_inv_diag = Tensor.diag s_inv in
    Tensor.matmul (Tensor.matmul v s_inv_diag) (Tensor.transpose_matrix u)

  let matrix_sqrt tensor =
    let eigenvalues, eigenvectors = Tensor.symeig tensor ~eigenvectors:true ~upper:true in
    let sqrt_eigenvalues = Tensor.sqrt (Tensor.relu eigenvalues) in
    let sqrt_eigenvalues_diag = Tensor.diag sqrt_eigenvalues in
    Tensor.matmul (Tensor.matmul eigenvectors sqrt_eigenvalues_diag) 
      (Tensor.transpose_matrix eigenvectors)

  let stable_cholesky tensor =
    let n = Tensor.size tensor 0 in
    let jitter = Tensor.eye n |> Tensor.mul_scalar 1e-6 in
    let stabilized = Tensor.add tensor jitter in
    Tensor.cholesky stabilized ~upper:false

  let is_positive_definite tensor =
    try
      let _ = Tensor.cholesky tensor ~upper:false in
      true
    with _ -> false

  let nearest_positive_definite tensor =
    let symmetric = Tensor.add tensor (Tensor.transpose_matrix tensor) |> 
                   Tensor.mul_scalar 0.5 in
    let eigenvalues, eigenvectors = Tensor.symeig symmetric ~eigenvectors:true ~upper:true in
    let min_eig = Tensor.min eigenvalues |> Tensor.to_float0_exn in
    if min_eig > 0. then symmetric
    else
      let epsilon = 1e-6 in
      let adjusted_eigenvalues = Tensor.relu (Tensor.add_scalar eigenvalues epsilon) in
      let eigenvalues_diag = Tensor.diag adjusted_eigenvalues in
      Tensor.matmul (Tensor.matmul eigenvectors eigenvalues_diag) 
        (Tensor.transpose_matrix eigenvectors)

  module InnerProduct = struct
    type t = {
      compute: Tensor.t -> Tensor.t -> Tensor.t;
      metric: Tensor.t option;
    }

    let euclidean = {
      compute = (fun x y -> 
        Tensor.sum (Tensor.mul x y) |> Tensor.reshape [1]);
      metric = None;
    }

    let weighted metric = {
      compute = (fun x y ->
        let mx = Tensor.matmul metric x in
        Tensor.matmul (Tensor.transpose_matrix y) mx);
      metric = Some metric;
    }

    let mahalanobis covariance = 
      let inv_cov = pseudo_inverse covariance in
      {
        compute = (fun x y ->
          let diff = Tensor.sub x y in
          Tensor.matmul 
            (Tensor.matmul (Tensor.transpose_matrix diff) inv_cov)
            diff);
        metric = Some inv_cov;
      }
  end

  module Decomposition = struct
    type spectral_decomposition = {
      eigenvalues: Tensor.t;
      eigenvectors: Tensor.t;
      condition_number: float;
    }

    let compute_spectral tensor =
      let eigenvalues, eigenvectors = 
        Tensor.symeig tensor ~eigenvectors:true ~upper:true in
      let max_eig = Tensor.max eigenvalues |> Tensor.to_float0_exn in
      let min_eig = Tensor.min eigenvalues |> Tensor.to_float0_exn in
      let condition = if min_eig = 0. then Float.infinity 
                     else abs_float (max_eig /. min_eig) in
      { eigenvalues; eigenvectors; condition_number = condition }

    let reconstruct_from_spectral decomp =
      let eigenvalues_diag = Tensor.diag decomp.eigenvalues in
      Tensor.matmul 
        (Tensor.matmul decomp.eigenvectors eigenvalues_diag)
        (Tensor.transpose_matrix decomp.eigenvectors)
  end

  module Stable = struct
    let add_jitter tensor epsilon =
      let n = Tensor.size tensor 0 in
      let jitter = Tensor.eye n |> Tensor.mul_scalar epsilon in
      Tensor.add tensor jitter

    let solve a b =
      let epsilon = 1e-10 in
      let stabilized_a = add_jitter a epsilon in
      try
        Tensor.solve stabilized_a b
      with _ ->
        let pinv_a = pseudo_inverse stabilized_a in
        Tensor.matmul pinv_a b

    let matrix_power tensor power =
      let decomp = Decomposition.compute_spectral tensor in
      let powered_eigenvalues = Tensor.pow decomp.eigenvalues (Tensor.scalar_tensor power) in
      let powered_diag = Tensor.diag powered_eigenvalues in
      Tensor.matmul 
        (Tensor.matmul decomp.eigenvectors powered_diag)
        (Tensor.transpose_matrix decomp.eigenvectors)
  end

  module Correlation = struct
    let compute_correlation_matrix covariance =
      let std = Tensor.sqrt (Tensor.diag covariance) in
      let std_outer = Tensor.matmul 
        (Tensor.reshape std [-1; 1]) 
        (Tensor.reshape std [1; -1]) in
      Tensor.div covariance std_outer

    let is_valid_correlation matrix =
      let diag_ones = 
        Tensor.sub (Tensor.diag matrix) (Tensor.ones [Tensor.size matrix 0]) |>
        Tensor.abs |>
        Tensor.max |>
        Tensor.to_float0_exn < 1e-6 in
      let symmetric = 
        Tensor.norm (Tensor.sub matrix (Tensor.transpose_matrix matrix)) |>
        Tensor.to_float0_exn < 1e-6 in
      let bounds_valid =
        Tensor.le_scalar matrix 1.0 |> Tensor.all |> Tensor.to_bool0_exn &&
        Tensor.ge_scalar matrix (-1.0) |> Tensor.all |> Tensor.to_bool0_exn in
      
      diag_ones && symmetric && bounds_valid && is_positive_definite matrix
  end
end

open Torch

module Spaces = struct
  module Polish = struct
    type t = {
      topology: [`Complete | `Incomplete];
      separable: bool;
      metrisable: bool;
      basis: Tensor.t list option;
    }

    let check_separability space points =
      let epsilon = 1e-6 in
      let is_dense point test_set =
        List.exists (fun p ->
          let dist = Tensor.sub point p |> Tensor.norm in
          Tensor.to_float0_exn dist < epsilon) test_set in
      List.for_all (fun p -> is_dense p points) points

    let verify_polish_properties space points =
      let completeness = match space.topology with
        | `Complete -> true
        | `Incomplete -> false in
      let separability = check_separability space points in
      completeness && separability && space.metrisable

    let create_with_basis dim = 
      let basis = List.init dim (fun i ->
        let v = Tensor.zeros [dim] in
        Tensor.copy_into v (Tensor.ones [1]) [|i|];
        v) in
      {
        topology = `Complete;
        separable = true;
        metrisable = true;
        basis = Some basis;
      }
  end

  module Tangent = struct
    type t = {
      dimension: int;
      base_point: Tensor.t;
      metric: Tensor.t;
      to_tangent: Tensor.t -> Tensor.t;
      from_tangent: Tensor.t -> Tensor.t;
    }

    let create ~base_point ~metric = 
      let dim = Tensor.size base_point 0 in
      {
        dimension = dim;
        base_point;
        metric;
        to_tangent = (fun x -> Tensor.sub x base_point);
        from_tangent = (fun v -> Tensor.add base_point v);
      }

    let parallel_transport from_space to_space v =
      let transition = Tensor.sub to_space.base_point from_space.base_point in
      let transport_matrix = LinearAlgebra.matrix_sqrt 
        (Tensor.matmul 
          (Tensor.matmul 
            (LinearAlgebra.pseudo_inverse from_space.metric)
            to_space.metric)
          transition) in
      Tensor.matmul transport_matrix v

    let metric_at_point space point =
      (* Compute metric tensor at given point *)
      let basis_vectors = match Polish.create_with_basis space.dimension with
        | {basis = Some b; _} -> b
        | _ -> failwith "Could not create basis" in
      
      let metric_components = List.map
        (fun u -> List.map
          (fun v ->
            let u_tangent = space.to_tangent u in
            let v_tangent = space.to_tangent v in
            Tensor.matmul
              (Tensor.matmul (Tensor.transpose_matrix u_tangent) space.metric)
              v_tangent)
          basis_vectors)
        basis_vectors in
      
      let stack_row = List.map (fun row ->
        Tensor.stack row ~dim:0) metric_components in
      Tensor.stack stack_row ~dim:0
  end

  module RiemannianManifold = struct
    type t = {
      dimension: int;
      metric: Tensor.t -> Tensor.t;
      christoffel: Tensor.t -> Tensor.t;
      exp_map: Tensor.t -> Tensor.t -> Tensor.t;
      log_map: Tensor.t -> Tensor.t -> Tensor.t;
    }

    let create ~dimension ~metric ~christoffel =
      let exp_map base_point v =
        Tensor.add base_point v in
      
      let log_map base_point x =
        Tensor.sub x base_point in

      { dimension; metric; christoffel; exp_map; log_map }

    let geodesic manifold p q t =
      let v = manifold.log_map p q in
      manifold.exp_map p (Tensor.mul_scalar v t)

    let parallel_transport manifold p q v =
      (* First-order parallel transport along geodesic *)
      let gamma_t = geodesic manifold p q 0.5 in
      let christ = manifold.christoffel gamma_t in
      let velocity = manifold.log_map p q in
      let transport = Tensor.add v
        (Tensor.matmul
          (Tensor.matmul christ velocity)
          v |> Tensor.mul_scalar (-0.5)) in
      transport
  end

  module FiberBundle = struct
    type t = {
      base_manifold: RiemannianManifold.t;
      fiber_dimension: int;
      total_space_dim: int;
      projection: Tensor.t -> Tensor.t;
      lift: Tensor.t -> Tensor.t;
      connection: Tensor.t -> Tensor.t -> Tensor.t;  (* Horizontal lift of tangent vectors *)
    }

    let create ~base_manifold ~fiber_dim =
      let total_dim = base_manifold.dimension + fiber_dim in
      {
        base_manifold;
        fiber_dimension = fiber_dim;
        total_space_dim = total_dim;
        projection = (fun x ->
          Tensor.narrow x ~dim:0 ~start:0 ~length:base_manifold.dimension);
        lift = (fun x ->
          let fiber = Tensor.zeros [fiber_dim] in
          Tensor.cat [x; fiber] ~dim:0);
        connection = (fun p v ->
          let base_component = Tensor.narrow v ~dim:0 ~start:0 
            ~length:base_manifold.dimension in
          let fiber_component = Tensor.zeros [fiber_dim] in
          Tensor.cat [base_component; fiber_component] ~dim:0);
      }

    let horizontal_lift bundle p v =
      bundle.connection p v

    let fiber_transport bundle p q x =
      let base_path t = RiemannianManifold.geodesic 
        bundle.base_manifold 
        (bundle.projection p) 
        (bundle.projection q) 
        t in
      
      let rec transport_step t acc =
        if t >= 1.0 then acc
        else
          let base_point = base_path t in
          let lifted = bundle.lift base_point in
          let horizontal = horizontal_lift bundle lifted 
            (Tensor.sub (base_path (t +. 0.1)) base_point) in
          let next_point = Tensor.add lifted horizontal in
          transport_step (t +. 0.1) next_point in
      
      transport_step 0.0 x

    let verify_bundle_structure bundle =
      let test_point = Tensor.ones [bundle.total_space_dim] in
      let projected = bundle.projection test_point in
      let lifted = bundle.lift projected in
      let back_projected = bundle.projection lifted in
      
      (* Check dimension compatibility *)
      let dims_match = 
        Tensor.size projected 0 = bundle.base_manifold.dimension &&
        Tensor.size lifted 0 = bundle.total_space_dim in
      
      (* Check projection consistency *)
      let projection_consistent = 
        Tensor.norm (Tensor.sub projected back_projected) |>
        Tensor.to_float0_exn < 1e-6 in
      
      dims_match && projection_consistent
  end

  module MetricGeometry = struct
    type metric_space = {
      dimension: int;
      distance: Tensor.t -> Tensor.t -> float;
      ball: Tensor.t -> float -> Tensor.t list;  (* Points in closed ball *)
    }

    let create_euclidean dim =
      {
        dimension = dim;
        distance = (fun x y ->
          Tensor.sub x y |> Tensor.norm |> Tensor.to_float0_exn);
        ball = (fun center radius ->
          let n_samples = 100 in  
          List.init n_samples (fun _ ->
            let direction = Tensor.randn [dim] in
            let normalized = Tensor.div direction (Tensor.norm direction) in
            let r = radius *. sqrt (Random.float 1.0) in
            Tensor.add center (Tensor.mul_scalar normalized r)));
      }

    let verify_metric_properties space points =
      let verify_triangle_inequality p q r =
        let d_pq = space.distance p q in
        let d_qr = space.distance q r in
        let d_pr = space.distance p r in
        d_pr <= d_pq +. d_qr +. 1e-6 in
      
      List.for_all (fun p ->
        List.for_all (fun q ->
          List.for_all (fun r ->
            verify_triangle_inequality p q r)
            points)
          points)
        points
  end
end

module AdjustedSpace = struct
  type t = {
    dimension: int;
    base_space: Spaces.RiemannianManifold.t;
    adjustment: Tensor.t;
    residual_space: Tensor.t -> Tensor.t;
  }

  let create ~base_space ~adjustment = 
    let dim = base_space.dimension in
    {
      dimension = dim;
      base_space;
      adjustment;
      residual_space = (fun x ->
        let adj = Tensor.matmul adjustment x in
        Tensor.sub x adj);
    }

  let inner_product space x y =
    let rx = space.residual_space x in
    let ry = space.residual_space y in
    Tensor.matmul 
      (Tensor.matmul 
        (Tensor.transpose_matrix rx)
        (space.base_space.metric rx))
      ry

  let norm space x =
    Tensor.sqrt (inner_product space x x)

  let project space x =
    let rx = space.residual_space x in
    Tensor.add 
      (Tensor.matmul space.adjustment x)
      rx

  module Sequential = struct
    type adjustment_sequence = {
      spaces: t list;
      composition: t option;
    }

    let compose_adjustments seq =
      let rec compose_pair adj1 adj2 =
        let dim = Tensor.size adj1 0 in
        let identity = Tensor.eye dim in
        let term1 = Tensor.sub identity adj1 in
        let term2 = Tensor.sub identity adj2 in
        Tensor.sub identity (Tensor.matmul term1 term2) in
      
      match seq.spaces with
      | [] -> None
      | [space] -> Some space.adjustment
      | space :: rest ->
          List.fold_left
            (fun acc next -> 
              match acc with
              | Some adj -> Some (compose_pair adj next.adjustment)
              | None -> None)
            (Some space.adjustment)
            rest

    let create spaces =
      let composed = compose_adjustments {spaces; composition=None} in
      {spaces; composition = composed}

    let adjust seq x =
      match seq.composition with
      | Some adj -> Tensor.matmul adj x
      | None ->
          List.fold_left
            (fun acc space -> project space acc)
            x
            seq.spaces
  end
end
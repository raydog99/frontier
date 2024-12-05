open Torch

module Grid = struct
  type point = float * float
  type t = {
    data: Tensor.t;
    nx: int;
    ny: int;
  }

  let make nx ny =
    { data = Tensor.zeros [nx; ny];
      nx = nx;
      ny = ny }

  let get grid i j =
    Tensor.select grid.data ~dim:0 ~index:i
    |> fun t -> Tensor.select t ~dim:0 ~index:j

  let set grid i j value =
    Tensor.copy_ (get grid i j) value
end

module CompleteMesh = struct
  type point = float * float
  type mesh_type = 
    | Fine      (* Ωhx,hy *)
    | AnisX     (* Ωhx,Hy *)
    | AnisY     (* ΩHx,hy *)
    | Coarse    (* ΩHx,Hy *)
    | Partition      (* Ωhx,hy_i,j *)

  type t = {
    points: point array;
    nx: int;
    ny: int;
    step_x: float;
    step_y: float;
    mesh_type: mesh_type;
  }

  let make_mesh nx ny nx_c ny_c mesh_type =
    let hx = 1.0 /. float_of_int (nx + 1) in
    let hy = 1.0 /. float_of_int (ny + 1) in
    let Hx = 1.0 /. float_of_int (nx_c + 1) in
    let Hy = 1.0 /. float_of_int (ny_c + 1) in
    
    let points = match mesh_type with
    | Fine -> 
        Array.init (nx * ny) (fun idx ->
          let i = idx / ny in
          let j = idx mod ny in
          (float_of_int i *. hx, float_of_int j *. hy))
    | AnisX ->
        Array.init (nx * ny_c) (fun idx ->
          let i = idx / ny_c in
          let j = idx mod ny_c in
          (float_of_int i *. hx, float_of_int j *. Hy))
    | AnisY ->
        Array.init (nx_c * ny) (fun idx ->
          let i = idx / ny in
          let j = idx mod ny in
          (float_of_int i *. Hx, float_of_int j *. hy))
    | Coarse ->
        Array.init (nx_c * ny_c) (fun idx ->
          let i = idx / ny_c in
          let j = idx mod ny_c in
          (float_of_int i *. Hx, float_of_int j *. Hy))
    | Partition ->
        [||] (* Partitions handled separately *)
    in
    {points; nx; ny; step_x = hx; step_y = hy; mesh_type}

  let check_subset mesh1 mesh2 =
    Array.for_all (fun p ->
      Array.exists ((=) p) mesh2.points
    ) mesh1.points

  let check_disjoint mesh1 mesh2 =
    Array.for_all (fun p ->
      not (Array.exists ((=) p) mesh2.points)
    ) mesh1.points

  let get_partitions nx ny nx_c ny_c =
    let partitions = ref [] in
    for i = 0 to nx_c - 1 do
      for j = 0 to ny_c - 1 do
        let points = ref [] in
        for k = 1 to nx/nx_c - 1 do
          for l = 1 to ny/ny_c - 1 do
            let x = float_of_int (i*nx/nx_c + k) /. float_of_int nx in
            let y = float_of_int (j*ny/ny_c + l) /. float_of_int ny in
            points := (x, y) :: !points
          done
        done;
        partitions := Array.of_list !points :: !partitions
      done
    done;
    !partitions
end

module Config = struct
  type t = {
    nx_f: int;
    ny_f: int;
    nz_f: int;
    nx_c: int;
    ny_c: int;
    nz_c: int;
    alpha: float array;
    beta: float array;
  }

  let make nx_f ny_f nz_f nx_c ny_c nz_c alpha beta =
    {nx_f; ny_f; nz_f; nx_c; ny_c; nz_c; alpha; beta}

  let validate config =
    if config.nx_f <= 0 || config.ny_f <= 0 || config.nz_f <= 0 then
      Error "Invalid grid dimensions"
    else if config.nx_c >= config.nx_f || 
            config.ny_c >= config.ny_f || 
            config.nz_c >= config.nz_f then
      Error "Coarse grid must be smaller than fine grid"
    else if config.nx_f mod config.nx_c <> 0 || 
            config.ny_f mod config.ny_c <> 0 || 
            config.nz_f mod config.nz_c <> 0 then
      Error "Grid dimensions must be compatible"
    else
      Ok ()
end

module Projector = struct
  type t = {
    source_mesh: CompleteMesh.t;
    target_mesh: CompleteMesh.t;
    matrix: Tensor.t;
  }

  let make source target =
    let n_source = source.CompleteMesh.nx * source.CompleteMesh.ny in
    let n_target = target.CompleteMesh.nx * target.CompleteMesh.ny in
    let mat = Tensor.zeros [n_target; n_source] in
    
    let set_projection_entries () =
      for m = 0 to n_target - 1 do
        for n = 0 to n_source - 1 do
          let i_m = m / target.CompleteMesh.ny in
          let j_m = m mod target.CompleteMesh.ny in
          let i_n = n / source.CompleteMesh.ny in
          let j_n = n mod source.CompleteMesh.ny in
          
          if i_m = i_n && j_n = j_m * (source.CompleteMesh.ny / target.CompleteMesh.ny) then
            Tensor.set mat m n (Scalar.float 1.0)
        done
      done in
    
    set_projection_entries ();
    {source_mesh = source; target_mesh = target; matrix = mat}

  let apply proj vec =
    Tensor.mm proj.matrix vec

  let transpose proj =
    {source_mesh = proj.target_mesh;
     target_mesh = proj.source_mesh;
     matrix = Tensor.transpose proj.matrix ~dim0:0 ~dim1:1}

  let verify_properties proj =
    let id_source = Tensor.eye (proj.source_mesh.CompleteMesh.nx * 
                                     proj.source_mesh.CompleteMesh.ny) in
    let id_target = Tensor.eye (proj.target_mesh.CompleteMesh.nx * 
                                     proj.target_mesh.CompleteMesh.ny) in
    
    let trans_proj = transpose proj in
    
    let prod1 = Tensor.mm proj.matrix trans_proj.matrix in
    let diff1 = Tensor.sub prod1 id_target in
    let prod2 = Tensor.mm trans_proj.matrix proj.matrix in
    let diff2 = Tensor.sub prod2 id_source in
    
    Tensor.norm diff1 < 1e-10 && Tensor.norm diff2 < 1e-10
end

module Discretization = struct
  type t = {
    aff: Tensor.t;
    afc: Tensor.t;
    acf: Tensor.t;
    acc: Tensor.t;
  }

  let tensor_product a b =
    let na = Tensor.size a 0 in
    let nb = Tensor.size b 0 in
    let result = Tensor.zeros [na * nb; na * nb] in
    
    for i = 0 to na - 1 do
      for j = 0 to na - 1 do
        for k = 0 to nb - 1 do
          for l = 0 to nb - 1 do
            let row = i * nb + k in
            let col = j * nb + l in
            let val_ = (Tensor.get a i j |> Scalar.to_float) *.
                      (Tensor.get b k l |> Scalar.to_float) in
            Tensor.set result row col (Scalar.float val_)
          done
        done
      done
    done;
    result

  let make_derivative_matrix n h =
    let mat = Tensor.zeros [n; n] in
    for i = 1 to n-2 do
      Tensor.set mat i (i-1) (Scalar.float (-1.0));
      Tensor.set mat i i (Scalar.float 2.0);
      Tensor.set mat i (i+1) (Scalar.float (-1.0))
    done;
    Tensor.div_scalar mat (Scalar.float (h *. h))

  let build config =
    let hx = 1.0 /. float_of_int config.nx_f in
    let hy = 1.0 /. float_of_int config.ny_f in
    let Hx = 1.0 /. float_of_int config.nx_c in
    let Hy = 1.0 /. float_of_int config.ny_c in

    let dx_f = make_derivative_matrix config.nx_f hx in
    let dy_f = make_derivative_matrix config.ny_f hy in
    let dx_c = make_derivative_matrix config.nx_c Hx in
    let dy_c = make_derivative_matrix config.ny_c Hy in

    let ix_f = Tensor.eye config.nx_f in
    let iy_f = Tensor.eye config.ny_f in
    let ix_c = Tensor.eye config.nx_c in
    let iy_c = Tensor.eye config.ny_c in

    let aff = tensor_product dx_f iy_f |> 
              fun m -> Tensor.add m (tensor_product ix_f dy_f) in
    let afc = tensor_product dx_f iy_c |>
              fun m -> Tensor.add m (tensor_product ix_f dy_c) in
    let acf = tensor_product dx_c iy_f |>
              fun m -> Tensor.add m (tensor_product ix_c dy_f) in
    let acc = tensor_product dx_c iy_c |>
              fun m -> Tensor.add m (tensor_product ix_c dy_c) in

    {aff; afc; acf; acc}
end

module SkeletonBuilder = struct
  type t = {
    config: Config.t;
    projector_fc: Projector.t;
    projector_cf: Projector.t;
  }

  let make config =
    let fine_mesh = CompleteMesh.make_mesh config.nx_f config.ny_f 
                     config.nx_c config.ny_c CompleteMesh.Fine in
    let anis_x = CompleteMesh.make_mesh config.nx_f config.ny_f 
                  config.nx_c config.ny_c CompleteMesh.AnisX in
    let anis_y = CompleteMesh.make_mesh config.nx_f config.ny_f 
                  config.nx_c config.ny_c CompleteMesh.AnisY in
    
    {config;
     projector_fc = Projector.make fine_mesh anis_x;
     projector_cf = Projector.make fine_mesh anis_y}

  let build_skeleton t uf_c uc_f ucc =
    (* Project solutions to coarse mesh *)
    let u_cc1 = Projector.apply t.projector_fc uf_c in
    let u_cc2 = Projector.apply t.projector_cf uc_f in

    (* Extrapolate at cross points *)
    let c1 = 4.0 /. 3.0 in
    let c2 = 4.0 /. 3.0 in
    let c3 = -1.0 /. 3.0 in
    let u_cc = Tensor.(add (add (mul_scalar u_cc1 (Scalar.float c1)) 
                                    (mul_scalar u_cc2 (Scalar.float c2)))
                               (mul_scalar ucc (Scalar.float c3))) in

    (* Compute corrections *)
    let e1 = Tensor.sub u_cc u_cc1 in
    let e2 = Tensor.sub u_cc u_cc2 in

    (* Apply corrections *)
    let uf_c_corr = Tensor.add uf_c (Projector.apply t.projector_fc e1) in
    let uc_f_corr = Tensor.add uc_f (Projector.apply t.projector_cf e2) in

    (* Merge solutions *)
    let w1 = Tensor.ones_like uf_c_corr in
    let w2 = Tensor.ones_like uc_f_corr in
    Tensor.(div (add (mul uf_c_corr w1) (mul uc_f_corr w2)) 
                     (add w1 w2))
end

module PartitionSolver = struct
  type t = {
    config: Config.t;
  }

  let make config = {config}

  let solve_partitions t skeleton rhs =
    let nx = t.config.nx_f in 
    let ny = t.config.ny_f in
    let nx_c = t.config.nx_c in
    let ny_c = t.config.ny_c in
    
    let result = Tensor.clone skeleton in
    
    (* Process each subdomain *)
    for i = 0 to nx_c - 2 do
      for j = 0 to ny_c - 2 do
        (* Extract subdomain boundaries *)
        let x_start = i * (nx / nx_c) + 1 in
        let x_end = (i + 1) * (nx / nx_c) - 1 in
        let y_start = j * (ny / ny_c) + 1 in
        let y_end = (j + 1) * (ny / ny_c) - 1 in
        
        let n_local = (x_end - x_start + 1) * (y_end - y_start + 1) in
        let a_local = Tensor.zeros [n_local; n_local] in
        let b_local = Tensor.zeros [n_local] in
        
        (* Build local system *)
        let idx = ref 0 in
        for x = x_start to x_end do
          for y = y_start to y_end do
            let hx = 1.0 /. float_of_int nx in
            let hy = 1.0 /. float_of_int ny in
            
            (* Set matrix coefficients *)
            Tensor.set a_local !idx !idx 
              (Scalar.float (2.0 /. (hx *. hx) +. 2.0 /. (hy *. hy)));
            
            if x > x_start then
              Tensor.set a_local !idx (!idx - (y_end - y_start + 1)) 
                (Scalar.float (-1.0 /. (hx *. hx)));
            if x < x_end then
              Tensor.set a_local !idx (!idx + (y_end - y_start + 1))
                (Scalar.float (-1.0 /. (hx *. hx)));
            if y > y_start then
              Tensor.set a_local !idx (!idx - 1)
                (Scalar.float (-1.0 /. (hy *. hy)));
            if y < y_end then
              Tensor.set a_local !idx (!idx + 1)
                (Scalar.float (-1.0 /. (hy *. hy)));
            
            (* Set RHS *)
            let rhs_val = Tensor.get rhs x y |> Scalar.to_float in
            Tensor.set b_local !idx (Scalar.float rhs_val);
            
            incr idx
          done
        done;
        
        let u_local = Tensor.solve a_local b_local in
        
        let idx = ref 0 in
        for x = x_start to x_end do
          for y = y_start to y_end do
            let sol = Tensor.get u_local !idx in
            Tensor.set result (x * ny + y) sol;
            incr idx
          done
        done
      done
    done;
    result
end

module Preconditioner = struct
  type t =
    | Jacobi
    | ILU of int
    | MultiGrid of int
    | BlockJacobi of int

  let create kind matrix = 
    match kind with
    | Jacobi ->
        let n = Tensor.size matrix 0 in
        let diag = Tensor.zeros [n] in
        for i = 0 to n-1 do
          diag.(i) <- 1.0 /. (Tensor.get matrix i i |>
                             Scalar.to_float)
        done;
        diag
    | ILU k ->
        let n = Tensor.size matrix 0 in
        let l = Tensor.eye n in
        let u = Tensor.clone matrix in
        for i = 0 to n-2 do
          for j = i+1 to min (i+k) (n-1) do
            let factor = (Tensor.get u j i |> Scalar.to_float) /. 
                        (Tensor.get u i i |> Scalar.to_float) in
            Tensor.set l j i (Scalar.float factor);
            for k = i+1 to min (i+k) (n-1) do
              let val_ = (Tensor.get u j k |> Scalar.to_float) -. 
                        factor *. (Tensor.get u i k |> Scalar.to_float) in
              Tensor.set u j k (Scalar.float val_)
            done
          done
        done;
        Tensor.mm l u
    | MultiGrid levels -> matrix  
    | BlockJacobi size -> matrix 

  let apply prec vec = Tensor.mul prec vec
end

module ErrorAnalysis = struct
  type error_stats = {
    l2_error: float;
    h1_error: float;
    max_error: float;
    convergence_rate: float;
  }

  let compute_errors exact_sol computed config =
    let nx = config.nx_f in
    let ny = config.ny_f in
    let hx = 1.0 /. float_of_int nx in
    let hy = 1.0 /. float_of_int ny in
    
    let l2_sum = ref 0.0 in
    let h1_sum = ref 0.0 in
    let max_err = ref 0.0 in
    
    for i = 0 to nx-1 do
      for j =

module ASDSM = struct
  type t = {
    config: Config.t;
    skeleton_builder: SkeletonBuilder.t;
    partition_solver: PartitionSolver.t;
    discretization: Discretization.t;
  }

  let create config =
    match Config.validate config with
    | Error msg -> failwith msg
    | Ok () ->
        {
          config;
          skeleton_builder = SkeletonBuilder.make config;
          partition_solver = PartitionSolver.make config;
          discretization = Discretization.build config;
        }

  let compute_residual u rhs config =
    let disc = Discretization.build config in
    Tensor.sub rhs (Tensor.mm disc.aff u)

  let solve t rhs tol max_iter =
    (* Initial guess computation *)
    let solve_anisotropic rhs_fc rhs_cf =
      let uf_c = Tensor.solve t.discretization.afc rhs_fc in
      let uc_f = Tensor.solve t.discretization.acf rhs_cf in
      let ucc = Tensor.solve t.discretization.acc 
                  (Tensor.add rhs_fc rhs_cf) in
      (uf_c, uc_f, ucc) in
    
    let (uf_c, uc_f, ucc) = solve_anisotropic rhs rhs in
    let u0 = SkeletonBuilder.build_skeleton t.skeleton_builder uf_c uc_f ucc in
    let u0 = PartitionSolver.solve_partitions t.partition_solver u0 rhs in
    
    let rec iterate u k =
      if k >= max_iter then u
      else
        let r = compute_residual u rhs t.config in
        let res_norm = Tensor.norm r in
        
        if Tensor.float_value res_norm < tol then u
        else          let (ef_c, ec_f, ecc) = solve_anisotropic r r in
          let e_skeleton = SkeletonBuilder.build_skeleton 
                            t.skeleton_builder ef_c ec_f ecc in
          let e = PartitionSolver.solve_partitions t.partition_solver e_skeleton r in
          
          let u_new = Tensor.add u e in
          iterate u_new (k + 1)
    in
    iterate u0 0
end
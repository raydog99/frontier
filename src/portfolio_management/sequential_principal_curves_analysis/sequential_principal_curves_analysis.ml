open Torch

type point = Tensor.t
type vector = Tensor.t
type matrix = Tensor.t

type transform_error = 
  | DimensionMismatch of string
  | SingularMatrix of string
  | NumericalInstability of string
  | InsufficientData of string
  | TopologyError of string
  | ConvergenceFailure of string

type validation_result = {
  valid: bool;
  error: transform_error option;
  message: string;
}

type computation_result = {
  success: bool;
  error: transform_error option;
  data: Tensor.t option;
}

module Result = struct
  type 'a t = ('a, transform_error) result

  let bind result f =
    match result with
    | Ok x -> f x
    | Error e -> Error e

  let map result f =
    match result with
    | Ok x -> Ok (f x)
    | Error e -> Error e
    
  let return x = Ok x
  
  let error e = Error e
end

(* Numerical operations *)
module Numerical = struct
  type integration_method = 
    | Trapezoidal
    | Simpson
    | GaussLegendre of int

  let gauss_kronrod_points = [|
    -0.949107912342759; -0.741531185599394; -0.405845151377397;
    0.0; 0.405845151377397; 0.741531185599394; 0.949107912342759
  |]
  
  let gauss_kronrod_weights = [|
    0.129484966168870; 0.279705391489277; 0.381830050505119;
    0.417959183673469; 0.381830050505119; 0.279705391489277;
    0.129484966168870
  |]

  let rec adaptive_integrate f a b tol max_subdivisions =
    let rec integrate_step a b depth acc =
      if depth >= max_subdivisions then
        {success = true; error = None; 
         data = Some (Tensor.float_vec [acc])}
      else
        let mid = (a +. b) /. 2.0 in
        let h = (b -. a) /. 2.0 in
        
        (* Compute Gauss-Kronrod approximation *)
        let gk_sum = ref 0.0 in
        let g_sum = ref 0.0 in
        Array.iteri (fun i x ->
          let point = mid +. h *. x in
          let weight = gauss_kronrod_weights.(i) in
          let fx = f point in
          gk_sum := !gk_sum +. weight *. fx;
          if i mod 2 = 0 then
            g_sum := !g_sum +. weight *. fx
        ) gauss_kronrod_points;
        
        let gk_value = h *. !gk_sum in
        let g_value = h *. !g_sum in
        let error = abs_float (gk_value -. g_value) in
        
        if error < tol then
          {success = true; error = None; 
           data = Some (Tensor.float_vec [acc +. gk_value])}
        else
          let left = integrate_step a mid (depth + 1) acc in
          match left with
          | {success = true; data = Some left_val; _} ->
              integrate_step mid b (depth + 1) 
                (Tensor.to_float0_exn left_val)
          | _ -> left
    in
    
    integrate_step a b 0 0.0
end

(* KD-Tree *)
module KDTree = struct
  type split_axis = int
  
  type tree = 
    | Leaf of point list
    | Node of {
        left: tree;
        right: tree;
        split_point: point;
        split_axis: split_axis;
        bounds: (float * float) array;
      }

  let rec build ?(max_leaf_size=10) ?(depth=0) points =
    let n = List.length points in
    let dims = Tensor.size (List.hd points) 0 in
    
    if n <= max_leaf_size then
      Leaf points
    else begin
      let split_axis = depth mod dims in
      
      let sorted = List.sort (fun a b ->
        compare 
          (Tensor.get a split_axis |> Tensor.to_float0_exn)
          (Tensor.get b split_axis |> Tensor.to_float0_exn)
      ) points in
      
      let median_idx = n / 2 in
      let median_point = List.nth sorted median_idx in
      
      let left_points, right_points = 
        List.partition (fun p ->
          (Tensor.get p split_axis |> Tensor.to_float0_exn) <
          (Tensor.get median_point split_axis |> Tensor.to_float0_exn)
        ) sorted in
      
      let bounds = Array.init dims (fun dim ->
        let dim_vals = List.map (fun p -> 
          Tensor.get p dim |> Tensor.to_float0_exn) points in
        (List.fold_left min Float.max_float dim_vals,
         List.fold_left max Float.min_float dim_vals)
      ) in
      
      Node {
        left = build ~max_leaf_size ~depth:(depth + 1) left_points;
        right = build ~max_leaf_size ~depth:(depth + 1) right_points;
        split_point = median_point;
        split_axis;
        bounds;
      }
    end

  let rec find_k_nearest tree k query =
    let pq = PriorityQueue.create k in
    find_k_nearest_helper tree query k Float.max_float pq;
    PriorityQueue.to_list pq

  and find_k_nearest_helper tree query k max_dist pq =
    match tree with
    | Leaf points ->
        List.iter (fun p ->
          let dist = Tensor.norm (Tensor.sub query p) |> 
            Tensor.to_float0_exn in
          PriorityQueue.add pq dist p
        ) points
    | Node {left; right; split_point; split_axis; bounds} ->
        let should_explore = Array.exists (fun (min_val, max_val) ->
          let q = Tensor.get query split_axis |> Tensor.to_float0_exn in
          q -. max_dist <= max_val && q +. max_dist >= min_val
        ) bounds in
        
        if should_explore then begin
          let dist_to_split = 
            Tensor.get query split_axis |> Tensor.to_float0_exn in
          let split_val = 
            Tensor.get split_point split_axis |> Tensor.to_float0_exn in
          
          if dist_to_split < split_val then begin
            find_k_nearest_helper left query k max_dist pq;
            find_k_nearest_helper right query k max_dist pq
          end else begin
            find_k_nearest_helper right query k max_dist pq;
            find_k_nearest_helper left query k max_dist pq
          end
        end
end

(* PDF Estimation *)
module PDFEstimator = struct
  type params = {
    bandwidth: float;
    kernel: [`Gaussian | `Epanechnikov];
    min_samples: int;
    adaptive: bool;
  }

  type t = {
    params: params;
    data: Tensor.t;
    tree: KDTree.tree;
    dimension: int;
  }

  let default_params = {
    bandwidth = 0.1;
    kernel = `Gaussian;
    min_samples = 10;
    adaptive = true;
  }

  let create ?(params=default_params) data =
    let n = Tensor.size data 0 in
    let d = Tensor.size data 1 in
    let points = 
      List.init n (fun i -> Tensor.select data 0 i) in
    let tree = KDTree.build points in
    Ok {
      params;
      data;
      tree;
      dimension = d;
    }

  let kernel_function kernel x =
    match kernel with
    | `Gaussian ->
        exp (-0.5 *. x *. x) /. sqrt (2.0 *. Float.pi)
    | `Epanechnikov ->
        if abs_float x <= 1.0 then
          0.75 *. (1.0 -. x *. x)
        else 0.0

  let estimate_density t point =
    let n = Tensor.size t.data 0 in
    let h = 
      if t.params.adaptive then
        (* Compute adaptive bandwidth based on local density *)
        let neighbors = KDTree.find_k_nearest t.tree t.params.min_samples point in
        let dists = List.map (fun p -> 
          Tensor.norm (Tensor.sub point p) |> Tensor.to_float0_exn) neighbors in
        let k_dist = List.fold_left max 0.0 dists in
        t.params.bandwidth *. k_dist
      else
        t.params.bandwidth in
    
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      let xi = Tensor.select t.data i 0 in
      let dist = Tensor.norm (Tensor.sub point xi) |> Tensor.to_float0_exn in
      sum := !sum +. kernel_function t.params.kernel (dist /. h)
    done;
    
    !sum /. (float_of_int n *. h ** float_of_int t.dimension)

  let estimate_gradient t point =
    let n = Tensor.size t.data 0 in
    let d = t.dimension in
    let grad = Tensor.zeros [d] in
    let h = t.params.bandwidth in
    
    for i = 0 to n - 1 do
      let xi = Tensor.select t.data i 0 in
      let diff = Tensor.sub point xi in
      let dist = Tensor.norm diff |> Tensor.to_float0_exn in
      let k_grad = match t.params.kernel with
        | `Gaussian ->
            let k = exp (-0.5 *. dist *. dist /. (h *. h)) in
            k *. (-dist /. (h *. h))
        | `Epanechnikov ->
            if dist <= h then
              -2.0 *. dist /. (h *. h)
            else 0.0
      in
      Tensor.add_ grad (Tensor.mul_scalar diff (k_grad /. dist))
    done;
    
    Tensor.div_scalar grad (float_of_int n *. h ** float_of_int (d + 1))
end

(* Principal Curve *)
module PrincipalCurve = struct
  type params = {
    max_iter: int;
    tol: float;
    min_points: int;
    smoothing: float;
  }

  type t = {
    points: point list;
    params: params;
    pdf_estimator: PDFEstimator.t;
  }

  let default_params = {
    max_iter = 100;
    tol = 1e-6;
    min_points = 20;
    smoothing = 0.1;
  }

  let create ?(params=default_params) pdf_estimator =
    Ok {
      points = [];
      params;
      pdf_estimator;
    }

  let smooth_points points smoothing =
    let n = List.length points in
    if n < 3 then points
    else
      let smoothed = ref [] in
      List.iteri (fun i p ->
        if i = 0 || i = n - 1 then
          smoothed := p :: !smoothed
        else
          let prev = List.nth points (i - 1) in
          let next = List.nth points (i + 1) in
          let s = Tensor.add
            (Tensor.mul_scalar prev (1.0 -. smoothing))
            (Tensor.mul_scalar next smoothing) in
          smoothed := s :: !smoothed
      ) points;
      List.rev !smoothed

  let project t point =
    let rec find_projection candidates best_dist best_proj =
      match candidates with
      | [] -> best_proj
      | p :: rest ->
          let dist = Tensor.norm (Tensor.sub point p) |> 
            Tensor.to_float0_exn in
          if dist < best_dist then
            find_projection rest dist p
          else
            find_projection rest best_dist best_proj
    in
    find_projection t.points Float.max_float (List.hd t.points)

  let tangent t point =
    let proj = project t point in
    let idx = ref 0 in
    let min_dist = ref Float.max_float in
    
    (* Find closest point index *)
    List.iteri (fun i p ->
      let dist = Tensor.norm (Tensor.sub p proj) |> 
        Tensor.to_float0_exn in
      if dist < !min_dist then begin
        min_dist := dist;
        idx := i
      end
    ) t.points;
    
    (* Compute tangent using neighboring points *)
    if !idx = 0 then
      let p1 = List.nth t.points 0 in
      let p2 = List.nth t.points 1 in
      Tensor.normalize (Tensor.sub p2 p1)
    else if !idx = List.length t.points - 1 then
      let p1 = List.nth t.points (!idx - 1) in
      let p2 = List.nth t.points !idx in
      Tensor.normalize (Tensor.sub p2 p1)
    else
      let p1 = List.nth t.points (!idx - 1) in
      let p2 = List.nth t.points (!idx + 1) in
      Tensor.normalize (Tensor.sub p2 p1)

  let fit t data =
    let n = Tensor.size data 0 in
    let rec iterate points iter =
      if iter >= t.params.max_iter then
        Ok {t with points}
      else begin
        (* Project points to curve *)
        let projections = List.map (fun p ->
          project {t with points} p
        ) (List.init n (fun i -> Tensor.select data i 0)) in
        
        (* Smooth projections *)
        let smoothed = smooth_points projections t.params.smoothing in
        
        (* Check convergence *)
        let max_diff = List.fold_left2 (fun acc p1 p2 ->
          max acc (Tensor.norm (Tensor.sub p1 p2) |> 
            Tensor.to_float0_exn)
        ) 0.0 points smoothed in
        
        if max_diff < t.params.tol then
          Ok {t with points = smoothed}
        else
          iterate smoothed (iter + 1)
      end
    in
    
    (* Initialize with first principal component *)
    let mean = Tensor.mean data 0 in
    let centered = Tensor.sub data (Tensor.expand mean [n; -1]) in
    let cov = Tensor.matmul 
      (Tensor.transpose centered 0 1) centered in
    match Tensor.svd cov with
    | Error e -> Error e
    | Ok (u, _, _) ->
        let direction = Tensor.select u 0 0 in
        let init_points = 
          List.init t.params.min_points (fun i ->
            let t = -1.0 +. 2.0 *. float_of_int i /. 
              float_of_int (t.params.min_points - 1) in
            Tensor.add mean (Tensor.mul_scalar direction t)
          ) in
        iterate init_points 0
end

(* Metric Tensor *)
module MetricTensor = struct
  type metric_type = 
    | Infomax
    | ErrorMinimization
    | Decorrelation

  type params = {
    metric_type: metric_type;
    gamma: float;
    adaptation_rate: float;
    min_eigenval: float;
  }

  let default_params = {
    metric_type = Infomax;
    gamma = 1.0;
    adaptation_rate = 0.1;
    min_eigenval = 1e-10;
  }

  (* Metric tensor with optimization criterion *)
  let compute_metric params pdf_val point local_cov =
    let n = Tensor.size point 0 in
    let base_metric = match params.metric_type with
      | Infomax ->
          (* |∇R(x)| ∝ p(x) *)
          Tensor.mul_scalar (Tensor.eye n) 
            (pdf_val ** params.gamma)
      | ErrorMinimization ->
          (* |∇R(x)| ∝ p(x)^(1/3) *)
          Tensor.mul_scalar (Tensor.eye n)
            (pdf_val ** (params.gamma /. 3.0))
      | Decorrelation ->
          (* Use local covariance structure *)
          let u, s, vt = 
            match Tensor.svd local_cov with
            | Ok (u, s, vt) -> (u, s, vt)
            | Error _ -> 
                (Tensor.eye n, Tensor.ones [n], Tensor.eye n) in
          let s_gamma = Tensor.pow s (Tensor.float_vec [params.gamma]) in
          Tensor.matmul 
            (Tensor.matmul u (Tensor.diag s_gamma))
            vt in
    
    (* Ensure positive definiteness *)
    let eigenvals = Tensor.eigenvalues base_metric in
    let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
    if min_eval < params.min_eigenval then
      Tensor.add base_metric 
        (Tensor.mul_scalar (Tensor.eye n)
           (params.min_eigenval -. min_eval))
    else
      base_metric

  (* Update metric using adaptation rate *)
  let adapt_metric old_metric new_metric params =
    Tensor.add
      (Tensor.mul_scalar old_metric (1.0 -. params.adaptation_rate))
      (Tensor.mul_scalar new_metric params.adaptation_rate)
end

(* Local Equalization *)
module LocalEqualization = struct
  type equalization_params = {
    neighborhood_size: int;
    min_points: int;
    smoothing: float;
  }

  let default_params = {
    neighborhood_size = 20;
    min_points = 10;
    smoothing = 0.1;
  }

  (* Compute local statistics *)
  let compute_local_stats data point params =
    let tree = KDTree.build 
      (List.init (Tensor.size data 0) 
         (fun i -> Tensor.select data i 0)) in
    let neighbors = KDTree.find_k_nearest tree 
      params.neighborhood_size point in
    
    let n = List.length neighbors in
    let neighbor_tensor = Tensor.stack neighbors 0 in
    let mean = Tensor.mean neighbor_tensor 0 in
    let centered = Tensor.sub neighbor_tensor 
      (Tensor.expand mean [n; -1]) in
    let cov = Tensor.matmul 
      (Tensor.transpose centered 0 1) centered in
    Ok (mean, Tensor.div_scalar cov (float_of_int (n - 1)))

  (* Equalize local distribution *)
  let equalize_local params data point metric =
    let u, s, vt = 
      match Tensor.svd cov with
      | Ok x -> x
      | Error e -> raise (Failure "SVD failed") inw
    
    (* Scale local distribution *)
    let s_inv_sqrt = Tensor.pow s 
      (Tensor.float_vec [-0.5]) in
    let whitening = Tensor.matmul
      (Tensor.matmul u (Tensor.diag s_inv_sqrt))
      vt in
    
    (* Apply metric *)
    let transformed = Tensor.matmul
      (Tensor.matmul whitening metric)
      (Tensor.transpose whitening 0 1) in
    
    Ok transformed
end

(* Sequential principle curves aanalysis *)
module SPCA = struct
  type params = {
    n_components: int;
    max_iter: int;
    tol: float;
    metric_params: MetricTensor.params;
    equalization_params: LocalEqualization.equalization_params;
    pdf_params: PDFEstimator.params;
    curve_params: PrincipalCurve.params;
  }

  type t = {
    params: params;
    pdf_estimator: PDFEstimator.t;
    curves: PrincipalCurve.t list;
    metrics: (point * matrix) list;
  }

  let default_params = {
    n_components = 2;
    max_iter = 100;
    tol = 1e-6;
    metric_params = MetricTensor.default_params;
    equalization_params = LocalEqualization.default_params;
    pdf_params = PDFEstimator.default_params;
    curve_params = PrincipalCurve.default_params;
  }

  let create ?(params=default_params) data =
    match PDFEstimator.create ~params:params.pdf_params data with
    | Error e -> Error e
    | Ok pdf_estimator ->
        Ok {
          params;
          pdf_estimator;
          curves = [];
          metrics = [];
        }

  (* Compute sequence of principal curves *)
  let compute_curves t data =
    let rec compute_next_curve curves n =
      if n >= t.params.n_components then
        Ok curves
      else
        match PrincipalCurve.create 
          ~params:t.params.curve_params t.pdf_estimator with
        | Error e -> Error e
        | Ok curve ->
            match PrincipalCurve.fit curve data with
            | Error e -> Error e
            | Ok fitted_curve ->
                compute_next_curve (fitted_curve :: curves) (n + 1)
    in
    compute_next_curve [] 0

  (* Transform point using SPCA *)
  let transform t point =
    (* Project along sequence of curves *)
    let rec project_point curves point acc =
      match curves with
      | [] -> Ok (List.rev acc)
      | curve :: rest ->
          let proj = PrincipalCurve.project curve point in
          let pdf_val = PDFEstimator.estimate_density 
            t.pdf_estimator proj in
          
          (* Compute local metric *)
          match compute_local_stats data proj 
            t.params.equalization_params with
          | Error e -> Error e
          | Ok (_, local_cov) ->
              let metric = MetricTensor.compute_metric
                t.params.metric_params pdf_val proj local_cov in
              
              (* Compute length along curve *)
              let length = compute_curve_length 
                curve point proj metric in
              project_point rest proj (length :: acc)
    in
    project_point t.curves point []

  (* Inverse transform *)
  let inverse_transform t coords =
    let rec reconstruct_point curves coords point =
      match curves, coords with
      | [], [] -> Ok point
      | curve :: rest_curves, coord :: rest_coords ->
          let proj = PrincipalCurve.project curve point in
          let tangent = PrincipalCurve.tangent curve proj in
          let new_point = Tensor.add proj 
            (Tensor.mul_scalar tangent coord) in
          reconstruct_point rest_curves rest_coords new_point
    in
    reconstruct_point t.curves coords (List.hd t.curves).points
end
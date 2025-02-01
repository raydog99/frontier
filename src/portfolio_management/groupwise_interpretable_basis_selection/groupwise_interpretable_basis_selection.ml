open Torch

type index_set = int list
  
(* Support set type for beta coefficients *)
type support = {
    indices: int list;
    dimension: int;
}
  
(* Cluster type for prototype clustering *)
type cluster = {
    points: Tensor.t;
    prototype: Tensor.t;
    prototype_idx: int;
    radius: float;
    members: index_set;
}

(* Asset categories *)
type asset_category =
    | Bond
    | Commodity  
    | Currency
    | DiversifiedPortfolio
    | Equity
    | Alternative
    | Inverse
    | Leveraged
    | RealEstate
    | Volatility

(* Model parameters *)
type model_params = {
    lambda: float;
    max_iterations: int;
    convergence_threshold: float;
    correlation_threshold: float;
}

(* ETF data types *)
type etf_metadata = {
    symbol: string;
    name: string;
    asset_class: asset_category;
    expense_ratio: float;
    inception_date: string;
    avg_volume: float;
    total_assets: float;
}

type etf_timeseries = {
    dates: string array;
    prices: Tensor.t;
    volumes: Tensor.t;
    returns: Tensor.t;
    adjusted_returns: Tensor.t;
}

type etf_record = {
    metadata: etf_metadata;
    timeseries: etf_timeseries;
}

(* Statistics types *)
type regression_result = {
    coefficients: Tensor.t;
    std_errors: Tensor.t;
    t_statistics: Tensor.t;
    r_squared: float;
    residuals: Tensor.t;
}

type significance_matrix = {
    count_matrix: Tensor.t;
    proportion_matrix: Tensor.t;
}

(* Create Jn matrix: n×n matrix of ones *)
let create_jn n =
    Tensor.ones [n; n]

(* Create J̄n matrix: 1/n * Jn *)
let create_jn_bar n =
    let jn = create_jn n in
    Tensor.div_scalar jn (float_of_int n)

(* Create identity matrix with optional dimension specification *)
let create_identity ?n () =
    match n with
    | Some dim -> Tensor.eye dim
    | None -> Tensor.eye 1

(* Matrix subscription for index sets *)
let matrix_subset mat index_set =
    let indices = Tensor.of_int1 index_set in
    Tensor.index_select mat ~dim:1 ~index:indices

(* Column operations *)
let column_means mat = 
    Tensor.mean mat ~dim:[0] ~keepdim:false

let column_vars mat =
    let means = column_means mat in
    let centered = Tensor.sub mat (Tensor.expand_as means mat) in
    Tensor.mean (Tensor.pow centered ~exponent:2.) ~dim:[0] ~keepdim:false

(* Safe matrix operations *)
let safe_matmul a b =
    let a_norm = Tensor.norm a |> Tensor.float_value in
    let b_norm = Tensor.norm b |> Tensor.float_value in
    let scale_factor = sqrt (max a_norm b_norm) in
    if scale_factor > 1e8 then
      let a_scaled = Tensor.div_scalar a scale_factor in
      let b_scaled = Tensor.div_scalar b scale_factor in
      Tensor.mul_scalar (Tensor.matmul a_scaled b_scaled) scale_factor
    else
      Tensor.matmul a b

(* L1 norm *)
let l1_norm v =
    Tensor.sum (Tensor.abs v) ~dim:[0] ~keepdim:false
    |> Tensor.float_value

(* L2 norm *)
let l2_norm v =
    Tensor.sum (Tensor.pow v ~exponent:2.) ~dim:[0] ~keepdim:false
    |> Tensor.sqrt
    |> Tensor.float_value

(* L-infinity norm *)
let linf_norm v =
    Tensor.max v ~dim:[0] ~keepdim:false
    |> fst
    |> Tensor.float_value

(* Support of a vector - indices of non-zero elements *)
let support v =
    let indices = 
      Tensor.nonzero v
      |> Tensor.squeeze
      |> Tensor.to_list1 Int64
      |> List.map Int64.to_int
    in
    {indices; dimension = Tensor.size v |> List.hd}

(* Basic statistical functions *)
let mean x = Tensor.mean x ~dim:[0] ~keepdim:false

let var x =
    let x_mean = mean x in
    let centered = Tensor.sub x (Tensor.expand_as x_mean x) in
    Tensor.mean (Tensor.pow centered ~exponent:2.) ~dim:[0] ~keepdim:false

let std x = Tensor.sqrt (var x)

(* Correlation matrix *)
let correlation_matrix x =
    let n = Tensor.size1 x in
    let x_centered = Tensor.sub x (mean x) in
    let std_x = std x in
    let normalized = Tensor.div x_centered (Tensor.expand_as std_x x) in
    Tensor.matmul (Tensor.transpose normalized ~dim0:0 ~dim1:1) normalized
    |> fun x -> Tensor.div_scalar x (float_of_int (n - 1))

(* OLS regression *)
let ols y x =
    let x_t = Tensor.transpose x ~dim0:0 ~dim1:1 in
    let xtx = Tensor.matmul x_t x in
    let xtx_inv = Tensor.inverse xtx in
    let xty = Tensor.matmul x_t y in
    let beta = Tensor.matmul xtx_inv xty in
    
    (* Compute residuals *)
    let y_hat = Tensor.matmul x beta in
    let residuals = Tensor.sub y y_hat in
    
    (* Compute R-squared *)
    let ss_tot = Tensor.sum (Tensor.pow (Tensor.sub y (mean y)) ~exponent:2.) in
    let ss_res = Tensor.sum (Tensor.pow residuals ~exponent:2.) in
    let r_squared = Tensor.sub (Tensor.ones []) (Tensor.div ss_res ss_tot)
      |> Tensor.float_value in
    
    (* Compute standard errors *)
    let n, p = Tensor.size2 x in
    let sigma_squared = Tensor.div (Tensor.sum (Tensor.pow residuals ~exponent:2.)) 
      (Tensor.full [] (float_of_int (n - p))) in
    let std_errors = Tensor.sqrt (Tensor.mul sigma_squared (Tensor.diag xtx_inv)) in
    
    (* Compute t-statistics *)
    let t_stats = Tensor.div beta std_errors in
    
    {coefficients = beta;
     std_errors;
     t_statistics = t_stats;
     r_squared;
     residuals}

(* Statistical tests *)
let t_test t_stat df =
    (* Approximate t-distribution CDF *)
    let x = t_stat /. sqrt (float_of_int df) in
    0.5 *. (1. +. (2. /. sqrt Float.pi) *. 
      (x +. x*.x*.x/.3. +. x*.x*.x*.x*.x/.10.))

let chi_square_test stat df =
(* Approximate chi-square CDF *)
    let x = stat /. 2. in
    let k = float_of_int df /. 2. in
    let rec gamma_inc x k acc n =
      if n > 100 then acc
      else
        let term = exp (k *. log x -. x -. log (float_of_int n)) in
        if term < 1e-10 then acc
        else gamma_inc x k (acc +. term) (n + 1)
    in
    gamma_inc x k 0. 1

module Clustering = struct
  (* Distance metric computation *)
  let compute_distance r1 r2 =
    let corr = correlation_matrix (Tensor.stack [r1; r2] ~dim:0)
      |> fun x -> Tensor.get x [|0; 1|] in
    1. -. abs_float corr

  (* Maximum distance from point to cluster *)
  let compute_max_distance x cluster =
    let distances = Tensor.empty [Tensor.size1 cluster.points] in
    
    for i = 0 to Tensor.size1 cluster.points - 1 do
      let point = Tensor.select cluster.points ~dim:0 ~index:i in
      let dist = compute_distance x point in
      Tensor.set distances [|i|] dist
    done;
    
    Tensor.max distances |> fst |> Tensor.float_value

  (* Compute minimax radius *)
  let compute_minimax_radius points =
    let n = Tensor.size1 points in
    let min_max_dist = ref Float.max_float in
    let prototype_idx = ref 0 in
    
    for i = 0 to n - 1 do
      let candidate = Tensor.select points ~dim:0 ~index:i in
      let max_dist = compute_max_distance candidate points in
      if max_dist < !min_max_dist then begin
        min_max_dist := max_dist;
        prototype_idx := i
      end
    done;
    
    (!min_max_dist, !prototype_idx)

  (* Create cluster from points *)
  let create_cluster points =
    let (radius, proto_idx) = compute_minimax_radius points in
    {
      points;
      prototype = Tensor.select points ~dim:0 ~index:proto_idx;
      prototype_idx = proto_idx;
      radius;
      members = List.init (Tensor.size1 points) (fun x -> x);
    }

  (* Merge clusters using minimax linkage *)
  let merge_clusters c1 c2 =
    let combined_points = Tensor.cat [c1.points; c2.points] ~dim:0 in
    let (radius, proto_idx) = compute_minimax_radius combined_points in
    {
      points = combined_points;
      prototype = Tensor.select combined_points ~dim:0 ~index:proto_idx;
      prototype_idx = proto_idx;
      radius;
      members = c1.members @ c2.members;
    }

  (* Main clustering function *)
  let cluster points config =
    (* Initialize singleton clusters *)
    let initial_clusters = List.init (Tensor.size1 points) (fun i ->
      let point = Tensor.select points ~dim:0 ~index:i in
      {
        points = Tensor.unsqueeze point ~dim:0;
        prototype = point;
        prototype_idx = 0;
        radius = 0.;
        members = [i];
      }
    ) in
    
    let rec merge_step clusters =
      if List.length clusters <= 1 then
        clusters
      else
        (* Find closest pair *)
        let min_dist = ref Float.max_float in
        let merge_pair = ref (0, 0) in
        
        List.iteri (fun i ci ->
          List.iteri (fun j cj ->
            if i < j then
              let merged = merge_clusters ci cj in
              if merged.radius < !min_dist then begin
                min_dist := merged.radius;
                merge_pair := (i, j)
              end
          ) clusters
        ) clusters;
        
        if !min_dist > config.correlation_threshold then
          clusters
        else
          (* Perform merge *)
          let (i, j) = !merge_pair in
          let ci = List.nth clusters i in
          let cj = List.nth clusters j in
          let merged = merge_clusters ci cj in
          let remaining = List.filteri 
            (fun idx _ -> idx <> i && idx <> j) clusters in
          merge_step (merged :: remaining)
    in
    
    merge_step initial_clusters
end

module Lasso = struct
  (* Default parameters *)
  let default_params = {
    lambda = 0.1;
    max_iterations = 1000;
    convergence_threshold = 1e-6;
    correlation_threshold = 0.5;
  }

  (* Soft thresholding operator *)
  let soft_threshold x lambda =
    let sign = Tensor.sign x in
    let abs_x = Tensor.abs x in
    Tensor.mul sign (Tensor.maximum 
      (Tensor.sub abs_x (Tensor.full [] lambda)) 
      (Tensor.zeros []))

  (* LASSO estimation *)
  let fit_lasso ?(params=default_params) y x =
    let n, p = Tensor.size2 x in
    let beta = Tensor.zeros [p] in
    let x_t = Tensor.transpose x ~dim0:0 ~dim1:1 in
    
    let rec iterate beta iter =
      if iter >= params.max_iterations then beta
      else
        let residuals = Tensor.sub y (Tensor.matmul x beta) in
        let grad = Tensor.matmul x_t residuals in
        let beta_new = soft_threshold grad params.lambda in
        
        let diff = Tensor.sub beta_new beta in
        let converged = 
          Tensor.norm diff ~p:2
          |> Tensor.float_value 
          |> (fun x -> x < params.convergence_threshold)
        in
        
        if converged then beta_new
        else iterate beta_new (iter + 1)
    in
    
    iterate beta 0

  (* Lambda selection with cross-validation *)
  let select_lambda y x n_folds =
    let n = Tensor.size1 x in
    let fold_size = n / n_folds in
    
    let lambda_grid = Array.init 100 (fun i ->
      1e-4 *. (1.1 ** float_of_int i))
    in
    
    let mse = Array.make (Array.length lambda_grid) 0. in
    
    for k = 0 to n_folds - 1 do
      let val_start = k * fold_size in
      let val_end = min (val_start + fold_size) n in
      
      let train_x = Tensor.cat [
        Tensor.narrow x ~dim:0 ~start:0 ~length:val_start;
        Tensor.narrow x ~dim:0 ~start:val_end ~length:(n - val_end)
      ] ~dim:0 in
      
      let train_y = Tensor.cat [
        Tensor.narrow y ~dim:0 ~start:0 ~length:val_start;
        Tensor.narrow y ~dim:0 ~start:val_end ~length:(n - val_end)
      ] ~dim:0 in
      
      let val_x = Tensor.narrow x ~dim:0 
        ~start:val_start ~length:(val_end - val_start) in
      let val_y = Tensor.narrow y ~dim:0 
        ~start:val_start ~length:(val_end - val_start) in
      
      Array.iteri (fun i lambda ->
        let beta = fit_lasso train_y train_x ~params:{
          default_params with lambda
        } in
        let pred = Tensor.matmul val_x beta in
        let fold_mse = Tensor.mse_loss pred val_y 
          ~reduction:Reduction.Mean
          |> Tensor.float_value in
        mse.(i) <- mse.(i) +. fold_mse /. float_of_int n_folds
      ) lambda_grid
    done;
    
    (* Find minimum MSE and corresponding lambda *)
    let min_mse = Array.fold_left min max_float mse in
    let best_idx = Array.find_index ((=.) min_mse) mse in
    lambda_grid.(best_idx)
end

(* Groupwise intrepretable basis selection algorithm *)
module Gibs = struct
  (* GIBS algorithm configuration *)
  type gibs_config = {
    correlation_threshold: float;
    min_observations: int;
    max_basis_assets: int;
    lambda_multiplier: float;
    include_ff5: bool;
  }

  (* Transform basis assets *)
  let transform_basis_assets market_return assets =
    let n = Array.length assets in
    let transformed = Array.make n (Tensor.zeros []) in
    transformed.(0) <- market_return;  (* Keep market return as is *)
    
    for i = 1 to n - 1 do
      let asset = assets.(i) in
      let x1 = Tensor.transpose market_return ~dim0:0 ~dim1:1 in
      let x1_inv = Tensor.inverse (Tensor.matmul x1 market_return) in
      let proj = Tensor.matmul 
        (Tensor.matmul (Tensor.matmul x1 x1_inv) x1)
        asset in
      transformed.(i) <- Tensor.sub asset proj
    done;
    transformed

  (* Select basis assets within category *)
  let select_category_basis assets config =
    let points = Array.map (fun a -> a.timeseries.returns) assets |>
      Tensor.stack ~dim:0 in
    
    let clusters = Clustering.cluster points {
      correlation_threshold = config.correlation_threshold;
      max_iterations = 1000;
      convergence_threshold = 1e-6;
    } in
    
    List.map (fun cluster ->
      Array.get assets cluster.prototype_idx
    ) clusters

  (* Main GIBS algorithm *)
  let run stocks basis_assets config =
    (* Step 1: Transform basis assets *)
    let market_return = (Array.get basis_assets 0).timeseries.returns in
    let transformed = transform_basis_assets market_return basis_assets in
    
    (* Step 2: Group by category *)
    let categorized = Hashtbl.create 10 in
    Array.iteri (fun i asset ->
      let current = try
        Hashtbl.find categorized asset.metadata.asset_class
      with Not_found -> [] in
      Hashtbl.replace categorized asset.metadata.asset_class 
        (asset :: current)
    ) basis_assets;
    
    (* Step 3: Select representatives from each category *)
    let representatives = Hashtbl.fold (fun _ assets acc ->
      let selected = select_category_basis (Array.of_list assets) config in
      selected @ acc
    ) categorized [] in
    
    (* Step 4: Modified LASSO regression *)
    let n_stocks = Array.length stocks in
    let rep_returns = List.map (fun r -> r.timeseries.returns) representatives |>
      Array.of_list |> Tensor.stack ~dim:0 in
    
    let betas = Array.make n_stocks (Tensor.zeros []) in
    let significant_sets = Array.make n_stocks [] in
    
    Array.iteri (fun i stock ->
      let stock_returns = stock.timeseries.returns in
      
      (* Select optimal lambda *)
      let lambda = Lasso.select_lambda stock_returns rep_returns 5 in
      
      (* Fit LASSO *)
      let beta = Lasso.fit_lasso stock_returns rep_returns
        ~params:{Lasso.default_params with 
                lambda = lambda *. config.lambda_multiplier} in
      
      betas.(i) <- beta;
      
      (* Get significant indices *)
      let sig_indices = Tensor.nonzero 
        (Tensor.gt (Tensor.abs beta) (Tensor.full [] 1e-5))
        |> Tensor.squeeze
        |> Tensor.to_list1 Int64
        |> List.map Int64.to_int in
      
      significant_sets.(i) <- sig_indices
    ) stocks;
    
    (* Step 5: Compute significance matrices *)
    let basis_classes = List.map (fun r -> r.metadata.asset_class) representatives in
    let stock_classes = Array.map (fun s -> s.metadata.asset_class) stocks |>
      Array.to_list in
    
    let count_matrix = SignificanceMatrices.compute_basis_counts 
      stock_classes basis_classes significant_sets in
    let proportion_matrix = SignificanceMatrices.compute_proportions count_matrix in
    
    (betas, significant_sets, count_matrix, proportion_matrix)
end

module SignificanceMatrices = struct
  (* Compute count matrix *)
  let compute_basis_counts security_classes basis_classes significant_sets =
    let m = List.length basis_classes in
    let l = List.length security_classes in
    let counts = Tensor.zeros [m; l] in
    
    (* Map classes to indices *)
    let basis_indices = List.mapi (fun i c -> (c, i)) basis_classes |> 
      List.to_seq |> Hashtbl.of_seq in
    let security_indices = List.mapi (fun i c -> (c, i)) security_classes |>
      List.to_seq |> Hashtbl.of_seq in
    
    (* Count significant basis assets for each security class *)
    Array.iteri (fun sec_idx sig_set ->
      let sec_class = Array.get security_classes sec_idx in
      let d = Hashtbl.find security_indices sec_class in
      
      List.iter (fun basis_idx ->
        let basis_class = List.nth basis_classes basis_idx in
        let b = Hashtbl.find basis_indices basis_class in
        let current = Tensor.get counts [|b; d|] in
        Tensor.set counts [|b; d|] (current +. 1.)
      ) sig_set
    ) significant_sets;
    
    counts

  (* Compute proportion matrix *)
  let compute_proportions count_matrix =
    let column_sums = Tensor.sum count_matrix ~dim:[1] ~keepdim:true in
    let props = Tensor.div count_matrix column_sums in
    
    (* Handle division by zero *)
    Tensor.where
      (Tensor.eq column_sums (Tensor.zeros_like column_sums))
      (Tensor.zeros_like props)
      props
end

module FactorAnalysis = struct
  (* Factor analysis configuration *)
  type factor_config = {
    min_variance_explained: float;
    max_correlation: float;
    significance_level: float;
    stability_threshold: float;
  }

  (* Compute factor exposures *)
  let compute_factor_exposures returns basis_assets rf_rate =
    let n_securities = Tensor.size1 returns in
    let n_basis = Tensor.size1 basis_assets in
    
    (* Compute excess returns *)
    let excess_returns = Tensor.sub returns 
      (Tensor.expand_as rf_rate returns) in
    let excess_basis = Tensor.sub basis_assets
      (Tensor.expand_as rf_rate basis_assets) in
    
    (* Initialize coefficient matrices *)
    let alphas = Tensor.zeros [n_securities] in
    let betas = Tensor.zeros [n_securities; n_basis] in
    let t_stats = Tensor.zeros [n_securities; n_basis + 1] in
    let r_squared = Tensor.zeros [n_securities] in
    
    for i = 0 to n_securities - 1 do
      let security_returns = Tensor.select excess_returns ~dim:0 ~index:i in
      
      (* Add constant term for alpha *)
      let x = Tensor.cat [
        Tensor.ones [Tensor.size1 excess_basis; 1];
        Tensor.transpose excess_basis ~dim0:0 ~dim1:1
      ] ~dim:1 in
      
      let reg_result = ols security_returns x in
      
      Tensor.set alphas [|i|] (Tensor.get reg_result.coefficients [|0|]);
      Tensor.copy_ (Tensor.select betas ~dim:0 ~index:i) 
        (Tensor.narrow reg_result.coefficients ~dim:0 ~start:1 
         ~length:n_basis);
      Tensor.copy_ (Tensor.select t_stats ~dim:0 ~index:i) 
        reg_result.t_statistics;
      Tensor.set r_squared [|i|] reg_result.r_squared
    done;
    
    (alphas, betas, t_stats, r_squared)

  (* Analyze factor stability *)
  let analyze_stability returns factors window_size =
    let n_periods = Tensor.size1 returns - window_size + 1 in
    let rolling_betas = Tensor.zeros [n_periods; Tensor.size2 factors] in
    
    for t = 0 to n_periods - 1 do
      let window_returns = Tensor.narrow returns ~dim:0 
        ~start:t ~length:window_size in
      let window_factors = Tensor.narrow factors ~dim:0 
        ~start:t ~length:window_size in
      
      let reg_result = ols window_returns window_factors in
      Tensor.copy_ (Tensor.select rolling_betas ~dim:0 ~index:t)
        reg_result.coefficients
    done;
    
    let beta_means = Tensor.mean rolling_betas ~dim:0 in
    let beta_stds = Tensor.std rolling_betas ~dim:0 ~unbiased:true in
    
    (* Compute coefficient of variation *)
    let cv = Tensor.div beta_stds (Tensor.abs beta_means) in
    Tensor.div_scalar (Tensor.ones_like cv) (Tensor.add_scalar cv 1.)

  (* Select significant factors *)
  let select_factors returns factors config =
    let n_factors = Tensor.size2 factors in
    let selected = ref [] in
    
    (* Compute factor loadings and stability *)
    let reg_result = ols returns factors in
    let stability = analyze_stability returns factors 50 in
    
    for i = 0 to n_factors - 1 do
      let loading = Tensor.get reg_result.coefficients [|i|] in
      let t_stat = Tensor.get reg_result.t_statistics [|i|] in
      let stab_score = Tensor.get stability [|i|] in
      
      if abs_float t_stat > 1.96 &&  (* 5% significance level *)
         stab_score > config.stability_threshold then
        selected := i :: !selected
    done;
    
    (* Check variance explained *)
    let selected_factors = List.map (fun i ->
      Tensor.select factors ~dim:1 ~index:i
    ) !selected |> 
      Array.of_list |> 
      Tensor.stack ~dim:1 in
    
    let reg_selected = ols returns selected_factors in
    if reg_selected.r_squared > config.min_variance_explained then
      !selected
    else
      []

  (* Orthogonalize factors *)
  let orthogonalize_factors factors =
    let n_factors = Tensor.size1 factors in
    let orthogonalized = Tensor.copy factors in
    
    for i = 1 to n_factors - 1 do
      let factor = Tensor.select orthogonalized ~dim:0 ~index:i in
      
      (* Project out previous factors *)
      for j = 0 to i - 1 do
        let prev_factor = Tensor.select orthogonalized ~dim:0 ~index:j in
        let x_t = Tensor.transpose prev_factor ~dim0:0 ~dim1:1 in
        let proj_matrix = Tensor.matmul 
          (Tensor.matmul prev_factor 
            (Tensor.inverse (Tensor.matmul x_t prev_factor)))
          x_t in
        let projection = Tensor.matmul proj_matrix factor in
        Tensor.copy_ factor (Tensor.sub factor projection)
      done;
      
      (* Normalize *)
      let norm = Tensor.norm factor |> Tensor.float_value in
      Tensor.copy_ factor (Tensor.div_scalar factor norm)
    done;
    
    orthogonalized
end
open Torch

let hierarchical_risk_parity returns =
  let corr_matrix = Tensor.corrcoef returns in
  let distance_matrix = Tensor.sub (Tensor.ones_like corr_matrix) corr_matrix in
  
  let rec cluster clusters =
    if List.length clusters = 1 then List.hd clusters
    else
      let min_distance = ref Float.max_float in
      let min_pair = ref (0, 0) in
      List.iteri (fun i ci ->
        List.iteri (fun j cj ->
          if i < j then
            let dist = Tensor.get distance_matrix [ci; cj] in
            if dist < !min_distance then (
              min_distance := dist;
              min_pair := (i, j)
            )
        ) clusters
      ) clusters;
      let (i, j) = !min_pair in
      let new_cluster = List.nth clusters i @ List.nth clusters j in
      cluster (new_cluster :: (List.filteri (fun idx _ -> idx <> i && idx <> j) clusters))
  in
  let tree = cluster (List.init (Tensor.shape1_exn returns) (fun i -> [i])) in
  
  let rec bisect cluster weights =
    match cluster with
    | [idx] -> [(idx, List.hd weights)]
    | _ ->
        let n = List.length cluster in
        let left = List.take (n / 2) cluster in
        let right = List.drop (n / 2) cluster in
        let left_var = Tensor.sum (Tensor.gather returns ~dim:0 ~index:(Tensor.of_int1 (Array.of_list left))) in
        let right_var = Tensor.sum (Tensor.gather returns ~dim:0 ~index:(Tensor.of_int1 (Array.of_list right))) in
        let left_weight = left_var /. (left_var +. right_var) in
        let right_weight = 1. -. left_weight in
        bisect left (List.map (fun w -> w *. left_weight) weights) @
        bisect right (List.map (fun w -> w *. right_weight) weights)
  in
  let initial_weights = List.init (List.length tree) (fun _ -> 1. /. float_of_int (List.length tree)) in
  let hrp_weights = bisect tree initial_weights in
  List.sort (fun (a, _) (b, _) -> compare a b) hrp_weights |> List.map snd

let robust_mean_variance_optimization returns covariance_matrix risk_free_rate confidence_level =
  let num_assets = Tensor.shape1_exn returns in
  let mean_returns = Tensor.mean returns ~dim:[1] in
  let std_returns = Tensor.std returns ~dim:[1] ~unbiased:true in
  let z_score = Torch_normal.ppf confidence_level in
  
  let adjusted_returns = Tensor.sub mean_returns (Tensor.mul (Tensor.scalar_tensor z_score) std_returns) in
  let adjusted_cov = Tensor.mul covariance_matrix (Tensor.scalar_tensor (1. +. z_score *. z_score)) in
  
  let ones = Tensor.ones [num_assets; 1] in
  let a = Tensor.mm (Tensor.mm (Tensor.transpose adjusted_returns ~dim0:0 ~dim1:1) (Tensor.inverse adjusted_cov)) adjusted_returns |> Tensor.to_float0_exn in
    let b = Tensor.mm (Tensor.mm (Tensor.transpose adjusted_returns ~dim0:0 ~dim1:1) (Tensor.inverse adjusted_cov)) ones |> Tensor.to_float0_exn in
    let c = Tensor.mm (Tensor.mm (Tensor.transpose ones ~dim0:0 ~dim1:1) (Tensor.inverse adjusted_cov)) ones |> Tensor.to_float0_exn in
    
    let lambda = (c *. risk_free_rate -. b) /. (a *. c -. b *. b) in
    let gamma = (a -. b *. risk_free_rate) /. (a *. c -. b *. b) in
    
    Tensor.add 
      (Tensor.mul (Tensor.scalar_tensor lambda) (Tensor.mm (Tensor.inverse adjusted_cov) adjusted_returns))
      (Tensor.mul (Tensor.scalar_tensor gamma) (Tensor.mm (Tensor.inverse adjusted_cov) ones))

let cdar_optimization returns alpha beta =
  let num_assets = Tensor.shape1_exn returns in
  let num_periods = Tensor.shape1_exn returns in
  
  let w = Tensor.randn [num_assets] in
  let u = Tensor.randn [num_periods] in
  let z = Tensor.randn [] in
  
  let objective w u z =
    Tensor.add (Tensor.scalar_tensor ((1. -. beta) /. alpha)) (Tensor.sum u) |> Tensor.add z
  in
  
  let constraints w u z =
    let portfolio_returns = Tensor.mm returns (Tensor.unsqueeze w ~dim:1) in
    let drawdowns = Tensor.sub (Tensor.cummax portfolio_returns ~dim:0 |> fst) portfolio_returns in
    [
      Tensor.sum w |> Tensor.sub (Tensor.scalar_tensor 1.);  (* Sum of weights = 1 *)
      Tensor.le drawdowns (Tensor.add z u);  (* Drawdown constraint *)
      Tensor.ge w (Tensor.zeros_like w);  (* Non-negative weights *)
    ]
  in
  
  let rec optimize iter w u z =
    if iter > 1000 then w
    else
      let obj = objective w u z in
      let constrs = constraints w u z in
      let grad_w, grad_u, grad_z = Tensor.grad [w; u; z] obj in
      let new_w = Tensor.sub w (Tensor.mul grad_w (Tensor.scalar_tensor 0.01)) in
      let new_u = Tensor.sub u (Tensor.mul grad_u (Tensor.scalar_tensor 0.01)) in
      let new_z = Tensor.sub z (Tensor.mul grad_z (Tensor.scalar_tensor 0.01)) in
      optimize (iter + 1) new_w new_u new_z
  in
  optimize 0 w u z
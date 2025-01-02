open Torch
open Glpk

module MeanVariance = struct
  type t = {
    mutable weights: Tensor.t;
    mutable expected_return: float;
    mutable risk: float;
  }

  let create num_assets =
    {
      weights = Tensor.ones [num_assets] |> Tensor.div_ (Tensor.float_scalar (float num_assets));
      expected_return = 0.;
      risk = 0.;
    }

  let calculate_portfolio_return weights returns =
    Tensor.(sum (mul weights returns) |> to_float0_exn)

  let calculate_portfolio_risk weights covariance_matrix =
    let open Tensor in
    matmul (matmul weights (expand covariance_matrix ~size:[-1; -1])) (unsqueeze weights 1)
    |> squeeze ~dim:[0; 1]
    |> to_float0_exn

  let optimize t returns covariance_matrix target_return =
    let num_assets = Tensor.size t.weights |> List.hd in
    let prob = Glpk.make_problem () in
    Glpk.set_obj_dir prob Glpk.Min;
    
    let vars = Array.init num_assets (fun _ -> Glpk.add_col prob) in
    Array.iter (fun v -> Glpk.set_col_bnds prob v Glpk.Db 0. 1.) vars;
    
    Array.iteri (fun i v ->
      Array.iteri (fun j u ->
        let cov = Tensor.get covariance_matrix [i; j] |> Tensor.to_float0_exn in
        Glpk.set_obj_coef prob v (cov *. (if i = j then 1. else 2.))
      ) vars
    ) vars;
    
    let sum_to_one = Glpk.add_row prob in
    Glpk.set_row_bnds prob sum_to_one Glpk.Fx 1. 1.;
    Array.iter (fun v -> Glpk.set_mat_row prob sum_to_one [|v, 1.|]) vars;
    
    let return_constraint = Glpk.add_row prob in
    Glpk.set_row_bnds prob return_constraint Glpk.Lo target_return target_return;
    Array.iteri (fun i v ->
      let r = Tensor.get returns [i] |> Tensor.to_float0_exn in
      Glpk.set_mat_row prob return_constraint [|v, r|]
    ) vars;
    
    Glpk.simplex prob;
    
    let solution = Array.map (Glpk.get_col_prim prob) vars in
    t.weights <- Tensor.of_float1 (Array.to_list solution)

  let update t returns =
    let portfolio_return = calculate_portfolio_return t.weights returns in
    t.expected_return <- portfolio_return;
    ()
end

module MeanAbsoluteDeviation = struct
  type t = {
    mutable weights: Tensor.t;
    mutable expected_return: float;
    mutable risk: float;
  }

  let create num_assets =
    {
      weights = Tensor.ones [num_assets] |> Tensor.div_ (Tensor.float_scalar (float num_assets));
      expected_return = 0.;
      risk = 0.;
    }

  let optimize t returns target_return =
    let num_assets = Tensor.size t.weights |> List.hd in
    let num_samples = Tensor.size returns |> List.nth 1 in
    let prob = Glpk.make_problem () in
    Glpk.set_obj_dir prob Glpk.Min;
    
    let w = Array.init num_assets (fun _ -> Glpk.add_col prob) in
    Array.iter (fun v -> Glpk.set_col_bnds prob v Glpk.Db 0. 1.) w;
    let y = Array.init num_samples (fun _ -> Glpk.add_col prob) in
    Array.iter (fun v -> Glpk.set_col_bnds prob v Glpk.Fr 0. 0.) y;
    
    Array.iter (fun v -> Glpk.set_obj_coef prob v (1. /. float num_samples)) y;
    
    let sum_to_one = Glpk.add_row prob in
    Glpk.set_row_bnds prob sum_to_one Glpk.Fx 1. 1.;
    Array.iter (fun v -> Glpk.set_mat_row prob sum_to_one [|v, 1.|]) w;
    
    let return_constraint = Glpk.add_row prob in
    Glpk.set_row_bnds prob return_constraint Glpk.Lo target_return target_return;
    Array.iteri (fun i v ->
      let r = Tensor.get returns [i] |> Tensor.mean |> Tensor.to_float0_exn in
      Glpk.set_mat_row prob return_constraint [|v, r|]
    ) w;
    
    for t = 0 to num_samples - 1 do
      let pos_dev = Glpk.add_row prob in
      let neg_dev = Glpk.add_row prob in
      Glpk.set_row_bnds prob pos_dev Glpk.Lo 0. 0.;
      Glpk.set_row_bnds prob neg_dev Glpk.Lo 0. 0.;
      Array.iteri (fun i v ->
        let r = Tensor.get returns [i; t] |> Tensor.to_float0_exn in
        Glpk.set_mat_row prob pos_dev [|v, r; y.(t), -1.|];
        Glpk.set_mat_row prob neg_dev [|v, -r; y.(t), -1.|]
      ) w
    done;
    
    Glpk.simplex prob;
    
    let solution = Array.map (Glpk.get_col_prim prob) w in
    t.weights <- Tensor.of_float1 (Array.to_list solution);
    t.risk <- Array.fold_left (fun acc v -> acc +. Glpk.get_col_prim prob v) 0. y /. float num_samples

  let update t returns =
    let portfolio_return = MeanVariance.calculate_portfolio_return t.weights returns in
    t.expected_return <- portfolio_return;
    ()
end

module ConditionalValueAtRisk = struct
  type t = {
    mutable weights: Tensor.t;
    mutable expected_return: float;
    mutable risk: float;
    confidence_level: float;
  }

  let create num_assets confidence_level =
    {
      weights = Tensor.ones [num_assets] |> Tensor.div_ (Tensor.float_scalar (float num_assets));
      expected_return = 0.;
      risk = 0.;
      confidence_level;
    }

  let optimize t returns target_return =
    let num_assets = Tensor.size t.weights |> List.hd in
    let num_samples = Tensor.size returns |> List.nth 1 in
    let prob = Glpk.make_problem () in
    Glpk.set_obj_dir prob Glpk.Min;
    
    let w = Array.init num_assets (fun _ -> Glpk.add_col prob) in
    Array.iter (fun v -> Glpk.set_col_bnds prob v Glpk.Db 0. 1.) w;
    let z = Glpk.add_col prob in
    Glpk.set_col_bnds prob z Glpk.Fr 0. 0.;
    let y = Array.init num_samples (fun _ -> Glpk.add_col prob) in
    Array.iter (fun v -> Glpk.set_col_bnds prob v Glpk.Lo 0. 0.) y;
    
    Glpk.set_obj_coef prob z 1.;
    Array.iter (fun v -> Glpk.set_obj_coef prob v (1. /. (float num_samples *. (1. -. t.confidence_level)))) y;
    
    let sum_to_one = Glpk.add_row prob in
    Glpk.set_row_bnds prob sum_to_one Glpk.Fx 1. 1.;
    Array.iter (fun v -> Glpk.set_mat_row prob sum_to_one [|v, 1.|]) w;
    
    let return_constraint = Glpk.add_row prob in
    Glpk.set_row_bnds prob return_constraint Glpk.Lo target_return target_return;
    Array.iteri (fun i v ->
      let r = Tensor.get returns [i] |> Tensor.mean |> Tensor.to_float0_exn in
      Glpk.set_mat_row prob return_constraint [|v, r|]
    ) w;
    
    for t = 0 to num_samples - 1 do
      let cvar_constraint = Glpk.add_row prob in
      Glpk.set_row_bnds prob cvar_constraint Glpk.Lo 0. 0.;
      let coeffs = Array.mapi (fun i v ->
        let r = Tensor.get returns [i; t] |> Tensor.to_float0_exn in
        (v, -r)
      ) w in
      Glpk.set_mat_row prob cvar_constraint (Array.append coeffs [|z, 1.; y.(t), -1.|])
    done;
    
    Glpk.simplex prob;
    
    let solution = Array.map (Glpk.get_col_prim prob) w in
    t.weights <- Tensor.of_float1 (Array.to_list solution);
    t.risk <- Glpk.get_col_prim prob z +. 
              (Array.fold_left (fun acc v -> acc +. Glpk.get_col_prim prob v) 0. y /. 
               (float num_samples *. (1. -. t.confidence_level)))

  let update t returns =
    let portfolio_return = MeanVariance.calculate_portfolio_return t.weights returns in
    t.expected_return <- portfolio_return;
    ()
end
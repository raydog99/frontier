open Torch

let evt_tail_risk returns threshold =
  let exceedances = Tensor.masked_select returns (Tensor.gt returns threshold) in
  let sorted_exceedances = Tensor.sort exceedances ~descending:true in
  let k = min 50 (Tensor.shape1_exn sorted_exceedances) in
  let tail_index = 1. /. (Tensor.mean (Tensor.log (Tensor.slice sorted_exceedances ~dim:0 ~start:0 ~end_:k)) -. 
                          Tensor.log (Tensor.select sorted_exceedances ~dim:0 ~index:(k-1)) 
                          |> Tensor.to_float0_exn)
  in tail_index

module Copula = struct
  type t = 
    | Gaussian of Tensor.t
    | Student of { nu: float; corr: Tensor.t }

  let gaussian_copula data =
    let corr = Tensor.corrcoef data in
    Gaussian corr

  let student_copula data nu =
    let corr = Tensor.corrcoef data in
    Student { nu; corr }

  let sample copula n =
    match copula with
    | Gaussian corr ->
        let l = Tensor.cholesky corr in
        let z = Tensor.randn [n; Tensor.shape1_exn corr] in
        Tensor.mm z l
    | Student { nu; corr } ->
        let l = Tensor.cholesky corr in
        let z = Tensor.randn [n; Tensor.shape1_exn corr] in
        let chi2 = Tensor.randn [n; 1] in
        Tensor.div (Tensor.mm z l) (Tensor.sqrt (Tensor.div chi2 nu))
end

let compute_cvar returns alpha =
  let sorted_returns = Tensor.sort returns ~descending:false in
  let cut_off = int_of_float (float_of_int (Tensor.shape1_exn returns) *. alpha) in
  let tail_returns = Tensor.slice sorted_returns ~dim:0 ~start:0 ~end_:cut_off in
  Tensor.mean tail_returns |> Tensor.to_float0_exn

let stress_test portfolio historical_returns scenarios =
  List.map (fun scenario ->
    let stressed_returns = Tensor.mul historical_returns scenario in
    let portfolio_return = Tensor.dot portfolio stressed_returns in
    Tensor.to_float0_exn portfolio_return
  ) scenarios
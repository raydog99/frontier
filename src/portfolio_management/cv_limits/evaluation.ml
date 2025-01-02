open Base
open Torch
open Types
open Model
open Stats
open Cross_validation

let evaluate_with_cv cv_type model data ~epochs =
    let folds = cross_validate cv_type data in
    let fold_results = List.map folds ~f:(fun (train_data, test_data) ->
      let trained_model = Model.train model train_data ~epochs in
      Model.evaluate trained_model (fst test_data) (snd test_data)
    ) in
    
    let mean = List.fold fold_results ~init:0. ~f:(+.) /. Float.of_int (List.length fold_results) in
    let std_dev = Float.sqrt (List.fold fold_results ~init:0. ~f:(fun acc r -> acc +. Float.((r - mean) ** 2.)) /. Float.of_int (List.length fold_results - 1)) in
    let ci = confidence_interval mean std_dev (List.length fold_results) 0.05 in
    { EvaluationResult.point_estimate = mean; confidence_interval = ci; standard_error = std_dev /. Float.sqrt (Float.of_int (List.length fold_results)) }

let plug_in model data ~epochs =
    let trained_model = Model.train model data ~epochs in
    let x, y = data in
    let result = Model.evaluate trained_model x y in
    let n = Tensor.shape x |> List.hd_exn in
    let std_dev = Float.sqrt (result /. Float.of_int n) in
    let ci = confidence_interval result std_dev n 0.05 in
    { EvaluationResult.point_estimate = result; confidence_interval = ci; standard_error = std_dev /. Float.sqrt (Float.of_int n) }

let k_fold_cv k = evaluate_with_cv (CVType.KFold k)
  let loocv = evaluate_with_cv CVType.LOOCV
  let stratified_k_fold_cv k = evaluate_with_cv (CVType.StratifiedKFold k)
  let time_series_cv n = evaluate_with_cv (CVType.TimeSeriesSplit n)
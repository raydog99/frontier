open Torch

type optimization_method = 
  | QuadraticProgramming
  | GradientDescent
  | SLSQP

let optimize_qp expected_returns covariance_matrix risk_aversion =
  let n = Tensor.shape expected_returns |> List.hd in
  
  let P = Tensor.(mul_scalar covariance_matrix (Float.mul 2.0 risk_aversion)) in
  let q = Tensor.(neg expected_returns) in
  let G = Tensor.(neg (eye n)) in
  let h = Tensor.zeros [n] in
  let A = Tensor.ones [1; n] in
  let b = Tensor.ones [1] in
  
  match Optim.qp ~P ~q ~G ~h ~A ~b () with
  | Ok solution -> solution
  | Error msg -> failwith (Printf.sprintf "QP optimization failed: %s" msg)

let optimize_gradient_descent expected_returns covariance_matrix risk_aversion max_iterations learning_rate =
  let n = Tensor.shape expected_returns |> List.hd in
  let weights = Tensor.((ones [n] / float n) |> set_requires_grad true) in
  
  let rec optimize_loop i =
    if i >= max_iterations then weights
    else begin
      let return = Tensor.(sum (weights * expected_returns)) in
      let risk = Tensor.(matmul (matmul weights (transpose covariance_matrix ~dim0:0 ~dim1:1)) weights) in
      let obj = Tensor.(return - (float risk_aversion * risk)) in
      
      Tensor.backward (Tensor.neg obj);
      Tensor.no_grad (fun () ->
        Tensor.(weights += (grad weights *! float learning_rate));
        Tensor.(weights *= (float 1. / sum weights))  (* Normalize *)
      );
      Tensor.zero_grad weights;
      optimize_loop (i + 1)
    end
  in
  optimize_loop 0

let optimize_slsqp expected_returns covariance_matrix risk_aversion =
  failwith "SLSQP optimization not implemented"

let optimize ?(method_=QuadraticProgramming) expected_returns covariance_matrix risk_aversion =
  match method_ with
  | QuadraticProgramming -> optimize_qp expected_returns covariance_matrix risk_aversion
  | GradientDescent -> optimize_gradient_descent expected_returns covariance_matrix risk_aversion 1000 0.01
  | SLSQP -> optimize_slsqp expected_returns covariance_matrix risk_aversion

let sharpe_ratio portfolio_return portfolio_risk =
  Tensor.(portfolio_return / sqrt portfolio_risk)
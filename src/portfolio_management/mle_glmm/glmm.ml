open Torch
open MatrixOps

type t = {
  spec: Types.model_spec;
  x: Tensor.t;         (* Fixed effects design matrix *)
  z: Tensor.t;         (* Random effects design matrix *)
  y: Tensor.t;         (* Response vector *)
  beta: Tensor.t;      (* Fixed effects parameters *)
  omega: Tensor.t;     (* Variance components *)
}

let create spec x z y =
  {
    spec;
    x;
    z;
    y;
    beta = Tensor.zeros [spec.n_fixed; 1];
    omega = Tensor.ones [2; 1];
  }

let linear_predictor t gamma =
  let fixed = t.x @. t.beta in
  let random = t.z @. gamma in
  Tensor.add fixed random

let compute_working_response t gamma =
  let eta = linear_predictor t gamma in
  match t.spec.distribution with
  | Binomial { trials } ->
      let p = Tensor.sigmoid eta in
      let w = Tensor.mul trials (Tensor.mul p (Tensor.sub (Tensor.ones_like p) p)) in
      let y_tilde = Tensor.add eta 
        (Tensor.div (Tensor.sub t.y (Tensor.mul trials p)) w) in
      (y_tilde, w)
  | Poisson ->
      let mu = Tensor.exp eta in
      let w = mu in
      let y_tilde = Tensor.add eta 
        (Tensor.div (Tensor.sub t.y mu) w) in
      (y_tilde, w)
  | Normal { variance } ->
      let w = Tensor.ones_like eta |> 
              Tensor.mul_scalar (Scalar.float (1.0 /. variance)) in
      (t.y, w)

let compute_r t w delta =
  let w_inv = Tensor.reciprocal w in
  let z_omega = t.z @. (Tensor.exp delta @. Tensor.transpose t.z ~dim0:0 ~dim1:1) in
  Tensor.add w_inv z_omega

let compute_psi t alpha delta gamma =
  let (y_tilde, w) = compute_working_response t gamma in
  let r = compute_r t w delta in
  let diff = Tensor.sub y_tilde (Tensor.add (linear_predictor t gamma) alpha) in
  
  let term1 = log_det r |> Tensor.mul_scalar (Scalar.float (-0.5)) in
  let r_inv = safe_inverse r in
  let term2 = Tensor.mm (Tensor.transpose diff ~dim0:0 ~dim1:1) 
                       (Tensor.mm r_inv diff) |>
              Tensor.mul_scalar (Scalar.float (-0.5)) in
  Tensor.add term1 term2

let fit ?(max_iter=100) ?(tol=1e-6) t =
  let state = Algo.run ~max_iter ~tol t in
  { t with beta = state.beta; omega = state.omega }
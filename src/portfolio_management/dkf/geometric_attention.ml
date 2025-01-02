open Torch

type t = {
  means: Tensor.t;
  covariances: Tensor.t;
}

let create n_0 d_x ~device =
  let means = Tensor.randn [n_0; d_x] ~device in
  let covariances = Tensor.randn [n_0; d_x; d_x] ~device in
  { means; covariances }

let forward t v =
  let weights = Tensor.softmax v ~dim:[0] ~dtype:(T Float) in
  let mean = Tensor.sum (Tensor.mul weights t.means) ~dim:[0] in
  let cov = Tensor.sum (Tensor.mul (Tensor.unsqueeze weights ~dim:2)
                          (Tensor.matmul t.covariances (Tensor.transpose t.covariances ~dim0:1 ~dim1:2)))
              ~dim:[0] in
  (mean, cov)

let parameters t =
  [t.means; t.covariances]
open Torch
open Type
  
let sample_posterior x y config sigma =
  let n = float_of_int (size x 0) in
  let p = size x 1 in
  
  (* Compute posterior mean *)
  let xtx = LinAlg.gram_matrix x in
  let xty = LinAlg.cross_product x y in
  let identity = eye p in
  let precision = add xtx (scalar_mul config.an identity) in
  let mean = LinAlg.solve precision xty in
  
  (* Sample from N(mean, sigma² * precision⁻¹) *)
  let eps = randn_like mean in
  let scale = sqrt sigma in
  let chol = cholesky precision in
  add mean (scalar_mul scale (LinAlg.solve chol eps))

let sample_posterior_distributed data config sigma =
  let p = data.p in
  
  let identity = eye p in
  let precision = add data.xtx (scalar_mul config.an identity) in
  let mean = LinAlg.solve precision data.xty in
  
  let eps = randn_like mean in
  let scale = sqrt sigma in
  let chol = cholesky precision in
  add mean (scalar_mul scale (LinAlg.solve chol eps))
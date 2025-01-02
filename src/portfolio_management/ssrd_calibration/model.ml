open Torch
open Types
open Logging
open Performance

let psi alpha t1 t2 =
  (exp (alpha *. t2) -. exp (alpha *. t1)) /. alpha

let theta alpha beta t1 t2 =
  let p1 = psi beta t1 t2 in
  let p2 = psi (alpha +. beta) t1 t2 in
  (p1 -. p2) /. alpha

let v0 params t x y T =
  exp (-. x *. psi (-.params.alpha1) t T -. params.alpha1 *. params.beta1 *. theta (-.params.alpha1) params.alpha1 t T) *.
  exp (-. y *. psi (-.params.alpha2) t T -. params.alpha2 *. params.beta2 *. theta (-.params.alpha2) params.alpha2 t T)

let h0 params t x y T =
  v0 params t x y T *. (y +. params.alpha2 *. params.beta2 *. psi params.alpha2 t T)

let v1 params t x y T =
  let term1 = params.sigma1 ** 2.0 *. psi (-.params.alpha1) t T *. 
              (x *. psi params.alpha1 t T +. params.alpha1 *. params.beta1 *. theta params.alpha1 params.alpha1 t T) in
  let term2 = params.sigma1 ** 4.0 *. psi (-.params.alpha1) t T ** 3.0 *. 
              (x *. psi params.alpha1 t T +. params.alpha1 *. params.beta1 *. theta params.alpha1 params.alpha1 t T) in
  v0 params t x y T *. (term1 +. term2)

let h1 params t x y T =
  let term1 = params.sigma2 ** 2.0 *. psi (-.params.alpha2) t T *. 
              (y *. psi params.alpha2 t T +. params.alpha2 *. params.beta2 *. theta params.alpha2 params.alpha2 t T) in
  let term2 = params.sigma2 ** 4.0 *. psi (-.params.alpha2) t T ** 3.0 *. 
              (y *. psi params.alpha2 t T +. params.alpha2 *. params.beta2 *. theta params.alpha2 params.alpha2 t T) in
  h0 params t x y T *. (1.0 +. term1 +. term2)

let v2 params t x y T =
  let term1 = params.sigma1 ** 2.0 *. psi (-.params.alpha1) t T *. 
              (x *. psi params.alpha1 t T +. params.alpha1 *. params.beta1 *. theta params.alpha1 params.alpha1 t T) in
  let term2 = params.sigma1 ** 4.0 *. psi (-.params.alpha1) t T ** 3.0 *. 
              (x *. psi params.alpha1 t T +. params.alpha1 *. params.beta1 *. theta params.alpha1 params.alpha1 t T) in
  let term3 = 0.5 *. params.sigma1 ** 4.0 *. psi (-.params.alpha1) t T ** 2.0 *. 
              (x *. psi params.alpha1 t T +. params.alpha1 *. params.beta1 *. theta params.alpha1 params.alpha1 t T) ** 2.0 in
  v0 params t x y T *. (term1 +. term2 +. term3)

let h2 params t x y T =
  let term1 = params.sigma2 ** 2.0 *. psi (-.params.alpha2) t T *. 
              (y *. psi params.alpha2 t T +. params.alpha2 *. params.beta2 *. theta params.alpha2 params.alpha2 t T) in
  let term2 = params.sigma2 ** 4.0 *. psi (-.params.alpha2) t T ** 3.0 *. 
              (y *. psi params.alpha2 t T +. params.alpha2 *. params.beta2 *. theta params.alpha2 params.alpha2 t T) in
  let term3 = 0.5 *. params.sigma2 ** 4.0 *. psi (-.params.alpha2) t T ** 2.0 *. 
              (y *. psi params.alpha2 t T +. params.alpha2 *. params.beta2 *. theta params.alpha2 params.alpha2 t T) ** 2.0 in
  h0 params t x y T *. (1.0 +. term1 +. term2 +. term3)

let v params t x y T order =
  let memoized_v0 = memoize (v0 params t x y)
  and memoized_v1 = memoize (v1 params t x y)
  and memoized_v2 = memoize (v2 params t x y) in
  match order with
  | Zeroth -> memoized_v0 T
  | First -> 
      let result = memoized_v0 T +. memoized_v1 T in
      debug (Printf.sprintf "First order v approximation: %.6f" result);
      result
  | Second -> 
      let result = memoized_v0 T +. memoized_v1 T +. memoized_v2 T in
      debug (Printf.sprintf "Second order v approximation: %.6f" result);
      result

let h params t x y T order =
  let memoized_h0 = memoize (h0 params t x y)
  and memoized_h1 = memoize (h1 params t x y)
  and memoized_h2 = memoize (h2 params t x y) in
  match order with
  | Zeroth -> memoized_h0 T
  | First -> 
      let result = memoized_h0 T +. memoized_h1 T in
      debug (Printf.sprintf "First order h approximation: %.6f" result);
      result
  | Second -> 
      let result = memoized_h0 T +. memoized_h1 T +. memoized_h2 T in
      debug (Printf.sprintf "Second order h approximation: %.6f" result);
      result

let simulate_ssrd params t =
  let open Tensor in
  let dt = 1.0 /. 252.0 in
  let steps = int_of_float (t /. dt) in
  let sqrt_dt = sqrt dt in
  
  let r = zeros [steps + 1] in
  let lambda = zeros [steps + 1] in
  
  r.${[|0|]} <- float params.r0;
  lambda.${[|0|]} <- float params.lambda0;
  
  for i = 1 to steps do
    let prev_r = r.${[|i - 1|]} in
    let prev_lambda = lambda.${[|i - 1|]} in
    
    let dW1 = randn [1] in
    let dW2 = randn [1] in
    
    let dr = params.alpha1 *. (params.beta1 -. prev_r) *. dt +.
             params.sigma1 *. sqrt (prev_r *. dt) *. dW1.${[|0|]} in
    
    let dlambda = params.alpha2 *. (params.beta2 -. prev_lambda) *. dt +.
                  params.sigma2 *. sqrt (prev_lambda *. dt) *.
                  (params.rho *. dW1.${[|0|]} +. sqrt (1.0 -. params.rho *. params.rho) *. dW2.${[|0|]}) in
    
    r.${[|i|]} <- prev_r +. dr;
    lambda.${[|i|]} <- prev_lambda +. dlambda;
  done;
  
  (r, lambda)
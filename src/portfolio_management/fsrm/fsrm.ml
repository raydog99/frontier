open Torch
open Fbm
open Fou
open Utils

type t = {
  fbm: Fbm.t;
  fou: Fou.t;
}

let create ~hurst ~eta ~kappa ~scale =
  {
    fbm = Fbm.create ~hurst ~scale;
    fou = Fou.create ~hurst ~eta ~kappa;
  }

let sample t n =
  let h_t = Fou.sample t.fou n in
  let b_h_t = Fbm.sample t.fbm n in
  Tensor.(mul h_t b_h_t)

let regularity t price =
  let n = Tensor.shape price |> List.hd in
  let log_price = Tensor.log price in
  let increments = Tensor.sub (Tensor.slice log_price ~dim:0 ~start:1 ~end:n ()) 
                              (Tensor.slice log_price ~dim:0 ~start:0 ~end:(n-1) ()) in
  let var = Tensor.var increments ~unbiased:true in
  Tensor.div (Tensor.log var) (Tensor.log (Tensor.scalar 2.))

let binarize x threshold =
  Tensor.ge x (Tensor.scalar threshold) |> Tensor.to_kind Uint8

let serial_information x lag =
  let n = Tensor.shape x |> List.hd in
  let x_binary = binarize x 0.5 in
  let x_lagged = Tensor.slice x_binary ~dim:0 ~start:0 ~end:(n - lag) () in
  let y_lagged = Tensor.slice x_binary ~dim:0 ~start:lag ~end:n () in
  
  let joint_prob = Stats.joint_probability x_lagged y_lagged in
  let marginal_prob_x = Tensor.mean x_lagged ~dim:0 in
  let marginal_prob_y = Tensor.mean y_lagged ~dim:0 in
  
  let joint_entropy = Stats.shannon_entropy joint_prob in
  let marginal_entropy_x = Stats.shannon_entropy marginal_prob_x in
  let marginal_entropy_y = Stats.shannon_entropy marginal_prob_y in
  
  let mutual_information = marginal_entropy_x +. marginal_entropy_y -. joint_entropy in
  mutual_information

let theoretical_serial_information t m =
  let fou = t.fou in
  let rho = Fou.autocorrelation fou m in
  let f x = x *. log x in
  1. +. f (0.5 -. (1. /. Float.pi) *. atan (rho /. sqrt (1. -. rho ** 2.))) +.
     f (0.5 +. (1. /. Float.pi) *. atan (rho /. sqrt (1. -. rho ** 2.)))

let plot_sample t n filename =
  let sample = sample t n in
  let x = Utils.linspace 0. (float_of_int (n - 1)) n in
  Visualization.plot_tensor ~title:"FSRM Sample" ~xlabel:"Time" ~ylabel:"Value" x sample filename

let plot_regularity t price filename =
  let reg = regularity t price in
  let x = Utils.linspace 0. (float_of_int ((Tensor.shape price |> List.hd) - 1)) (Tensor.shape price |> List.hd) in
  Visualization.plot_tensor ~title:"Estimated Regularity" ~xlabel:"Time" ~ylabel:"Hurst Exponent" x reg filename

let local_regularity t price epsilon =
  let n = Tensor.shape price |> List.hd in
  let log_price = Tensor.log price in
  let increments = Tensor.sub (Tensor.slice log_price ~dim:0 ~start:1 ~end:n ()) 
                              (Tensor.slice log_price ~dim:0 ~start:0 ~end:(n-1) ()) in
  let rolling_var = Tensor.unfold increments ~dimension:0 ~size:(int_of_float epsilon) ~step:1 
                    |> Tensor.var ~dim:1 ~unbiased:true in
  Tensor.div (Tensor.log rolling_var) (Tensor.log (Tensor.scalar 2.))

let multifractal_spectrum t price =
  let n = Tensor.shape price |> List.hd in
  let epsilons = Utils.linspace 2. (float_of_int (n / 2)) 20 in
  let h_values = Tensor.stack (List.map (fun e -> local_regularity t price e) (Tensor.to_float1_exn epsilons)) ~dim:0 in
  let f_values = Tensor.div (Tensor.log (Tensor.mean (Tensor.abs h_values) ~dim:1)) (Tensor.log epsilons) in
  (h_values, f_values)

let residuals t price =
  let n = Tensor.shape price |> List.hd in
  let log_price = Tensor.log price in
  let increments = Tensor.sub (Tensor.slice log_price ~dim:0 ~start:1 ~end:n ()) 
                              (Tensor.slice log_price ~dim:0 ~start:0 ~end:(n-1) ()) in
  let h_t = regularity t price in
  Tensor.sub increments (Tensor.mul h_t (Tensor.log (Tensor.scalar t.fbm.scale)))

let standardized_residuals t price =
  let res = residuals t price in
  let h_t = regularity t price in
  Tensor.div res (Tensor.pow (Tensor.scalar t.fbm.scale) h_t)
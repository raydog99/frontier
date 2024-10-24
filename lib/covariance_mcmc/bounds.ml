open Torch
open Types

let bernstein samples spectral_gap variance_bound max_norm t =
  let actual_bound = 
    let d = Tensor.size samples 1 |> float_of_int in
    let alpha = (2.0 -. spectral_gap) /. spectral_gap in
    let beta = 8.0 /. (Float.pi *. spectral_gap) in
    let exp_term = 
      -.(t *. t) /. (32.0 *. alpha *. variance_bound +. beta *. max_norm *. t)
    in
    d ** (2.0 -. Float.pi /. 4.0) *. exp exp_term
  in
  
  let empirical_bound = 
    let n_trials = 1000 in
    let count = ref 0 in
    for _ = 1 to n_trials do
      let test_samples = Tensor.randn_like samples in
      let norm = Tensor.norm test_samples ~p:(Scalar 2) ~dim:[1] |> Tensor.max |> fst |> Tensor.item in
      if norm >= t then incr count
    done;
    float_of_int !count /. float_of_int n_trials
  in
  
  actual_bound >= empirical_bound

let tail_bounds samples c_pi ts =
  let mean = Tensor.mean samples ~dim:[0] ~keepdim:true in
  let centered = Tensor.sub samples mean in
  let norms = Tensor.norm centered ~p:(Scalar 2) ~dim:[1] ~keepdim:false in
  let tr_sigma = Tensor.mm (Tensor.transpose centered 0 1) centered
                |> Tensor.trace 
                |> Tensor.item in
  
  List.map (fun t ->
    let theoretical_bound = 
      exp (-.(t -. 1.) *. sqrt (tr_sigma /. c_pi)) in
    let empirical_bound =
      let threshold = t *. sqrt tr_sigma in
      let exceed_count = Tensor.sum (Tensor.gt norms 
        (Tensor.full_like norms threshold))
        |> Tensor.item
        |> int_of_float in
      float_of_int exceed_count /. float_of_int (Tensor.size samples 0) in
    empirical_bound <= theoretical_bound
  ) ts

let covariance_concentration samples epsilon delta c_pi =
  let n = Tensor.size samples 0 in
  let d = Tensor.size samples 1 in
  
  let true_cov = Tensor.mm (Tensor.transpose samples 0 1) samples in
  let est_cov = Tensor.mm (Tensor.transpose samples 0 1) samples
                |> Tensor.div_scalar (float_of_int n) in
      
  let diff = Tensor.sub est_cov true_cov in
  let id_term = Tensor.mul_scalar (Tensor.eye d) delta in
  let upper_bound = Tensor.add 
    (Tensor.mul_scalar true_cov epsilon)
    id_term in
  
  let diff_norm = Tensor.norm diff ~p:(Scalar 2) |> Tensor.item in
  let bound_norm = Tensor.norm upper_bound ~p:(Scalar 2) |> Tensor.item in
  
  diff_norm <= bound_norm

let all_bounds samples chain c_pi epsilon delta device_config =
  let failures = ref [] in
  
  (* Check Matrix Bernstein *)
  let bernstein_ok = bernstein samples 0.99 1.0 1.0 epsilon in
  if not bernstein_ok then
    failures := "Matrix Bernstein inequality" :: !failures;
  
  (* Check tail bounds *)
  let tail_ok = tail_bounds samples c_pi [1.0; 2.0; 3.0] in
  if not (List.for_all (fun x -> x) tail_ok) then
    failures := "Tail bounds" :: !failures;
  
  (* Check covariance concentration *)
  let concentration_ok = covariance_concentration
    samples epsilon delta c_pi in
  if not concentration_ok then
    failures := "Covariance concentration" :: !failures;
  
  List.length !failures = 0, !failures
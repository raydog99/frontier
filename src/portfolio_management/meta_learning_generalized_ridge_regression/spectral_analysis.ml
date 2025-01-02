open Torch
open Types
open Matrix_ops

let compute_esd matrix =
  let eigenvals = Tensor.symeig matrix ~eigenvectors:false in
  let sorted_vals = Tensor.sort eigenvals ~descending:false |> fst in
  let n = Tensor.size sorted_vals |> List.hd in
  
  (* Compute support *)
  let min_eval = Tensor.get sorted_vals [|0|] |> Tensor.to_float0_exn in
  let max_eval = Tensor.get sorted_vals [|n-1|] |> Tensor.to_float0_exn in
  
  (* Compute density using kernel density estimation *)
  let density x =
    let bandwidth = (max_eval -. min_eval) /. (sqrt (float_of_int n)) in
    let kernel u = exp (-. u *. u /. (2. *. bandwidth *. bandwidth)) 
                  /. (sqrt (2. *. Float.pi) *. bandwidth) in
    let sum_ref = ref 0. in
    for i = 0 to n - 1 do
      let eval = Tensor.get sorted_vals [|i|] |> Tensor.to_float0_exn in
      sum_ref := !sum_ref +. kernel ((x -. eval) /. bandwidth)
    done;
    !sum_ref /. float_of_int n
  in
  
  (* Compute Stieltjes transform *)
  let stieltjes z =
    let real = Complex.re z and imag = Complex.im z in
    let sum_ref = ref Complex.zero in
    for i = 0 to n - 1 do
      let eval = Tensor.get sorted_vals [|i|] |> Tensor.to_float0_exn in
      let denom = Complex.div 
        Complex.one 
        (Complex.add (Complex.make (eval -. real) (-.imag)) z) in
      sum_ref := Complex.add !sum_ref denom
    done;
    Complex.div !sum_ref (Complex.make (float_of_int n) 0.)
  in
  
  {support = (min_eval, max_eval); density; stieltjes}

let compute_lsd ~matrix ~gamma =
  let esd = compute_esd matrix in
  let (a, b) = esd.support in
  
  (* Compute limiting density using Marchenko-Pastur equation *)
  let density x =
    if x < a || x > b then 0.
    else
      (* Numerical integration for MP equation *)
      let dx = (b -. a) /. 1000. in
      let sum_ref = ref 0. in
      for i = 0 to 999 do
        let t = a +. dx *. float_of_int i in
        let integrand = esd.density t *. t /. 
          ((t -. x) ** 2. +. (gamma *. t *. esd.density t) ** 2.) in
        sum_ref := !sum_ref +. integrand *. dx
      done;
      !sum_ref /. (Float.pi *. gamma)
  in
  
  (* Compute limiting Stieltjes transform *)
  let stieltjes z =
    let real = Complex.re z and imag = Complex.im z in
    (* Solve fixed point equation numerically *)
    let rec iterate s iter =
      if iter > 100 then s
      else
        let new_s = ref Complex.zero in
        let dx = (b -. a) /. 1000. in
        for i = 0 to 999 do
          let t = a +. dx *. float_of_int i in
          let denom = Complex.add 
            (Complex.make (t *. (1. -. gamma) -. real) (-.imag))
            (Complex.mul (Complex.make (gamma *. t) 0.) s) in
          let term = Complex.mul 
            (Complex.make (esd.density t *. dx) 0.)
            (Complex.div Complex.one denom) in
          new_s := Complex.add !new_s term
        done;
        if Complex.norm (Complex.sub s !new_s) < 1e-10 
        then !new_s
        else iterate !new_s (iter + 1)
    in
    iterate Complex.zero 0
  in
  
  {support = (a, b); density; stieltjes}

let verify_convergence ~matrix ~gamma ~num_points =
  let lsd = compute_lsd ~matrix ~gamma in
  let esd = compute_esd matrix in
  
  (* Generate evaluation points *)
  let eval_points = List.init num_points (fun i ->
    let t = float_of_int i /. float_of_int (num_points - 1) in
    let (a, b) = esd.support in
    a +. t *. (b -. a)
  ) in
  
  (* Compute convergence at each point *)
  List.map (fun x ->
    let empirical = esd.density x in
    let limiting = lsd.density x in
    abs_float (empirical -. limiting)
  ) eval_points
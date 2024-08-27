open Torch
open Lwt

exception Invalid_parameter of string
exception Calibration_error of string

module Log = struct
  let info msg = Printf.printf "[INFO] %s\n" msg
  let warn msg = Printf.eprintf "[WARN] %s\n" msg
  let error msg = Printf.eprintf "[ERROR] %s\n" msg
end

module Utils = struct
  let normal_cdf x =
    0.5 *. (1. +. Stdlib.erf (x /. sqrt 2.))

  let normal_pdf x =
    exp (-0.5 *. x *. x) /. sqrt (2. *. Float.pi)

  let sample_gamma shape scale =
    let u = Tensor.uniform_real ~from:0. ~to_:1. [1] |> Tensor.get [0] in
    -.(log u) *. scale /. shape

  let sample_poisson lambda =
    let l = exp (-.lambda) in
    let rec loop k p =
      let u = Tensor.uniform_real ~from:0. ~to_:1. [1] |> Tensor.get [0] in
      let p = p *. u in
      if p > l then loop (k + 1) p else k
    in
    loop 0 1.0

  let black_scholes_price s k r t sigma =
    let d1 = (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
    let d2 = d1 -. sigma *. sqrt t in
    s *. normal_cdf d1 -. k *. exp (-.r *. t) *. normal_cdf d2

  let black_scholes_implied_vol s k t r price option_type =
    let rec newton_raphson sigma iter =
      if iter > 100 then raise (Invalid_parameter "Implied volatility calculation did not converge")
      else
        let bs_price = black_scholes_price s k r t sigma in
        let vega = s *. sqrt t *. normal_pdf ((log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t)) in
        let diff = 
          match option_type with
          | `Call -> bs_price -. price
          | `Put -> bs_price -. price +. k *. exp (-.r *. t) -. s
        in
        if abs_float diff < 1e-8 then sigma
        else
          let new_sigma = sigma -. diff /. vega in
          newton_raphson new_sigma (iter + 1)
    in
    newton_raphson 0.5 0

  let parallel_map f xs =
    let num_domains = Domain.recommended_domain_count () in
    let chunk_size = Array.length xs / num_domains in
    
    let worker i =
      let start = i * chunk_size in
      let end_ = if i = num_domains - 1 then Array.length xs else (i + 1) * chunk_size in
      Array.init (end_ - start) (fun j -> f xs.(start + j))
    in
    
    Array.init num_domains worker
    |> Array.to_list
    |> Lwt_list.map_p Lwt.return
    |> Lwt_main.run
    |> Array.concat

  let cubic_spline_interpolation xs ys =
    let n = Array.length xs - 1 in
    let h = Array.init n (fun i -> xs.(i+1) -. xs.(i)) in
    let alpha = Array.init (n-1) (fun i ->
      3. /. h.(i+1) *. (ys.(i+2) -. ys.(i+1)) -. 3. /. h.(i) *. (ys.(i+1) -. ys.(i))
    ) in
    let l = Array.make (n+1) 0. in
    let mu = Array.make (n+1) 0. in
    let z = Array.make (n+1) 0. in
    l.(0) <- 1.;
    for i = 1 to n-1 do
      l.(i) <- 2. *. (xs.(i+1) -. xs.(i-1)) -. h.(i-1) *. mu.(i-1);
      mu.(i) <- h.(i) /. l.(i);
      z.(i) <- (alpha.(i-1) -. h.(i-1) *. z.(i-1)) /. l.(i);
    done;
    l.(n) <- 1.;
    let c = Array.make (n+1) 0. in
    for j = n-1 downto 0 do
      c.(j) <- z.(j) -. mu.(j) *. c.(j+1);
    done;
    let b = Array.init n (fun i ->
      (ys.(i+1) -. ys.(i)) /. h.(i) -. h.(i) *. (c.(i+1) +. 2. *. c.(i)) /. 3.
    ) in
    let d = Array.init n (fun i -> (c.(i+1) -. c.(i)) /. (3. *. h.(i))) in
    fun x ->
      let i = ref 0 in
      while !i < n && x > xs.(!i+1) do incr i done;
      if !i = n then ys.(n)
      else
        let dx = x -. xs.(!i) in
        ys.(!i) +. b.(!i) *. dx +. c.(!i) *. dx ** 2. +. d.(!i) *. dx ** 3.
end

module NumericalIntegration = struct
  let gauss_legendre f a b n =
    let xi, wi = match n with
      | 2 -> [|-0.5773502692; 0.5773502692|], [|1.; 1.|]
      | 3 -> [|-0.7745966692; 0.; 0.7745966692|], [|0.5555555556; 0.8888888889; 0.5555555556|]
      | 4 -> [|-0.8611363116; -0.3399810436; 0.3399810436; 0.8611363116|],
             [|0.3478548451; 0.6521451549; 0.6521451549; 0.3478548451|]
      | 5 -> [|-0.9061798459; -0.5384693101; 0.; 0.5384693101; 0.9061798459|],
             [|0.2369268850; 0.4786286705; 0.5688888889; 0.4786286705; 0.2369268850|]
      | _ -> failwith "Unsupported number of points for Gauss-Legendre quadrature"
    in
    let sum = ref 0. in
    for i = 0 to n - 1 do
      let x = ((b -. a) *. xi.(i) +. (b +. a)) /. 2. in
      sum := !sum +. wi.(i) *. f x
    done;
    (b -. a) /. 2. *. !sum

  let rec adaptive_simpson f a b eps =
    let c = (a +. b) /. 2. in
    let h = b -. a in
    let fa = f a and fb = f b and fc = f c in
    let s = h /. 6. *. (fa +. 4. *. fc +. fb) in
    let rec adaptive s1 s2 a c b fa fc fb =
      let d = (a +. c) /. 2. and e = (c +. b) /. 2. in
      let fd = f d and fe = f e in
      let s1 = (c -. a) /. 6. *. (fa +. 4. *. fd +. fc) in
      let s2 = (b -. c) /. 6. *. (fc +. 4. *. fe +. fb) in
      let s' = s1 +. s2 in
      if abs_float (s' -. s) <= 15. *. eps
      then s' +. (s' -. s) /. 15.
      else adaptive s1 s2 a d c fa fd fc +. adaptive s1 s2 c e b fc fe fb
    in
    adaptive s s a c b fa fc fb
end

module CEV = struct
  let sample f0 sigma0 beta t =
    let alpha = 1. /. (2. *. (1. -. beta)) in
    let z0 = f0 ** (2. *. (1. -. beta)) /. ((1. -. beta) ** 2. *. sigma0 ** 2. *. t) in
    
    let x = Utils.sample_gamma alpha 1. in
    if x <= z0 /. 2. then 0.
    else
      let n = Utils.sample_poisson (z0 /. 2. -. x) in
      let z = 2. *. Utils.sample_gamma (float_of_int n +. 1.) 1. in
      ((1. -. beta) ** 2. *. sigma0 ** 2. *. t *. z) ** (1. /. (2. *. (1. -. beta)))
end

module Sabr = struct
  type model = Standard | Shifted of float | Dynamic of (float -> float)
  type t = {
    f0: float;
    alpha0: float;
    nu: float;
    rho: float;
    beta: float;
    model: model;
  }

  let create f0 alpha0 nu rho beta model =
    if f0 <= 0. then raise (Invalid_parameter "f0 must be positive")
    else if alpha0 <= 0. then raise (Invalid_parameter "alpha0 must be positive")
    else if nu < 0. then raise (Invalid_parameter "nu must be non-negative")
    else if rho <= -1. || rho >= 1. then raise (Invalid_parameter "rho must be between -1 and 1")
    else if beta < 0. || beta > 1. then raise (Invalid_parameter "beta must be between 0 and 1")
    else { f0; alpha0; nu; rho; beta; model }

  let sample_average_variance sabr t alpha_t alpha_t_plus_h =
    let nu_hat = sabr.nu *. sqrt t in
    let z_hat = (1. /. nu_hat) *. log (alpha_t_plus_h /. alpha_t) in
    
    let c = 0.5 *. (alpha_t_plus_h /. alpha_t +. alpha_t /. alpha_t_plus_h) in
    
    let m1 = Utils.normal_cdf (z_hat +. nu_hat) -. Utils.normal_cdf (z_hat -. nu_hat) in
    let m2 = Utils.normal_cdf (z_hat +. 2. *. nu_hat) -. Utils.normal_cdf (z_hat -. 2. *. nu_hat) in
    
    let mu = (alpha_t_plus_h /. alpha_t) *. m1 /. (2. *. nu_hat) in
    let mu_2 = (alpha_t_plus_h /. alpha_t) ** 2. *. (m2 /. (4. *. nu_hat ** 2.) -. c *. m1 /. (2. *. nu_hat ** 2.)) in
    
    let v = sqrt (mu_2 /. (mu ** 2.) -. 1.) in
    
    let sigma = sqrt (log (1. +. (36. /. 25.) *. v ** 2.)) in
    
    let u = Tensor.uniform_real ~from:0. ~to_:1. [1] |> Tensor.get [0] in
    
    mu /. 6. *. (1. +. 5. *. exp (sigma *. (Tensor.randn [1] |> Tensor.get [0]) -. 0.5 *. sigma ** 2.))

  let sample_conditional_forward_price sabr t f_t alpha_t i_h_t =
    let beta_star = 1. -. sabr.beta in
    let rho_star = sqrt (1. -. sabr.rho ** 2.) in
    
    let f_bar = f_t *. exp (
      sabr.rho *. (alpha_t -. sabr.alpha0) /. (sabr.nu *. f_t ** beta_star) -.
      0.5 *. sabr.rho ** 2. *. alpha_t ** 2. *. t *. i_h_t /. (f_t ** (2. *. beta_star))
    ) in
    
    CEV.sample f_bar (rho_star *. alpha_t *. sqrt i_h_t) sabr.beta t

  let simulate sabr t steps =
    let dt = t /. float_of_int steps in
    let sqrt_dt = sqrt dt in
    
    let rec sim_loop i f alpha acc_f acc_alpha =
      if i = steps then
        (Tensor.of_float_list (List.rev acc_f) [steps + 1],
         Tensor.of_float_list (List.rev acc_alpha) [steps + 1])
      else
        let dw = Tensor.randn [1] ~dtype:Float in
        let new_alpha = alpha *. exp (sabr.nu *. sqrt_dt *. Tensor.get dw [0] -. 0.5 *. sabr.nu *. sabr.nu *. dt) in
        let i_h_t = sample_average_variance sabr dt alpha new_alpha in
        let new_f = sample_conditional_forward_price sabr dt f alpha i_h_t in
        sim_loop (i + 1) new_f new_alpha (new_f :: acc_f) (new_alpha :: acc_alpha)
    in

    sim_loop 0 sabr.f0 sabr.alpha0 [sabr.f0] [sabr.alpha0]

  let simulate_terminal sabr t n_paths =
    let sim_path _ =
      let rec sim_loop f alpha remaining_t =
        if remaining_t <= 0. then f
        else
          let dt = min remaining_t 0.25 in
          let new_alpha = alpha *. exp (sabr.nu *. sqrt dt *. Tensor.get (Tensor.randn [1]) [0] -. 0.5 *. sabr.nu *. sabr.nu *. dt) in
          let i_h_t = sample_average_variance sabr dt alpha new_alpha in
          let new_f = sample_conditional_forward_price sabr dt f alpha i_h_t in
          sim_loop new_f new_alpha (remaining_t -. dt)
      in
      sim_loop sabr.f0 sabr.alpha0 t
    in
    Tensor.init [n_paths] ~f:(fun _ -> sim_path ())

  let price_european_options sabr t strikes n_paths option_type =
    let terminal_prices = simulate_terminal sabr t n_paths in
    Utils.parallel_map (fun k ->
      let payoffs = 
        match option_type with
        | `Call -> Tensor.max (Tensor.sub terminal_prices (Tensor.full [n_paths] k)) (Tensor.zeros [n_paths])
        | `Put -> Tensor.max (Tensor.sub (Tensor.full [n_paths] k) terminal_prices) (Tensor.zeros [n_paths])
      in
      let price = Tensor.mean payoffs |> Tensor.to_float0_exn in
      let std_error = Tensor.std payoffs ~dim:[0] ~unbiased:true |> Tensor.to_float0_exn /. sqrt (float_of_int n_paths) in
      (price, std_error)
    ) strikes

  let implied_volatilities sabr t strikes prices option_type =
    Utils.parallel_map (fun (k, p) ->
      let implied_vol = Utils.black_scholes_implied_vol sabr.f0 k t 0. p option_type in
      let vega = Utils.black_scholes_price sabr.f0 k 0. t (implied_vol +. 0.01) -. 
                 Utils.black_scholes_price sabr.f0 k 0. t (implied_vol -. 0.01) in
      let std_error = p *. 0.01 /. vega in  (* Approximate standard error *)
      (implied_vol, std_error)
    ) (Array.combine strikes prices)

  let hagan_formula sabr k t =
    let f = sabr.f0 in
    let alpha = sabr.alpha0 in
    let beta = sabr.beta in
    let rho = sabr.rho in
    let nu = sabr.nu in
    
    let log_fk = log (f /. k) in
    let fk_mid = (f *. k) ** ((1. -. beta) /. 2.) in
    let gamma1 = beta /. fk_mid in
    let gamma2 = -. beta *. log_fk /. fk_mid in
    let x = log ((sqrt (1. -. 2. *. rho *. gamma1 +. gamma1 *. gamma1) +. gamma1 -. rho) /. (1. -. rho)) in
    
    let epsilon = 1e-7 in
    let z = 
      if abs_float log_fk > epsilon then
        (nu /. alpha) *. fk_mid *. log_fk /. x
      else
        nu *. fk_mid /. alpha
    in
    
    let a = 1. +. ((2. -. 3. *. rho *. rho) /. 24.) *. z *. z in
    let b = 1. +. (1. /. 24.) *. (log_fk *. log_fk /. (fk_mid *. fk_mid)) +. (1. /. 1920.) *. (log_fk *. log_fk *. log_fk *. log_fk /. (fk_mid *. fk_mid *. fk_mid *. fk_mid)) in
    
    (alpha /. fk_mid) *. (1. +. (((1. -. beta) ** 2. /. 24.) *. alpha *. alpha /. (fk_mid *. fk_mid) +. 0.25 *. rho *. beta *. nu *. alpha /. fk_mid +. ((2. -. 3. *. rho *. rho) /. 24.) *. nu *. nu) *. t) *. z *. a /. (x *. b)

  let calibrate f0 strikes market_prices t option_type =
    let objective params =
      let alpha0, nu, rho, beta = params.(0), params.(1), params.(2), params.(3) in
      let sabr = create f0 alpha0 nu rho beta Standard in
      let model_prices = price_european_options sabr t strikes 10000 option_type in
      Array.fold_left2 (fun acc (mp, _) (pp, _) -> acc +. (mp -. pp) ** 2.) 0. market_prices model_prices
    in
    
    let initial_guess = [|0.3; 0.5; 0.0; 0.5|] in
    let lower_bounds = [|0.001; 0.001; -0.999; 0.001|] in
    let upper_bounds = [|2.0; 2.0; 0.999; 0.999|] in
    
    let result = Optimizers.minimize ~f:objective ~initial:initial_guess ~lower_bounds ~upper_bounds () in
    let alpha0, nu, rho, beta = result.(0), result.(1), result.(2), result.(3) in
    create f0 alpha0 nu rho beta Standard

  let local_volatility sabr t k =
    let f = match sabr.model with
      | Standard -> sabr.f0
      | Shifted shift -> sabr.f0 +. shift
      | Dynamic f -> f t
    in
    let alpha = sabr.alpha0 in
    let beta = sabr.beta in
    let rho = sabr.rho in
    let nu = sabr.nu in
    
    let fk_mid = (f *. k) ** ((1. -. beta) /. 2.) in
    let gamma1 = beta /. fk_mid in
    let gamma2 = -. beta *. (log (f /. k)) /. fk_mid in
    
    let a = (1. -. beta) ** 2. *. t /. (24. *. fk_mid ** 2.) in
    let b = 0.25 *. rho *. beta *. nu *. t /. fk_mid in
    let c = (2. -. 3. *. rho ** 2.) *. nu ** 2. *. t /. 24. in
    
    alpha *. (1. +. (a +. b +. c)) *. k ** (beta -. 1.)

  let forward_volatility sabr t1 t2 k =
    let v1 = hagan_formula sabr k t1 in
    let v2 = hagan_formula sabr k t2 in
    sqrt ((v2 ** 2. *. t2 -. v1 ** 2. *. t1) /. (t2 -. t1))

  let delta sabr t k s option_type =
    let h = 0.01 *. s in
    let price1 = price_european_options sabr t [|s -. h|] 100000 option_type |> Array.get 0 |> fst in
    let price2 = price_european_options sabr t [|s +. h|] 100000 option_type |> Array.get 0 |> fst in
    (price2 -. price1) /. (2. *. h)

  let vega sabr t k s option_type =
    let h = 0.01 *. sabr.alpha0 in
    let sabr1 = {sabr with alpha0 = sabr.alpha0 -. h} in
    let sabr2 = {sabr with alpha0 = sabr.alpha0 +. h} in
    let price1 = price_european_options sabr1 t [|k|] 100000 option_type |> Array.get 0 |> fst in
    let price2 = price_european_options sabr2 t [|k|] 100000 option_type |> Array.get 0 |> fst in
    (price2 -. price1) /. (2. *. h)

  let rho_sensitivity sabr t k s option_type =
    let h = 0.01 in
    let sabr1 = {sabr with rho = sabr.rho -. h} in
    let sabr2 = {sabr with rho = sabr.rho +. h} in
    let price1 = price_european_options sabr1 t [|k|] 100000 option_type |> Array.get 0 |> fst in
    let price2 = price_european_options sabr2 t [|k|] 100000 option_type |> Array.get 0 |> fst in
    (price2 -. price1) /. (2. *. h)
end

module Test = struct
  let test_black_scholes () =
    let s = 100. and k = 100. and r = 0.05 and t = 1. and sigma = 0.2 in
    let price = Utils.black_scholes_price s k r t sigma in
    let expected = 10.45 in
    assert (abs_float (price -. expected) < 0.01)

  let test_sabr_simulation () =
    let sabr = Sabr.create 100. 0.2 0.3 0.5 0.7 Sabr.Standard in
    let _, volatility = Sabr.simulate sabr 1. 252 in
    let final_vol = Tensor.get volatility [251] in
    assert (final_vol > 0.)

  let test_option_pricing () =
    let sabr = Sabr.create 100. 0.2 0.3 0.5 0.7 Sabr.Standard in
    let prices = Sabr.price_european_options sabr 1. [|100.|] 10000 `Call in
    assert (Array.length prices = 1)

  let test_calibration () =
    let market_prices = [|10.; 11.; 12.|] in
    let strikes = [|90.; 100.; 110.|] in
    let sabr = Sabr.calibrate 100. strikes market_prices 1. `Call in
    assert (sabr.f0 = 100.)

  let run_all_tests () =
    test_black_scholes ();
    test_sabr_simulation ();
    test_option_pricing ();
    test_calibration ();
    Log.info "All tests passed successfully!"
end
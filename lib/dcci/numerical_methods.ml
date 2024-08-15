open Types
open Torch

let normal_cdf x =
  let half = Scalar.float 0.5 in
  let one = Scalar.float 1.0 in
  Tensor.(half * (one + erf (x / sqrt (Scalar.float 2.0))))

let black_scholes_call s k r t sigma =
  let d1 = Tensor.((log (s / k) + (r + sigma * sigma / Scalar.float 2.0) * t) / (sigma * sqrt t)) in
  let d2 = Tensor.(d1 - sigma * sqrt t) in
  Tensor.(s * normal_cdf d1 - k * exp (Scalar.neg r * t) * normal_cdf d2)

let rec fft_inplace a =
  let n = Array.length a in
  if n <= 1 then ()
  else begin
    let m = n / 2 in
    let b = Array.sub a 0 m in
    let c = Array.sub a m m in
    fft_inplace b;
    fft_inplace c;
    for k = 0 to m - 1 do
      let t = Complex.polar 1.0 (-2.0 *. Float.pi *. float k /. float n) in
      let t = Complex.mul t c.(k) in
      a.(k) <- Complex.add b.(k) t;
      a.(k+m) <- Complex.sub b.(k) t
    done
  end

let gauss_laguerre_quadrature f n =
  let roots, weights = Quadrature.gauss_laguerre n in
  Array.fold_left2 (fun acc x w -> acc +. w *. f x) 0. roots weights

let heston_characteristic_function_stable params s k tau =
  let { kappa; theta; chi; rho; v0; _ } = params in
  fun u ->
    let a = kappa *. theta in
    let b = kappa -. rho *. chi *. u.Complex.re in
    let d = Complex.sqrt (Complex.sub (Complex.mul (Complex.of_float b) (Complex.of_float b)) 
                          (Complex.mul (Complex.of_float (chi *. chi)) 
                                       (Complex.mul u (Complex.sub u (Complex.of_float 1.)))))
    in
    let g1 = Complex.add (Complex.sub (Complex.of_float b) d) (Complex.of_float chi) in
    let g2 = Complex.sub (Complex.sub (Complex.of_float b) d) (Complex.of_float chi) in
    
    let term1 = Complex.exp (Complex.mul (Complex.of_float (Complex.i *. u.Complex.re *. log (s /. k))) 
                                         (Complex.of_float tau))
    in
    let term2 = Complex.exp (Complex.div (Complex.mul (Complex.of_float a) 
                                                      (Complex.mul (Complex.of_float tau) g2))
                                         (Complex.mul (Complex.of_float chi) (Complex.of_float chi)))
    in
    let term3 = Complex.exp (Complex.div (Complex.of_float (v0 *. Complex.i *. u.Complex.re))
                                         (Complex.mul (Complex.of_float chi) g1))
    in
    
    Complex.mul term1 (Complex.mul term2 term3)

let heston_option_price_gl params s k tau option_type =
  let char_func = heston_characteristic_function_stable params s k tau in
  let integrand u =
    let cf = char_func (Complex.of_float (-. u)) in
    (Complex.re cf *. cos (u *. log (s /. k)) +. Complex.im cf *. sin (u *. log (s /. k))) /. u
  in
  let price = gauss_laguerre_quadrature integrand 64 in
  let price = s -. k *. exp (-. params.kappa *. tau) *. (0.5 +. price /. Float.pi) in
  match option_type with
  | Call -> price
  | Put -> price +. k *. exp (-. params.kappa *. tau) -. s

let sabr_volatility alpha beta rho nu f k t =
  let z = (nu /. alpha) *. (f *. k) ** ((1. -. beta) /. 2.) *. log (f /. k) in
  let x = log ((sqrt (1. -. 2. *. rho *. z +. z *. z) +. z -. rho) /. (1. -. rho)) in
  alpha *. (f *. k) ** ((beta -. 1.) /. 2.) *. (z /. x) *. 
  (1. +. ((2. -. 3. *. rho *. rho) *. nu *. nu *. t) /. 24.)

let calibrate_local_volatility_surface_improved forward_prices option_prices strikes maturities =
  let n_strikes = Tensor.size strikes 0 in
  let n_maturities = Tensor.size maturities 0 in
  let local_vol = Tensor.zeros [n_maturities; n_strikes] in

  for i = 0 to n_maturities - 1 do
    for j = 0 to n_strikes - 1 do
      let t = Tensor.get maturities [|i|] in
      let k = Tensor.get strikes [|j|] in
      let f = Tensor.get forward_prices [|i|] in
      let implied_vol = black_scholes_implied_volatility (Tensor.get option_prices [|i; j|]) f k t in
      
      let reg_factor = 0.01 in
      let reg_vol = implied_vol +. reg_factor *. (k -. f) ** 2. /. (f *. f *. t) in
      
      Tensor.set local_vol [|i; j|] reg_vol
    done
  done;

  let smoothed_vol = Tensor.zeros_like local_vol in
  for i = 1 to n_maturities - 2 do
    for j = 1 to n_strikes - 2 do
      let vol = Tensor.get local_vol [|i; j|] in
      let vol_up = Tensor.get local_vol [|i-1; j|] in
      let vol_down = Tensor.get local_vol [|i+1; j|] in
      let vol_left = Tensor.get local_vol [|i; j-1|] in
      let vol_right = Tensor.get local_vol [|i; j+1|] in
      let smoothed = (vol +. vol_up +. vol_down +. vol_left +. vol_right) /. 5. in
      Tensor.set smoothed_vol [|i; j|] smoothed
    done
  done;

  smoothed_vol

let crank_nicolson_pde_adaptive grid dx dt r sigma f =
  let nx = Tensor.shape grid.x |> List.hd in
  let nt = Tensor.shape grid.t |> List.hd in
  
  let a = Tensor.full [nx] (-0.5 * dt * sigma ** 2. / dx ** 2. + 0.25 * dt * r / dx) in
  let b = Tensor.full [nx] (1. + dt * sigma ** 2. / dx ** 2. + 0.5 * dt * r) in
  let c = Tensor.full [nx] (-0.5 * dt * sigma ** 2. / dx ** 2. - 0.25 * dt * r / dx) in
  
  for j = nt - 2 downto 0 do
    let rhs = Tensor.get grid.v [|j+1|] in
    let new_v = tridiag_solver a b c rhs in
    
    let error = Tensor.abs (Tensor.sub new_v (Tensor.get grid.v [|j|])) in
    let max_error = Tensor.max error |> Tensor.to_float0_exn in
    if max_error > 1e-3 then begin
      let half_dt = dt /. 2. in
      let half_a = Tensor.full [nx] (-0.5 * half_dt * sigma ** 2. / dx ** 2. + 0.25 * half_dt * r / dx) in
      let half_b = Tensor.full [nx] (1. + half_dt * sigma ** 2. / dx ** 2. + 0.5 * half_dt * r) in
      let half_c = Tensor.full [nx] (-0.5 * half_dt * sigma ** 2. / dx ** 2. - 0.25 * half_dt * r / dx) in
      
      let mid_v = tridiag_solver half_a half_b half_c rhs in
      let final_v = tridiag_solver half_a half_b half_c mid_v in
      Tensor.set grid.v [|j|] final_v
    end else begin
      Tensor.set grid.v [|j|] new_v
    end
  done;
  
  grid

let fast_fourier_transform_option_pricing char_func s k tau =
  let n = 4096 in
  let alpha = 1.5 in
  let eta = 2.0 *. Float.pi /. float n in
  let lambda = 2.0 *. Float.pi /. (float n *. eta) in
  let b = Float.log (s /. k) +. lambda in

  let x = Array.init n (fun j -> b +. float j *. lambda /. float n) in
  let u = Array.init n (fun j -> float j *. eta) in

  let fft_input = Array.init n (fun j ->
    let uj = u.(j) in
    let exp_term = Complex.exp (Complex.mul Complex.i (Complex.of_float (b *. uj))) in
    let char_term = char_func (Complex.neg (Complex.of_float (uj -. Complex.i *. alpha))) in
    Complex.(mul (div exp_term (mul (of_float uj) (of_float (uj +. Complex.i *. alpha)))) char_term)
  ) in

  fft_inplace fft_input;

  let option_price = ref 0.0 in
  for j = 0 to n - 1 do
    let re = Complex.re fft_input.(j) in
    let im = Complex.im fft_input.(j) in
    option_price := !option_price +. re *. cos (eta *. float j *. x.(0)) -. im *. sin (eta *. float j *. x.(0))
  done;

  s *. exp (-. alpha *. x.(0)) *. !option_price /. Float.pi
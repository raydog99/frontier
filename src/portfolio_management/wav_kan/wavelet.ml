open Torch

module Base = struct
  type property = {
    support_width: float;
    vanishing_moments: int;
    regularity: float;
    symmetry: [`Symmetric | `Antisymmetric | `None];
  }

  type family = 
    | Ricker of property
    | Morlet of property * float
    | Paul of property * int
    | DOG of property * int
    | Shannon of property
    | Meyer of property
    | ComplexGaussian of property * int

  let ricker_props = {
    support_width = 5.0;
    vanishing_moments = 2;
    regularity = 2.0;
    symmetry = `Symmetric;
  }

  let morlet_props = {
    support_width = 6.0;
    vanishing_moments = 2;
    regularity = 1.5;
    symmetry = `Symmetric;
  }

  let get_family = function
    | "ricker" -> Ricker ricker_props
    | "morlet" -> Morlet (morlet_props, 5.0)
    | "paul" -> Paul (ricker_props, 2)
    | "dog" -> DOG (ricker_props, 2)
    | "shannon" -> Shannon ricker_props
    | "meyer" -> Meyer ricker_props
    | "cgauss" -> ComplexGaussian (ricker_props, 2)
    | _ -> failwith "Unknown wavelet family"
end

module Function = struct
  open Torch

  let ricker t =
    let t_sq = t *. t in
    (2.0 /. (sqrt 3.0 *. Float.pi ** 0.25)) *. 
    (1.0 -. t_sq) *. exp (-. t_sq /. 2.0)

  let morlet t omega0 =
    let t_sq = t *. t in
    exp (-. t_sq /. 2.0) *. cos (omega0 *. t)

  let paul t m =
    let z = Complex.{ re = t; im = 0.0 } in
    let i = Complex.{ re = 0.0; im = 1.0 } in
    let rec factorial n = 
      if n <= 1 then 1 else n * factorial (n - 1) in
    let norm = sqrt (2.0 ** float m *. float (factorial m) /. 
                    Float.pi *. float (factorial (2 * m))) in
    let denom = Complex.(exp { re = 0.0; 
                              im = -. float (m + 1) *. atan2 t 1.0 }) in
    Complex.(norm *. Re (mul (of_float 1.0) denom))

  let dog t n =
    let t_sq = t *. t in
    let exp_term = exp (-. t_sq /. 2.0) in
    let rec hermite n x =
      match n with
      | 0 -> 1.0
      | 1 -> 2.0 *. x
      | _ -> 2.0 *. x *. hermite (n-1) x -. 
             2.0 *. float (n-1) *. hermite (n-2) x
    in
    (-1.0 ** float n) *. exp_term *. hermite n t

  let meyer t =
    let pi = Float.pi in
    let abs_t = abs_float t in
    if abs_t < 2.0 *. pi /. 3.0 then 0.0
    else if abs_t < 4.0 *. pi /. 3.0 then
      let y = 9.0 *. (abs_t /. (2.0 *. pi) -. 2.0 /. 3.0) in
      let v = y *. y *. (3.0 -. 2.0 *. y) in
      sin (pi *. v /. 2.0)
    else if abs_t < 8.0 *. pi /. 3.0 then
      let y = 9.0 /. 4.0 *. (1.0 -. abs_t /. (2.0 *. pi)) in
      let v = y *. y *. (3.0 -. 2.0 *. y) in
      cos (pi *. v /. 2.0)
    else 0.0

  let to_tensor_op f t =
    Tensor.float_value t |> f |> Tensor.of_float
end

module Analysis = struct
  open Torch

  let compute_spectrum signal wavelet_fn scales =
    let n = Tensor.shape signal |> List.hd in
    let num_scales = Tensor.shape scales |> List.hd in
    let spectrum = Tensor.zeros [num_scales; n] in
    
    for i = 0 to num_scales - 1 do
      let scale = Tensor.get scales [i] in
      let wavelet = Tensor.map (fun t ->
        let scaled_t = Tensor.float_value t /. Tensor.float_value scale in
        Function.to_tensor_op wavelet_fn scaled_t
      ) (Tensor.arange 0 n ~dtype:(T Float)) in
      
      let conv = Tensor.conv1d signal wavelet ~padding:Same in
      Tensor.copy_ conv ~src:(Tensor.slice spectrum ~dim:0 ~start:i ~length:1)
    done;
    spectrum

  let extract_ridges transform threshold =
    let _, n = match Tensor.shape transform with
      | [s; n] -> s, n
      | _ -> failwith "Invalid transform shape" in
    
    let ridges = Tensor.zeros_like transform in
    for t = 0 to n - 1 do
      let col = Tensor.slice transform ~dim:1 ~start:t ~length:1 in
      let max_val = Tensor.max col ~dim:0 ~keepdim:true |> fst in
      let mask = Tensor.gt col (max_val * Tensor.of_float threshold) in
      Tensor.copy_ mask 
        ~src:(Tensor.slice ridges ~dim:1 ~start:t ~length:1)
    done;
    ridges

  let compute_localization transform =
    let time_spread = Tensor.var transform ~dim:1 ~unbiased:true in
    let freq_spread = Tensor.var transform ~dim:0 ~unbiased:true in
    Tensor.float_value time_spread, Tensor.float_value freq_spread
end
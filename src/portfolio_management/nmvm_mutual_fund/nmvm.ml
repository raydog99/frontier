open Torch

type t = {
  mu: Tensor.t;
  gamma: Tensor.t;
  sigma: Tensor.t;
  z_dist: (unit -> Tensor.t);
}

type distribution = 
  | Normal of float * float
  | LogNormal of float * float
  | Gamma of float * float
  | InverseGaussian of float * float

let create mu gamma sigma dist =
  if Tensor.shape mu <> Tensor.shape gamma then
    failwith "Dimensions of mu and gamma must match";
  if Tensor.shape mu |> List.hd <> (Tensor.shape sigma |> List.hd) then
    failwith "Number of assets in mu must match rows in sigma";
  let z_dist = match dist with
    | Normal (mu, sigma) -> 
        if sigma <= 0. then failwith "Normal distribution sigma must be positive";
        (fun () -> Tensor.normal ~mean:(Tensor.scalar_float mu) ~std:(Tensor.scalar_float sigma) [])
    | LogNormal (mu, sigma) -> 
        if sigma <= 0. then failwith "LogNormal distribution sigma must be positive";
        (fun () -> Tensor.log_normal ~mean:(Tensor.scalar_float mu) ~std:(Tensor.scalar_float sigma) [])
    | Gamma (alpha, beta) -> 
        if alpha <= 0. || beta <= 0. then failwith "Gamma distribution parameters must be positive";
        (fun () -> Tensor.gamma ~alpha:(Tensor.scalar_float alpha) ~beta:(Tensor.scalar_float beta) [])
    | InverseGaussian (mu, lambda) -> 
        if mu <= 0. || lambda <= 0. then failwith "InverseGaussian distribution parameters must be positive";
        (fun () -> 
          let y = Tensor.normal ~mean:(Tensor.scalar_float 0.) ~std:(Tensor.scalar_float 1.) [] in
          let x = Tensor.((scalar_float mu) + (scalar_float (mu *. mu /. (2. *. lambda))) * y * y / (scalar_float lambda) - 
                          (scalar_float (mu /. (2. *. lambda))) * sqrt ((scalar_float (4. *. mu *. lambda)) * y * y + (scalar_float (mu *. mu)) * y * y * y * y))
          in
          x)
  in
  { mu; gamma; sigma; z_dist }

let sample model =
  let n = Tensor.shape model.mu |> List.hd in
  let normal = Tensor.randn [n] in
  let z_sample = model.z_dist () in
  Tensor.(model.mu + model.gamma * z_sample + sqrt z_sample * (matmul model.sigma normal))

let expected_return model =
  Tensor.(model.mu + model.gamma * (model.z_dist ()))

let covariance model =
  let ez = model.z_dist () in
  let ez2 = Tensor.(ez * ez) in
  Tensor.(matmul model.sigma (transpose 0 1 model.sigma) * ez +
          matmul model.gamma (transpose 0 1 model.gamma) * (ez2 - ez * ez))

let infinity_number model =
  let compute_laplace_transform s =
    match model.z_dist with
    | Normal (mu, sigma) ->
        exp (s *. mu +. 0.5 *. s *. s *. sigma *. sigma)
    | LogNormal (mu, sigma) ->
        exp (s *. exp (mu +. 0.5 *. sigma *. sigma))
    | Gamma (alpha, beta) ->
        (beta /. (beta -. s)) ** alpha
    | InverseGaussian (mu, lambda) ->
        exp (lambda /. mu *. (1. -. sqrt (1. -. 2. *. mu *. mu *. s /. lambda)))
  in
  let rec binary_search low high epsilon =
    if high -. low < epsilon then
      low
    else
      let mid = (low +. high) /. 2. in
      if compute_laplace_transform mid = infinity then
        binary_search low mid epsilon
      else
        binary_search mid high epsilon
  in
  -. (binary_search 0. 1000. 1e-6)

let generate_samples model num_samples =
  List.init num_samples (fun _ -> sample model)

let estimate_parameters historical_returns =
  let n = List.length historical_returns in
  let returns_tensor = Tensor.of_float1 (Array.of_list historical_returns) in
  
  let mu = Tensor.mean returns_tensor in
  
  let centered_returns = Tensor.sub returns_tensor mu in
  let sigma = Tensor.std centered_returns ~dim:[0] ~unbiased:true in
  
  (* Estimate gamma and z parameters using method of moments *)
  let m2 = Tensor.mean (Tensor.pow centered_returns (Tensor.scalar_float 2.)) in
  let m3 = Tensor.mean (Tensor.pow centered_returns (Tensor.scalar_float 3.)) in
  let m4 = Tensor.mean (Tensor.pow centered_returns (Tensor.scalar_float 4.)) in
  
  let gamma = Tensor.div m3 (Tensor.pow sigma (Tensor.scalar_float 3.)) in
  
  let k = Tensor.div (Tensor.sub m4 (Tensor.pow m2 (Tensor.scalar_float 2.))) 
                     (Tensor.pow m2 (Tensor.scalar_float 2.)) in
  
  (* Estimate z distribution parameters *)
  let z_mean = Tensor.div (Tensor.sub (Tensor.scalar_float 1.) k) 
                          (Tensor.sub k (Tensor.pow gamma (Tensor.scalar_float 2.))) in
  let z_var = Tensor.div (Tensor.sub k (Tensor.scalar_float 1.))
                         (Tensor.pow (Tensor.sub k (Tensor.pow gamma (Tensor.scalar_float 2.))) (Tensor.scalar_float 2.)) in
  
  let z_dist = 
    if Tensor.(to_float0 z_var < to_float0 z_mean) then
      Gamma (Tensor.to_float0 (Tensor.pow z_mean (Tensor.scalar_float 2.) / z_var),
             Tensor.to_float0 (z_mean / z_var))
    else
      InverseGaussian (Tensor.to_float0 z_mean, 
                       Tensor.to_float0 (Tensor.pow z_mean (Tensor.scalar_float 3.) / z_var))
  in
  
  create mu gamma sigma z_dist
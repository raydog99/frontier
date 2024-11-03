open Torch

let create_interpolation_noise ~shape ~t ~config =
  let open Tensor in
  let t = Scalar.float t in
  let t_ratio = div t (scalar config.terminal_time) in
  let time_scaling = 
    mul t_ratio (sub (scalar config.terminal_time) t_ratio) 
    |> div (scalar config.terminal_time)
    |> sqrt in
  randn shape |> mul time_scaling

let interpolate ~source ~target ~t ~config = 
  let open Tensor in
  let t_ratio = Scalar.float t /. config.terminal_time in
  let noise = 
    rand_like target 
    |> mul_scalar config.inject_noise_std in
  let weighted_target = 
    target 
    |> add noise
    |> mul_scalar t_ratio in
  let weighted_source =
    source
    |> mul_scalar (1. -. t_ratio) in
  let interpolation = add weighted_target weighted_source in
  let noise = create_interpolation_noise 
    ~shape:(shape interpolation) ~t ~config in
  add interpolation noise

let compute_cost_matrix x y =
  let open Tensor in
  let x_squared = sum (mul x x) ~dim:[1] ~keepdim:true in
  let y_squared = sum (mul y y) ~dim:[1] ~keepdim:true |> transpose ~dim0:0 ~dim1:1 in
  let xy = matmul x (transpose y ~dim0:0 ~dim1:1) in
  add x_squared (sub y_squared (mul_scalar xy 2.))

let sinkhorn ~cost_matrix ~epsilon ~num_iters =
  let open Tensor in
  let n, m = shape cost_matrix |> function
    | [n; m] -> n, m
    | _ -> failwith "Invalid cost matrix shape" in
  
  let mu = ones [n; 1] |> div_scalar (float_of_int n) in
  let nu = ones [m; 1] |> div_scalar (float_of_int m) in
  let kernel = div cost_matrix (neg_scalar epsilon) |> exp in
  
  let rec iterate u v i =
    if i = 0 then (u, v) else
    let u_new = div mu (matmul kernel v) in
    let v_new = div nu (matmul (transpose kernel ~dim0:0 ~dim1:1) u_new) in
    iterate u_new v_new (i - 1)
  in
  
  let u, v = iterate (ones [n; 1]) (ones [m; 1]) num_iters in
  mul_scalar (mul (mul kernel (u |> unsqueeze ~dim:1)) 
    (transpose v ~dim0:0 ~dim1:1)) epsilon

let sample_ot_maps ~source ~target ~epsilon ~num_iters =
  let cost_matrix = compute_cost_matrix source target in
  let transport_plan = sinkhorn ~cost_matrix ~epsilon ~num_iters in
  let barycentric_projection = matmul transport_plan target in
  source, barycentric_projection
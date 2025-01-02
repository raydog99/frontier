open Torch

type t = {
  a: Tensor.t;
  b: Tensor.t;
  c: Tensor.t;
  q: Tensor.t;
  r: Tensor.t;
  device: Device.t;
}

let create ~d_x ~d_y ~device =
  let a = Tensor.randn [d_x; d_x] ~device in
  let b = Tensor.randn [d_x; d_y] ~device in
  let c = Tensor.randn [d_y; d_x] ~device in
  let q = Tensor.randn [d_x; d_x] ~device |> Tensor.matmul ~other:(Tensor.transpose a ~dim0:0 ~dim1:1) in
  let r = Tensor.randn [d_y; d_y] ~device |> Tensor.matmul ~other:(Tensor.transpose c ~dim0:0 ~dim1:1) in
  { a; b; c; q; r; device }

let generate t ~num_timesteps ~batch_size =
  let d_x = Tensor.shape t.a |> List.hd in
  let d_y = Tensor.shape t.c |> List.hd in

  let x_init = Tensor.randn [batch_size; d_x] ~device:t.device in
  let x_seq = Tensor.empty [num_timesteps; batch_size; d_x] ~device:t.device in
  let y_seq = Tensor.empty [num_timesteps; batch_size; d_y] ~device:t.device in

  Tensor.copy_ (Tensor.slice x_seq ~dim:0 ~start:0 ~end_:1 ~step:1) x_init;

  for i = 1 to num_timesteps - 1 do
    let prev_x = Tensor.slice x_seq ~dim:0 ~start:(i-1) ~end_:i ~step:1 in
    let process_noise = Tensor.randn [batch_size; d_x] ~device:t.device in
    let new_x = Tensor.(matmul t.a prev_x + matmul t.q process_noise) in
    Tensor.copy_ (Tensor.slice x_seq ~dim:0 ~start:i ~end_:(i+1) ~step:1) new_x;

    let measurement_noise = Tensor.randn [batch_size; d_y] ~device:t.device in
    let y = Tensor.(matmul t.c (Tensor.slice x_seq ~dim:0 ~start:i ~end_:(i+1) ~step:1) + matmul t.r measurement_noise) in
    Tensor.copy_ (Tensor.slice y_seq ~dim:0 ~start:i ~end_:(i+1) ~step:1) y;
  done;

  (y_seq, x_seq)
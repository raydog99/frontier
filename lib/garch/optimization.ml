open Torch

let adam params ~loss_fn ~learning_rate ~beta1 ~beta2 ~epsilon ~max_iter =
  let m = List.map (fun p -> Tensor.zeros_like p) params in
  let v = List.map (fun p -> Tensor.zeros_like p) params in
  let rec loop iter m v =
    if iter > max_iter then params
    else
      let loss, grads = Tensor.grad_and_value params (fun ps -> loss_fn ps) in
      let m' = List.map2 (fun mi gi -> Tensor.(mi * Scalar.f beta1 + gi * Scalar.f (1.0 -. beta1))) m grads in
      let v' = List.map2 (fun vi gi -> Tensor.(vi * Scalar.f beta2 + gi * gi * Scalar.f (1.0 -. beta2))) v grads in
      let m_hat = List.map (fun mi -> Tensor.(mi / (Scalar.f (1.0 -. beta1 ** float_of_int iter)))) m' in
      let v_hat = List.map (fun vi -> Tensor.(vi / (Scalar.f (1.0 -. beta2 ** float_of_int iter)))) v' in
      let params' = List.map2 (fun p mh vh -> 
        Tensor.(p - Scalar.f learning_rate * mh / (sqrt vh + Scalar.f epsilon))
      ) params m_hat v_hat in
      Garch_error_handling.check_convergence (Tensor.to_float0_exn loss) 1e-6;
      loop (iter + 1) m' v'
  in
  loop 1 m v
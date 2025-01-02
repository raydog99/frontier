open Torch

let positive_definite_projection tensor =
  let u, s, v = Tensor.svd tensor in
  let s' = Tensor.relu s in
  Tensor.matmul 
    (Tensor.matmul u (Tensor.diag s'))
    (Tensor.transpose v)

let parallel_transport_tensor start_point end_point vector christoffel =
  let velocity = Tensor.sub end_point start_point in
  let transported = ref vector in
  let steps = 100 in
  let dt = 1. /. float_of_int steps in
  
  for step = 0 to steps - 1 do
    let t = float_of_int step *. dt in
    let current = 
      Tensor.add start_point 
        (Tensor.mul_scalar velocity t) in
    
    let correction = 
      Array.fold_left (fun acc symbols ->
        Array.fold_left (fun acc_inner row ->
          Tensor.add acc_inner (
            Array.fold_left2 (fun a s v ->
              Tensor.add a 
                (Tensor.mul_scalar s 
                   (Tensor.dot !transported v))
            ) (Tensor.zeros_like !transported)
              row
              (Array.to_list 
                 (Tensor.to_array1 velocity))
          )
        ) acc symbols
      ) (Tensor.zeros_like !transported)
        (christoffel current) in
    
    transported := 
      Tensor.sub !transported 
        (Tensor.mul_scalar correction dt)
  done;
  !transported

let geodesic_distance metric x y =
  let diff = Tensor.sub y x in
  let g = metric x in
  sqrt (Tensor.dot diff (Tensor.mv g diff))
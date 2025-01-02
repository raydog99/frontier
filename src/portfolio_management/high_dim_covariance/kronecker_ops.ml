open Torch

let matrix_vector_product mat vec dim =
  let vec_reshaped = Tensor.reshape vec [|dim; dim|] in
  let result = Tensor.mm (Tensor.mm mat vec_reshaped) 
    (Tensor.transpose mat ~dim0:0 ~dim1:1) in
  Tensor.reshape result [|-1|]

let transpose_product mat vec dim =
  let vec_reshaped = Tensor.reshape vec [|dim; dim|] in
  let result = Tensor.mm 
    (Tensor.transpose vec_reshaped ~dim0:0 ~dim1:1) 
    (Tensor.transpose mat ~dim0:0 ~dim1:1) in
  Tensor.reshape result [|-1|]

let efficient_outer_product a b =
  let a_flat = Tensor.reshape a [|-1|] in
  let b_flat = Tensor.reshape b [|-1|] in
  Tensor.mm (Tensor.unsqueeze a_flat ~dim:1)
    (Tensor.unsqueeze b_flat ~dim:0)
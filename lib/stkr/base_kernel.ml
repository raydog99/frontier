open Torch
open Types
  
let center_kernel k x y =
    let batch_size = Tensor.size x 0 in
    let k_xy = k x y in
    let k_x = Tensor.mean k_xy ~dim:[1] ~keepdim:true in
    let k_y = Tensor.mean k_xy ~dim:[0] ~keepdim:true in
    let k_mean = Tensor.mean k_xy |> Tensor.expand_as k_xy in
    k_xy - k_x - k_y + k_mean

let rbf_kernel sigma x y =
    let x_norm = Tensor.pow (Tensor.norm2 x ~dim:[1] ~keepdim:true) 2 in
    let y_norm = Tensor.pow (Tensor.norm2 y ~dim:[1] ~keepdim:true) 2 in
    let xy = Tensor.mm x (Tensor.transpose y 0 1) in
    let dist = x_norm + Tensor.transpose y_norm 0 1 - (2. *. xy) in
    Tensor.exp (dist /. (-2. *. (sigma *. sigma)))
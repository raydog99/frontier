open Torch

let eigensystem tensor =
    let eigenvalues, eigenvectors = Tensor.symeig tensor ~eigenvectors:true in
    let sorted_indices = Tensor.argsort eigenvalues ~descending:true in
    let sorted_eigenvalues = Tensor.index_select eigenvalues 0 sorted_indices in
    let sorted_eigenvectors = Tensor.index_select eigenvectors 1 sorted_indices in
    sorted_eigenvalues, sorted_eigenvectors

let solve_conjugate_gradient a b epsilon max_iter =
    let n = Tensor.size b 0 in
    let x = Tensor.zeros [n] in
    let r = Tensor.sub b (Tensor.matmul a x) in
    let p = Tensor.copy r in
    let r_norm = Tensor.dot r r in
    
    let rec iterate x p r k =
      if k >= max_iter || Tensor.float_value r_norm < epsilon then x
      else
        let ap = Tensor.matmul a p in
        let alpha = Tensor.div r_norm (Tensor.dot p ap) in
        let x' = Tensor.add x (Tensor.mul alpha p) in
        let r' = Tensor.sub r (Tensor.mul alpha ap) in
        let r_norm' = Tensor.dot r' r' in
        let beta = Tensor.div r_norm' r_norm in
        let p' = Tensor.add r' (Tensor.mul beta p) in
        iterate x' p' r' (k + 1)
    in
    iterate x p r 0
open Torch
open Types

module SparseOps = struct
  type csr_matrix = {
    values: Tensor.t;
    row_ptr: int array;
    col_idx: int array;
  }

  let to_csr sparse_mat =
    let rows, _ = sparse_mat.shape in
    let row_ptr = Array.make (rows + 1) 0 in
    let sorted_indices = Array.sort (fun (r1, _) (r2, _) -> 
      compare r1 r2) sparse_mat.indices in
    
    (* Count elements per row *)
    Array.iter (fun (r, _) -> row_ptr.(r + 1) <- row_ptr.(r + 1) + 1) 
      sorted_indices;
    
    (* Cumulative sum *)
    for i = 1 to rows do
      row_ptr.(i) <- row_ptr.(i) + row_ptr.(i - 1)
    done;
    
    let col_idx = Array.map snd sorted_indices in
    {
      values = sparse_mat.values;
      row_ptr;
      col_idx;
    }

  let csr_matmul csr_mat vec =
    let n = Array.length csr_mat.row_ptr - 1 in
    let result = Tensor.zeros [n] in
    
    for i = 0 to n - 1 do
      let row_start = csr_mat.row_ptr.(i) in
      let row_end = csr_mat.row_ptr.(i + 1) in
      let mut_sum = ref 0. in
      for j = row_start to row_end - 1 do
        let col = csr_mat.col_idx.(j) in
        let val_ = Tensor.get csr_mat.values [j] in
        let vec_val = Tensor.get vec [col] in
        mut_sum := !mut_sum +. val_ *. vec_val
      done;
      Tensor.set result [i] !mut_sum
    done;
    result
end

(* Convert dense to sparse with threshold *)
let to_sparse tensor threshold =
  let rows, cols = Tensor.size tensor 0, Tensor.size tensor 1 in
  let values, indices = 
    Tensor.fold_left2 (fun (vs, is) i j v ->
      if abs_float v > threshold then
        (v :: vs, (i, j) :: is)
      else (vs, is)
    ) ([], []) tensor
  in
  {
    values = Tensor.of_float1 (Array.of_list values);
    indices = Array.of_list indices;
    shape = (rows, cols);
  }

(* STKR-Prop for simple transformations *)
let propagate_simple_sparse kernel poly_params params x_labeled y_labeled x_unlabeled =
  let n_labeled = Tensor.size x_labeled 0 in
  let n_total = n_labeled + (Tensor.size x_unlabeled 0) in
  let x_all = Tensor.cat [x_labeled; x_unlabeled] 0 in
  
  (* Compute and sparsify kernel matrix *)
  let k_matrix = kernel x_all x_all in
  let sparse_k = to_sparse k_matrix params.epsilon in
  let csr_k = SparseOps.to_csr sparse_k in
  
  let alpha = Tensor.zeros [n_labeled] in
  let rec iterate alpha iter =
    if iter >= params.max_iter then alpha
    else
      (* Efficient sparse operations *)
      let alpha_ext = Tensor.cat [alpha; Tensor.zeros [n_total - n_labeled]] 0 in
      let v = SparseOps.csr_matmul csr_k alpha_ext in
      
      (* Update using polynomial coefficients *)
      let v = List.fold_left (fun acc p ->
        let coeff = poly_params.coefficients.(p-1) in
        let term = SparseOps.csr_matmul csr_k v in
        Tensor.add acc (Tensor.mul term coeff)
      ) v (List.init poly_params.degree (fun i -> i + 1)) in
      
      let u = Tensor.narrow v 0 0 n_labeled in
      let error = Tensor.sub u y_labeled in
      let error_norm = Tensor.norm2 error in
      
      if Tensor.float_value error_norm < params.epsilon then alpha
      else
        let alpha' = Tensor.sub alpha (Tensor.mul error params.learning_rate) in
        iterate alpha' (iter + 1)
  in
  iterate alpha 0

(* Inverse Laplacian propagation *)
let propagate_inverse_sparse kernel params x_labeled y_labeled x_unlabeled =
  let n_labeled = Tensor.size x_labeled 0 in
  let n_total = n_labeled + (Tensor.size x_unlabeled 0) in
  let x_all = Tensor.cat [x_labeled; x_unlabeled] 0 in
  
  let k_matrix = kernel x_all x_all in
  let sparse_k = to_sparse k_matrix params.epsilon in
  let csr_k = SparseOps.to_csr sparse_k in
  
  let theta = Tensor.zeros [n_total] in
  let y_ext = Tensor.cat [y_labeled; Tensor.zeros [n_total - n_labeled]] 0 in
  
  let rec iterate theta iter =
    if iter >= params.max_iter then theta
    else
      (* Efficient inverse Laplacian operations *)
      let v = SparseOps.csr_matmul csr_k theta in
      let u = Tensor.cat [
        Tensor.narrow v 0 0 n_labeled;
        Tensor.zeros [n_total - n_labeled]
      ] 0 in
      
      (* Apply inverse Laplacian transformation *)
      let u = Tensor.add u (Tensor.mul v params.lambda) in
      let error = Tensor.sub u y_ext in
      let error_norm = Tensor.norm2 error in
      
      if Tensor.float_value error_norm < params.epsilon then theta
      else
        let theta' = Tensor.sub theta (Tensor.mul error params.learning_rate) in
        iterate theta' (iter + 1)
  in
  iterate theta 0
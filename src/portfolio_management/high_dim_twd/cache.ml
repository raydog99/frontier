open Torch

let eigendecomp = Hashtbl.create 10
let hd_lca_values = Hashtbl.create 1000

let clear () =
  Hashtbl.clear eigendecomp;
  Hashtbl.clear hd_lca_values
  
let get_eigendecomp op =
  match Hashtbl.find_opt eigendecomp op with
  | Some (vals, vecs) -> vals, vecs
  | None ->
      let vals, vecs = Tensor.symeig op ~eigenvectors:true in
      Hashtbl.add eigendecomp op (vals, vecs);
      vals, vecs
      
let get_hd_lca p1 p2 =
  let key = (Tensor.hash p1, Tensor.hash p2) in
  match Hashtbl.find_opt hd_lca_values key with
  | Some lca -> lca
  | None ->
      let lca = Twd_impl.compute_multi_scale_hd_lca p1 p2 in
      Hashtbl.add hd_lca_values key lca;
      lca
open Torch

type t = 
  | KL
  | Jensen_Shannon
  | Custom of (Tensor.t -> Tensor.t -> Tensor.t)

let compute = function
  | KL -> fun p q ->
      let eps = Tensor.full [1] 1e-10 in
      Tensor.(sum (mul p (log (div (add p eps) (add q eps)))))
  | Jensen_Shannon -> fun p q ->
      let m = Tensor.(div (add p q) (float 2.0)) in
      let kl = fun x y -> 
        let eps = Tensor.full [1] 1e-10 in
        Tensor.(sum (mul x (log (div (add x eps) (add y eps))))) in
      Tensor.(div (add (kl p m) (kl q m)) (float 2.0))
  | Custom f -> f
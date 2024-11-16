open Torch

type point = Tensor.t

module Dataset = struct
  type labeled = {
    x: Tensor.t;
    y: Tensor.t;
  }

  type unlabeled = {
    x: Tensor.t;
  }

  let create_labeled xs ys =
    {x = Tensor.stack xs; y = Tensor.of_float1 ys}

  let create_unlabeled xs =
    {x = Tensor.stack xs}
end

module Config = struct
  type t = {
    epsilon: float;              
    lambda: float;              
    truncation_k: int;          
    truncation_q: int;          
    time: float;                
  }

  let create ~epsilon ~lambda ~truncation_k ~truncation_q ~time = {
    epsilon;
    lambda;
    truncation_k;
    truncation_q;
    time;
  }
end
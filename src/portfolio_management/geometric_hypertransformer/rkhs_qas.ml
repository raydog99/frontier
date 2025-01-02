open Torch

type kernel = {
  k: Tensor.t -> Tensor.t -> float;
  grad_k: Tensor.t -> Tensor.t -> Tensor.t;
  feature_map: Tensor.t -> Tensor.t option;
}

type point = {
  value: Tensor.t;
  kernel: kernel;
  norm: float;
}

type t = point

type mixing_function = {
  mix: float array -> point array -> point;
  constant: float;
  power: int;
}

let gaussian_kernel sigma = {
  k = (fun x y ->
    let diff = Tensor.sub x y in
    let sq_dist = Tensor.dot diff diff in
    exp (-. Tensor.float_value sq_dist /. (2. *. sigma *. sigma)));
  
  grad_k = (fun x y ->
    let diff = Tensor.sub x y in
    Tensor.mul_scalar diff 
      (-1. /. (sigma *. sigma)));
  
  feature_map = (fun x -> None);  (* Infinite dimensional *)
}

let create_mixing constant power =
  let mix weights points =
    match (Array.get points 0).kernel.feature_map 
            (Array.get points 0).value with
    | Some feat_map ->
        (* Mix in feature space if available *)
        let mixed_repr = Array.fold_left2
          (fun acc w p ->
            match p.kernel.feature_map p.value with
            | Some fm -> 
                Tensor.add acc (Tensor.mul_scalar fm w)
            | None -> acc)
          (Tensor.zeros_like feat_map)
          weights points in
        
        {(Array.get points 0) with 
         value = mixed_repr;
         norm = sqrt (
           (Array.get points 0).kernel.k mixed_repr mixed_repr)}

    | None ->
        (* Mix using kernel trick *)
        let gram = Array.make_matrix 
          (Array.length points) 
          (Array.length points) 
          0. in
        
        for i = 0 to Array.length points - 1 do
          for j = 0 to Array.length points - 1 do
            gram.(i).(j) <- 
              points.(i).kernel.k 
                points.(i).value points.(j).value
          done
        done;
        
        {(Array.get points 0) with
         value = Array.fold_left2
           (fun acc w p ->
              Tensor.add acc 
                (Tensor.mul_scalar p.value w))
           (Tensor.zeros_like (Array.get points 0).value)
           weights 
           points}
  in
  {mix; constant; power}
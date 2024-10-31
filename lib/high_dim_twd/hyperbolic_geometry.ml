open Torch

let poincare_distance x y =
  let n = Tensor.size x 0 in
  let x_last = Tensor.select x (n-1) 0 in
  let y_last = Tensor.select y (n-1) 0 in
  let diff = Tensor.sub x y in
  let norm_squared = Tensor.norm diff ~p:2 in
  let norm_squared = Tensor.mul norm_squared norm_squared in
  
  let denom = Tensor.mul 
    (Tensor.mul_scalar (Tensor.sqrt (Tensor.mul x_last y_last)) 2.0) in
  let arg = Tensor.div norm_squared denom in
  Tensor.float_value (Tensor.mul_scalar (Tensor.asinh arg) 2.0)

let frechet_mean points =
  let n = Tensor.size points 0 in
  if n = 0 then Tensor.zeros [1]
  else if n = 1 then Tensor.select points 0 0
  else
    let mean = ref (Tensor.select points 0 0) in
    let max_iter = 100 in
    let tol = 1e-6 in
    
    for _ = 1 to max_iter do
      let grad = Tensor.zeros_like !mean in
      for i = 0 to n - 1 do
        let p = Tensor.select points 0 i in
        let dist = poincare_distance !mean p in
        let dir = Tensor.sub p !mean in
        let dir_norm = Tensor.norm dir in
        if Tensor.float_value dir_norm > tol then begin
          let normalized_dir = Tensor.div dir dir_norm in
          let weighted_dir = Tensor.mul_scalar normalized_dir dist in
          Tensor.add_ grad weighted_dir
        end
      done;
      
      let grad_norm = Tensor.norm grad in
      if Tensor.float_value grad_norm < tol then
        raise Exit
      else
        let step_size = 0.1 in
        let update = Tensor.mul_scalar grad (step_size /. float_of_int n) in
        mean := Tensor.add !mean update
    done;
    !mean

let stable_poincare_distance x y =
  let n = Tensor.size x 0 in
  let x_last = Tensor.select x (n-1) 0 in
  let y_last = Tensor.select y (n-1) 0 in
  let diff = Tensor.sub x y in
  let norm_squared = Tensor.norm diff ~p:2 in
  let norm_squared = Tensor.mul norm_squared norm_squared in
  
  let eps = 1e-10 in
  let denom = Tensor.mul 
    (Tensor.mul_scalar 
       (Tensor.sqrt (Tensor.add (Tensor.mul x_last y_last) eps)) 2.0) in
  let arg = Tensor.div norm_squared denom in
  let arg = Tensor.clamp arg ~min:(-1e6) ~max:1e6 in
  Tensor.float_value (Tensor.mul_scalar (Tensor.asinh arg) 2.0)
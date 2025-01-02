open Torch
open Types

let solve_odes (params: market_params) (T: float) : ode_solution =
  let n_steps = 1000 in
  let dt = T /. float_of_int n_steps in
  
  let a00 = Tensor.zeros [n_steps + 1] in
  let a11 = Tensor.zeros [n_steps + 1] in
  let a10 = Tensor.zeros [n_steps + 1] in
  let b0 = Tensor.zeros [n_steps + 1] in
  let b1 = Tensor.zeros [n_steps + 1] in
  let c = Tensor.zeros [n_steps + 1] in
  
  Tensor.set a00 [|n_steps|] (Scalar.float 0.7);
  Tensor.set a11 [|n_steps|] (Scalar.float 0.2);
  Tensor.set a10 [|n_steps|] (Scalar.float 0.3);
  Tensor.set b0 [|n_steps|] (Scalar.float (-1.3));
  Tensor.set b1 [|n_steps|] (Scalar.float (-0.7));
  Tensor.set c [|n_steps|] (Scalar.float 1.2);
  
  for i = n_steps - 1 downto 0 do
    let t = float_of_int i *. dt in
    let a00_t = Tensor.get a00 [|i + 1|] in
    let a11_t = Tensor.get a11 [|i + 1|] in
    let a10_t = Tensor.get a10 [|i + 1|] in
    let b0_t = Tensor.get b0 [|i + 1|] in
    let b1_t = Tensor.get b1 [|i + 1|] in
    let c_t = Tensor.get c [|i + 1|] in
    
    let da00 = Tensor.(
      sub (mul (float (-1. *. params.gamma)) (mul a00_t (mul (float params.sigma0) (float params.sigma0))))
          (add (mul (float (-1. *. params.gamma)) (mul (transpose a10_t) (mul (float params.sigma) (float params.sigma))))
               (mul (float (2. *. params.k0)) a00_t))
    ) in
    
    let da11 = Tensor.(
      sub (mul (float (-1. *. params.gamma)) (mul a11_t (mul (float params.sigma) (float params.sigma))))
          (mul (float (2. *. params.k)) a11_t)
    ) in
    
    let da10 = Tensor.(
      sub (mul (float (-1. *. params.gamma)) (mul a10_t (mul (float params.sigma0) (float params.sigma0))))
          (add (mul (float (-1. *. params.gamma)) (mul a11_t (mul (float params.sigma) (float params.sigma))))
               (mul (float (params.k0 +. params.k)) a10_t))
    ) in
    
    let db0 = Tensor.(
      add (mul (sub (mul (float (-1. *. params.gamma)) (mul a00_t (mul (float params.sigma0) (float params.sigma0))))
                    (float params.k0))
               b0_t)
          (sub (mul (float (-1. *. params.gamma)) (mul (transpose a10_t) (mul (float params.sigma) (float params.sigma))))
               (add (mul (float params.k0) (mul a00_t (float params.m0)))
                    (mul (float params.k) (transpose a10_t))))
    ) in
    
    let db1 = Tensor.(
      add (mul (sub (mul (float (-1. *. params.gamma)) (mul a11_t (mul (float params.sigma) (float params.sigma))))
                    (float params.k))
               b1_t)
          (sub (mul (float (-1. *. params.gamma))
                    (add (mul a10_t (mul (float params.sigma0) (float params.sigma0)))
                         (mul (transpose a10_t) b0_t)))
               (add (mul (float params.k) (mul a11_t (float params.m)))
                    (mul (float params.k0) (mul a10_t (float params.m0)))))
    ) in
    
    let dc = Tensor.(
      sub (add (mul (float (-0.5 *. params.gamma)) (mul (transpose b0_t) (mul (float params.sigma0) (float params.sigma0))))
               (add (mul (float (-0.5 *. params.gamma)) (mul (transpose b1_t) (mul (float params.sigma) (float params.sigma))))
                    (add (mul (float params.k0) (mul (transpose b0_t) (float params.m0)))
                         (mul (float params.k) (mul (transpose b1_t) (float params.m))))))
          (add (mul (float 0.5) (trace (mul a00_t (mul (float params.sigma0) (float params.sigma0)))))
               (mul (float 0.5) (trace (mul a11_t (mul (float params.sigma) (float params.sigma))))))
    ) in
    
    Tensor.set a00 [|i|] Tensor.(sub a00_t (mul (float dt) da00));
    Tensor.set a11 [|i|] Tensor.(sub a11_t (mul (float dt) da11));
    Tensor.set a10 [|i|] Tensor.(sub a10_t (mul (float dt) da10));
    Tensor.set b0 [|i|] Tensor.(sub b0_t (mul (float dt) db0));
    Tensor.set b1 [|i|] Tensor.(sub b1_t (mul (float dt) db1));
    Tensor.set c [|i|] Tensor.(sub c_t (mul (float dt) dc));
  done;
  
  { a00; a11; a10; b0; b1; c }
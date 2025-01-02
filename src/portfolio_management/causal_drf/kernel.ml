open Torch

type kernel_type = 
  | Gaussian of float
  | Laplace of float 
  | InverseMultiquadric of float

type kernel_params = {
  bandwidth: float;
  degree: int option;
  scale: float;
  normalize: bool;
}

let evaluate_kernel kernel x1 x2 =
  match kernel with
  | Gaussian sigma -> 
      let diff = Tensor.sub x1 x2 in
      let squared_dist = Tensor.(sum (mul diff diff)) in
      Tensor.exp (Tensor.div squared_dist (Tensor.float_tensor [|-2.0 *. sigma *. sigma|]))
  | Laplace sigma ->
      let diff = Tensor.sub x1 x2 in
      Tensor.exp (Tensor.div (Tensor.norm diff) 
        (Tensor.float_tensor [|-1.0 *. sigma|]))
  | InverseMultiquadric c ->
      let diff = Tensor.sub x1 x2 in
      Tensor.pow (Tensor.add (Tensor.dot diff diff) 
        (Tensor.float_tensor [|c *. c|])) (-0.5)

let compute_kernel_matrix data kernel =
  let n = Tensor.size2 data.features 0 in
  let k_mat = Tensor.zeros [|n; n|] in
  
  for i = 0 to n-1 do
    for j = i to n-1 do
      let x_i = Tensor.get data.outcome i in
      let x_j = Tensor.get data.outcome j in
      let k_ij = evaluate_kernel kernel x_i x_j in
      Tensor.set k_mat [|i; j|] (Tensor.float_value k_ij);
      if i <> j then
        Tensor.set k_mat [|j; i|] (Tensor.float_value k_ij)
    done
  done;
  k_mat

let estimate_cme data kernel x =
  let n = Tensor.size2 data.features 0 in
  let weights = ref (Tensor.zeros [|n|]) in
  
  (* Calculate weights based on distance to x *)
  for i = 0 to n-1 do
    let x_i = Tensor.get data.features i in
    let dist = evaluate_kernel kernel x_i x in
    Tensor.set !weights [|i|] (Tensor.float_value dist)
  done;
  
  (* Normalize weights *)
  let sum_weights = Tensor.sum !weights in
  weights := Tensor.div !weights sum_weights;
  
  (* Compute weighted sum *)
  let cme = ref (Tensor.zeros_like (Tensor.get data.outcome 0)) in
  for i = 0 to n-1 do
    let w_i = Tensor.get !weights [|i|] in
    let y_i = Tensor.get data.outcome i in
    cme := Tensor.add !cme (Tensor.mul_scalar y_i w_i)
  done;
  !cme

let approximate_kernel data n_features bandwidth =
  let d = Tensor.size2 data.outcome 1 in
  let n = Tensor.size2 data.outcome 0 in
  
  (* Generate random features *)
  let omega = Tensor.mul_scalar 
    (Tensor.randn [|n_features; d|]) 
    (1.0 /. bandwidth) in
  let b = Tensor.rand [|n_features|] *. 2.0 *. Float.pi in
  
  (* Compute feature map *)
  let phi = Tensor.zeros [|n; n_features|] in
  for i = 0 to n-1 do
    let x = Tensor.get data.outcome i in
    let wx_b = Tensor.add 
      (Tensor.matmul omega (Tensor.reshape x [|d; 1|])) 
      (Tensor.reshape b [|n_features; 1|]) in
    let features = Tensor.cos 
      (Tensor.mul_scalar wx_b (sqrt (2.0 /. float_of_int n_features))) in
    Tensor.copy_ (Tensor.select phi 0 i) features
  done;
  phi

let nystrom_approximation data m bandwidth =
  let n = Tensor.size2 data.outcome 0 in
  
  (* Select landmark points *)
  let landmarks = Array.init m (fun _ -> Random.int n) in
  
  (* Compute kernel submatrices *)
  let k_mm = compute_kernel_matrix 
    {data with outcome = Tensor.index_select data.outcome 0 
      (Tensor.of_int1 landmarks)} 
    (Gaussian bandwidth) in
  let k_nm = Tensor.zeros [|n; m|] in
  
  for i = 0 to n-1 do
    for j = 0 to m-1 do
      let k_ij = evaluate_kernel 
        (Gaussian bandwidth) 
        (Tensor.get data.outcome i) 
        (Tensor.get data.outcome landmarks.(j)) in
      Tensor.set k_nm [|i; j|] (Tensor.float_value k_ij)
    done
  done;
  
  (* Compute approximation *)
  let eigenvals, eigenvecs = Tensor.symeig k_mm in
  let sqrt_einv = Tensor.map (fun x -> 
    if x > 1e-10 then 1.0 /. sqrt x else 0.0) eigenvals in
  
  let feat = Tensor.matmul k_nm 
    (Tensor.matmul eigenvecs (Tensor.diag sqrt_einv)) in
  feat

let median_heuristic data =
  let n = Tensor.size2 data.outcome 0 in
  let distances = ref [] in
  
  for i = 0 to min n 1000 do
    let idx1 = Random.int n in
    let idx2 = Random.int n in
    if idx1 <> idx2 then
      let x1 = Tensor.get data.outcome idx1 in
      let x2 = Tensor.get data.outcome idx2 in
      let dist = Tensor.(sum (mul (sub x1 x2) (sub x1 x2))) in
      distances := Tensor.float_value dist :: !distances
  done;
  
  let sorted = List.sort compare !distances in
  sqrt (List.nth sorted (List.length sorted / 2) /. 2.0)

let verify_kernel_properties kernel =
  let verify_boundedness = function
    | Gaussian _ | Laplace _ | InverseMultiquadric _ -> true in
  
  let verify_characteristic = function
    | Gaussian _ | Laplace _ -> true
    | InverseMultiquadric _ -> false in
  
  let verify_translation_invariant = function
    | Gaussian _ | Laplace _ -> true
    | InverseMultiquadric _ -> false in
  
  {
    bounded = verify_boundedness kernel;
    characteristic = verify_characteristic kernel;
    translation_invariant = verify_translation_invariant kernel;
  }
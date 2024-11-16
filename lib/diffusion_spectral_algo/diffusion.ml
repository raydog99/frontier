open Torch

module DiffusionSpace = struct
  type t = {
    heat_kernel: HeatKernel.t;
    time: float;
    manifold: Manifold.t;
  }

  let create manifold heat_kernel time = {
    heat_kernel;
    time;
    manifold;
  }

  let inner_product space f g =
    let h = HeatKernel.build_matrix space.heat_kernel space.manifold.points space.time in
    let f_exp = Tensor.(exp (space.heat_kernel.eigenvalues * (space.time /. 2.))) in
    let g_exp = Tensor.(exp (space.heat_kernel.eigenvalues * (space.time /. 2.))) in
    let f_transformed = Tensor.(matmul h f) in
    let g_transformed = Tensor.(matmul h g) in
    Tensor.(sum (f_transformed * g_transformed)) |> Tensor.float_value

  let embed space f =
    let h = HeatKernel.build_matrix space.heat_kernel space.manifold.points space.time in
    Tensor.(matmul h f)

  let embed_adjoint space f =
    let h = HeatKernel.build_matrix space.heat_kernel space.manifold.points space.time in
    Tensor.(matmul (transpose h ~dim0:0 ~dim1:1) f)
end

module IntegralOperator = struct
  type t = {
    kernel: HeatKernel.t;
    points: Tensor.t;
  }

  let create kernel points = {
    kernel;
    points;
  }

  let apply op f =
    let h = HeatKernel.build_matrix op.kernel op.points 1.0 in
    Tensor.(matmul h f)

  let operator_norm op =
    let h = HeatKernel.build_matrix op.kernel op.points 1.0 in
    let eigenvalues, _ = Tensor.linalg_eigh h in
    Tensor.max eigenvalues |> Tensor.float_value

  let power_iterate op max_iter tol =
    let n = Tensor.size op.points |> List.hd in
    let v = Tensor.randn [n; 1] in
    let v = Tensor.(v / norm v) in
    let rec iterate i prev_lambda v =
      if i >= max_iter then prev_lambda
      else
        let v_new = apply op v in
        let lambda = Tensor.(norm v_new) |> Tensor.float_value in
        let v_new = Tensor.(v_new / norm v_new) in
        if abs_float (lambda -. prev_lambda) < tol then lambda
        else iterate (i + 1) lambda v_new
    in
    iterate 0 0. v
end

module PowerSpace = struct
  type t = {
    alpha: float;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
    time: float;
  }

  let create alpha eigenvalues eigenvectors time = {
    alpha;
    eigenvalues;
    eigenvectors;
    time;
  }

  let norm space f =
    let coeffs = Tensor.matmul Tensor.(transpose space.eigenvectors) f in
    let powered = Tensor.zeros_like coeffs in
    for i = 0 to (Tensor.size coeffs |> List.hd) - 1 do
      let lambda = Tensor.float_value (Tensor.get space.eigenvalues [i]) in
      let coeff = Tensor.float_value (Tensor.get coeffs [i]) in
      let powered_coeff = 
        coeff *. (lambda ** space.alpha) *. 
        exp(-. space.alpha *. lambda *. space.time) in
      Tensor.set powered [i] powered_coeff
    done;
    Tensor.norm powered |> Tensor.float_value

  let project space f =
    let coeffs = Tensor.matmul Tensor.(transpose space.eigenvectors) f in
    let projected = Tensor.zeros_like space.eigenvectors in
    let p = float_of_int (List.hd (Tensor.size space.eigenvalues)) in
    for i = 0 to (Tensor.size coeffs |> List.hd) - 1 do
      let lambda = Tensor.float_value (Tensor.get space.eigenvalues [i]) in
      let coeff = Tensor.float_value (Tensor.get coeffs [i]) in
      let scale = p ** (space.alpha /. 2.) *. 
                 exp(-. space.alpha *. lambda *. space.time /. 2.) in
      let vec = Tensor.slice space.eigenvectors [i] in
      Tensor.copy_ (Tensor.slice projected [i]) Tensor.(vec * (coeff *. scale))
    done;
    projected
end
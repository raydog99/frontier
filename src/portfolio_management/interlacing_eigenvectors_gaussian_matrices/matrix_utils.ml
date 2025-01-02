open Torch

let create_symmetric_gaussian_matrix n =
  Utils.assert_positive_float "n" (float_of_int n);
  let lower_triangular = Tensor.randn [n; n] ~dtype:Kind.Float in
  let symmetric = Tensor.(lower_triangular + transpose lower_triangular ~dim0:0 ~dim1:1) in
  Tensor.(symmetric / (float_scalar (sqrt (float_of_int (2 * n)))))

let create_dyson_brownian_motion n t =
  Utils.assert_positive_float "t" t;
  let a = create_symmetric_gaussian_matrix n in
  let h = create_symmetric_gaussian_matrix n in
  Tensor.(a + (h * (float_scalar (sqrt t))))

let truncate_matrix x n =
  let full_size = Tensor.size x ~dim:0 in
  if n > full_size then
    failwith (Printf.sprintf "Truncation size %d exceeds matrix size %d" n full_size);
  let padded = Tensor.pad x ~pad:[0; 0; 0; 0] ~mode:"constant" ~value:0. in
  Tensor.narrow padded ~dim:0 ~start:0 ~length:n |> Tensor.narrow ~dim:1 ~start:0 ~length:n

let create_spiked_matrix n lambda =
  Utils.assert_positive_float "lambda" lambda;
  let v = Tensor.ones [n; 1] ~dtype:Kind.Float |> Tensor.div_scalar (Tensor.float_scalar (sqrt (float_of_int n))) in
  Tensor.mm v (Tensor.transpose v ~dim0:0 ~dim1:1) |> Tensor.mul_scalar lambda

let create_goe_matrix n =
  Utils.assert_positive_float "n" (float_of_int n);
  let lower_triangular = Tensor.randn [n; n] ~dtype:Kind.Float in
  let symmetric = Tensor.(lower_triangular + transpose lower_triangular ~dim0:0 ~dim1:1) in
  Tensor.(symmetric / (float_scalar (sqrt (float_of_int (2 * n)))))

let create_wishart_matrix n p =
  Utils.assert_positive_float "n" (float_of_int n);
  Utils.assert_positive_float "p" (float_of_int p);
  let x = Tensor.randn [n; p] ~dtype:Kind.Float in
  let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
  Tensor.matmul xt x
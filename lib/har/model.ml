open Base
open Torch
open Lwt
open Lwt.Infix
open Har_kernel

module type HAR = sig
  type t

  val create : ?batch_size:int -> ?early_stopping:float -> float list -> order -> t
  val fit : t -> Tensor.t -> Tensor.t -> (t, string) result
  val predict : t -> Tensor.t -> (Tensor.t, string) result
  val cross_validate : t -> Tensor.t -> Tensor.t -> int -> (t, string) result Lwt.t
  val mse : t -> Tensor.t -> Tensor.t -> (float, string) result
  val r2_score : t -> Tensor.t -> Tensor.t -> (float, string) result
  val get_params : t -> (float option * order, string) result
  val save_model : t -> string -> (unit, string) result
  val load_model : string -> (t, string) result
end

module Make (P : sig val p : int end) = struct
  type t = {
    lambdas: float list;
    order: order;
    n: int;
    x: Tensor.t;
    y: Tensor.t;
    k: (int -> int -> float) option;
    alpha: Tensor.t option;
    lambda: float option;
    batch_size: int option;
    early_stopping: float option;
  }

  let create ?(batch_size=None) ?(early_stopping=None) lambdas order =
    { lambdas; order; n = 0; x = Tensor.zeros [0; P.p]; y = Tensor.zeros [0]; 
      k = None; alpha = None; lambda = None; 
      batch_size = batch_size; early_stopping = early_stopping }

  let fit_with_lambda t lambda =
    try
      let n = t.n in
      let k = Option.value_exn t.k in
      let identity = Tensor.eye n in
      let k_tensor = Tensor.of_fun2 [n; n] k in
      let k_reg = Tensor.add k_tensor (Tensor.mul_scalar identity lambda) in
      let alpha = Tensor.matmul (Tensor.inverse k_reg) t.y in
      Ok { t with alpha = Some alpha; lambda = Some lambda }
    with
    | _ -> Error "Failed to fit model with given lambda"

  let fit t x y =
    let n = Tensor.shape x |> List.hd_exn in
    let k = kernel_function t.order x in
    let t' = { t with n; x; y; k = Some k } in
    fit_with_lambda t' (List.hd_exn t.lambdas)

  let predict_batch t x' start_idx end_idx =
    match t.alpha with
    | None -> Error "Model not fitted"
    | Some alpha ->
        try
          let batch_size = end_idx - start_idx in
          let k = Tensor.of_fun2 [batch_size; t.n] (fun i j ->
            let x'i = Tensor.select x' 0 (i + start_idx) in
            let xj = Tensor.select t.x 0 j in
            Tensor.to_float0_exn (kernel t.order x'i xj)
          ) in
          Ok (Tensor.matmul k (Tensor.unsqueeze alpha 1) |> Tensor.squeeze_last)
        with
        | _ -> Error "Failed to predict batch"

  let predict t x' =
    match t.batch_size with
    | None -> predict_batch t x' 0 (Tensor.shape x' |> List.hd_exn)
    | Some batch_size ->
        try
          let n = Tensor.shape x' |> List.hd_exn in
          let num_batches = (n + batch_size - 1) / batch_size in
          let predictions = ref [] in
          for i = 0 to num_batches - 1 do
            let start_idx = i * batch_size in
            let end_idx = Int.min (start_idx + batch_size) n in
            match predict_batch t x' start_idx end_idx with
            | Ok batch_pred -> predictions := batch_pred :: !predictions
            | Error msg -> raise (Invalid_argument msg)
          done;
          Ok (Tensor.cat (List.rev !predictions) ~dim:0)
        with
        | Invalid_argument msg -> Error msg
        | _ -> Error "Failed to predict"

  let leave_one_out_error t lambda =
    try
      let n = t.n in
      let k = Option.value_exn t.k in
      let identity = Tensor.eye n in
      let k_tensor = Tensor.of_fun2 [n; n] k in
      let k_reg = Tensor.add k_tensor (Tensor.mul_scalar identity lambda) in
      let k_inv = Tensor.inverse k_reg in
      let h = Tensor.diagonal (Tensor.matmul k_tensor k_inv) in
      let y_pred = Tensor.matmul k_tensor (Tensor.matmul k_inv t.y) in
      let errors = Tensor.div (Tensor.sub t.y y_pred) (Tensor.sub (Tensor.ones [n]) h) in
      Ok (Tensor.mean (Tensor.square errors))
    with
    | _ -> Error "Failed to compute leave-one-out error"

  let cross_validate_lambda t lambda =
    match leave_one_out_error t lambda with
    | Ok error -> Lwt.return (Ok (lambda, Tensor.to_float0_exn error))
    | Error msg -> Lwt.return (Error msg)

  let cross_validate t x y folds =
    let n = Tensor.shape x |> List.hd_exn in
    let k = kernel_function t.order x in
    let t' = { t with n; x; y; k = Some k } in
    let%lwt results = Lwt_list.map_p (cross_validate_lambda t') t.lambdas in
    match Lwt_list.fold_left_s (fun acc result ->
      match acc, result with
      | Ok (best_lambda, best_error), Ok (lambda, error) ->
          if Float.(error < best_error) then
            (match t.early_stopping with
             | Some threshold when Float.((best_error - error) / best_error < threshold) ->
                 Lwt.return (Error "Early stopping criterion met")
             | _ -> Lwt.return (Ok (lambda, error)))
          else Lwt.return (Ok (best_lambda, best_error))
      | Error msg, _ -> Lwt.return (Error msg)
      | _, Error msg -> Lwt.return (Error msg)
    ) (Ok (Float.infinity, Float.infinity)) results with
    | Ok (best_lambda, _) -> Lwt.return (fit_with_lambda t' best_lambda)
    | Error msg -> Lwt.return (Error msg)

  let mse t x y =
    match predict t x with
    | Ok y_pred -> Ok (Tensor.mse y y_pred |> Tensor.to_float0_exn)
    | Error msg -> Error msg

  let r2_score t x y =
    match predict t x with
    | Ok y_pred ->
        let ss_tot = Tensor.sum (Tensor.square (Tensor.sub y (Tensor.mean y))) in
        let ss_res = Tensor.sum (Tensor.square (Tensor.sub y y_pred)) in
        Ok (Tensor.sub (Tensor.of_float 1.) (Tensor.div ss_res ss_tot) |> Tensor.to_float0_exn)
    | Error msg -> Error msg

  let get_params t = Ok (t.lambda, t.order)

  let save_model t filename =
    try
      let oc = Caml.open_out filename in
      Caml.Marshal.to_channel oc t [];
      Caml.close_out oc;
      Ok ()
    with
    | _ -> Error "Failed to save model"

  let load_model filename =
    try
      let ic = Caml.open_in filename in
      let t : t = Caml.Marshal.from_channel ic in
      Caml.close_in ic;
      Ok t
    with
    | _ -> Error "Failed to load model"
end
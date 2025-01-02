open Torch

module GBM = struct
  type tree = {
    feature: int;
    threshold: float;
    left: tree option;
    right: tree option;
    value: float;
  }

  type model = {
    trees: tree list;
    learning_rate: float;
  }

  let rec predict_tree tree features =
    match tree.left, tree.right with
    | None, None -> tree.value
    | Some left, Some right ->
        if features.{tree.feature} <= tree.threshold then
          predict_tree left features
        else
          predict_tree right features
    | _ -> failwith "Invalid tree structure"

  let predict model features =
    List.fold_left (fun acc tree ->
      acc +. model.learning_rate *. predict_tree tree features
    ) 0. model.trees

  let train features targets num_trees max_depth learning_rate =
    { trees = []; learning_rate }
end

module NN = struct
  type model = {
    layers: Tensor.t list;
    biases: Tensor.t list;
  }

  let forward model x =
    List.fold_left2 (fun acc layer bias ->
      Tensor.(mm acc layer + bias) |> Tensor.relu
    ) x model.layers model.biases

  let train x y hidden_sizes learning_rate num_epochs =
    { layers = []; biases = [] }

  let predict model x =
    forward model x
end

module GPR = struct
  type model = {
    kernel: Tensor.t -> Tensor.t -> Tensor.t;
    x_train: Tensor.t;
    y_train: Tensor.t;
    noise: float;
  }

  let rbf_kernel length_scale =
    fun x y ->
      let diff = Tensor.sub x y in
      Tensor.exp (Tensor.div (Tensor.neg (Tensor.pow diff 2.)) (2. *. length_scale *. length_scale))

  let predict model x =
    let k_xx = model.kernel model.x_train model.x_train in
    let k_xx_inv = Tensor.inverse (Tensor.add k_xx (Tensor.eye (Tensor.shape2_exn k_xx |> fst) ~scalar:(model.noise ** 2.))) in
    let k_x = model.kernel model.x_train x in
    let mean = Tensor.mm (Tensor.mm (Tensor.transpose k_x ~dim0:0 ~dim1:1) k_xx_inv) model.y_train in
    let var = Tensor.sub (model.kernel x x) (Tensor.mm (Tensor.mm (Tensor.transpose k_x ~dim0:0 ~dim1:1) k_xx_inv) k_x) in
    (mean, var)

  let train x y length_scale noise =
    { kernel = rbf_kernel length_scale; x_train = x; y_train = y; noise }
end
open Base
open Torch
open Types

module Model = struct
  type t = {
    model: ModelType.t;
    loss: LossType.t;
    optimizer: Optimizer.t;
  }

  let create model loss learning_rate =
    let optimizer = Optimizer.adam (model.parameters()) ~learning_rate in
    { model; loss; optimizer }

  let predict t x =
    t.model.forward x

  let evaluate t x y =
    let y_pred = predict t x in
    Tensor.to_float0_exn (t.loss y_pred y)

  let train t data ~epochs =
    let x, y = data in
    for _ = 1 to epochs do
      let y_pred = predict t x in
      let loss = t.loss y_pred y in
      Optimizer.zero_grad t.optimizer;
      Tensor.backward loss;
      Optimizer.step t.optimizer
    done;
    t
end

module LinearModel = struct
  let create input_dim output_dim =
    let w = Tensor.randn [input_dim; output_dim] in
    let b = Tensor.zeros [output_dim] in
    let parameters = [w; b] in
    {
      ModelType.forward = (fun x -> Tensor.(x @@ w + b));
      parameters = (fun () -> parameters);
    }
end

module MLP = struct
  let create input_dim hidden_dim output_dim =
    let w1 = Tensor.randn [input_dim; hidden_dim] in
    let b1 = Tensor.zeros [hidden_dim] in
    let w2 = Tensor.randn [hidden_dim; output_dim] in
    let b2 = Tensor.zeros [output_dim] in
    let parameters = [w1; b1; w2; b2] in
    {
      ModelType.forward = (fun x ->
        let h = Tensor.(x @@ w1 + b1) |> Tensor.relu in
        Tensor.(h @@ w2 + b2)
      );
      parameters = (fun () -> parameters);
    }
end

module RandomForest = struct
  type tree = Leaf of float | Node of { feature: int; threshold: float; left: tree; right: tree }

  let rec predict_tree tree x =
    match tree with
    | Leaf value -> value
    | Node { feature; threshold; left; right } ->
      if Tensor.get x [feature] <= threshold then
        predict_tree left x
      else
        predict_tree right x

  let create n_trees max_depth n_features =
    let trees = Array.init n_trees ~f:(fun _ ->
      let rec build_tree depth =
        if depth >= max_depth then
          Leaf (Random.float 1.0)
        else
          Node {
            feature = Random.int n_features;
            threshold = Random.float 1.0;
            left = build_tree (depth + 1);
            right = build_tree (depth + 1);
          }
      in
      build_tree 0
    ) in
    
    {
      ModelType.forward = (fun x ->
        let predictions = Array.map trees ~f:(fun tree -> predict_tree tree x) in
        Tensor.of_float1 predictions |> Tensor.mean
      );
      parameters = (fun () -> []); 
    }
end
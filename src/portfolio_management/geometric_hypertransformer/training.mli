type optimizer_config = {
  learning_rate: float;
  momentum: float;
  weight_decay: float;
}

type training_state = {
  model: (Tensor.t, Tensor.t) GeometricHypertransformer.t;
  optimizer: optimizer_config;
  iteration: int;
  best_loss: float;
}

val create_training_state:
  model:(Tensor.t, Tensor.t) GeometricHypertransformer.t ->
  optimizer_config ->
  training_state

val train_epoch:
  training_state ->
  (Tensor.t * Tensor.t) list ->
  training_state * float

val evaluate:
  training_state ->
  (Tensor.t * Tensor.t) list ->
  float

val riemannian_step:
  training_state ->
  Tensor.t ->  (* gradient *)
  InformationGeometry.statistical_manifold memory
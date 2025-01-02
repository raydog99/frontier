open Torch

type encoder_type = Discrete | Continuous

type t = {
  encoder_type: encoder_type;
  embedding: nn option;
  fc_network: nn option;
  num_classes: int option;
  condition_dim: int;
}

let create_discrete ~num_classes ~condition_dim =
  let embedding = Nn.embedding ~num_embeddings:num_classes ~embedding_dim:condition_dim () in
  { encoder_type = Discrete; embedding = Some embedding; fc_network = None; num_classes = Some num_classes; condition_dim }

let create_continuous ~input_dim ~condition_dim =
  let fc_network = Nn.sequential [
    Nn.linear ~in_features:input_dim ~out_features:64 ();
    Nn.relu ();
    Nn.linear ~in_features:64 ~out_features:condition_dim ();
  ] in
  { encoder_type = Continuous; embedding = None; fc_network = Some fc_network; num_classes = None; condition_dim }

let encode t condition =
  match t.encoder_type with
  | Discrete ->
      let class_index = Tensor.to_int0_exn condition in
      Nn.forward (Option.get t.embedding) (Tensor.of_int0 class_index)
  | Continuous ->
      Nn.forward (Option.get t.fc_network) condition
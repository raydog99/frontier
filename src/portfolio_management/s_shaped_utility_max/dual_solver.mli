open Torch

type dual_params = {
  model_params: Model.model_params;
  utility: Utility.Utility;
  network: Neural_network.Network;
}

val create_dual_params : model_params:Model.model_params -> utility:Utility.Utility -> network_architecture:Neural_network.NetworkArchitecture -> dual_params
val dual_loss : dual_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val train_dual : dual_params -> Optimizer.t -> Data_handling.data_loader -> int -> unit
val compute_primal_from_dual : dual_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
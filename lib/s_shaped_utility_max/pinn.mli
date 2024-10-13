open Torch

type pinn_params = {
  model_params: Model.model_params;
  utility: Utility.Utility;
  network: Neural_network.Network;
}

val create_pinn_params : model_params:Model.model_params -> utility:Utility.Utility -> network_architecture:Neural_network.NetworkArchitecture -> pinn_params
val pinn_loss : pinn_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val train_pinn : pinn_params -> Optimizer.t -> Data_handling.data_loader -> int -> unit
val evaluate_pinn : pinn_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val train_pinn_with_concavification : pinn_params -> Optimizer.t -> Data_handling.data_loader -> int -> unit
open Torch

module type MODEL_CONFIG = sig
  val input_dim: int
  val qas_space: (module QASSpace.QAS_SPACE)
  val holder_params: HolderApproximation.params
  val approximation_error: TransformerBounds.approximation_error
end

module MakeModel(Config: MODEL_CONFIG) = struct
  let create_model () =
    (* Create base geometric transformer *)
    let base_transformer = 
      GeometricTransformer.create
        ~input_dim:Config.input_dim
        ~qas_space:Config.qas_space
        ~holder_params:Config.holder_params
        ~error:Config.approximation_error in

    (* Compute theoretical bounds *)
    let bounds = 
      TransformerBounds.compute_bounds
        ~holder_params:Config.holder_params
        ~error:Config.approximation_error in

    (* Create hypertransformer *)
    GeometricHypertransformer.create
      ~base_transformer
      ~memory_size:bounds.memory_bound.size
      ~bounds
end

(* Training utilities *)
module Training = struct
  type training_config = {
    learning_rate: float;
    batch_size: int;
    num_epochs: int;
    momentum: float;
  }

  let train_step model batch config =
    let open GeometricHypertransformer in
    
    let loss = ref 0. in
    let gradients = ref [] in
    
    List.iter (fun (input, target) ->
      (* Forward pass *)
      let output = forward model input in
      
      (* Compute loss *)
      let batch_loss = 
        let module QAS = (val model.qas_space) in
        QAS.distance output target
      in
      loss := !loss +. batch_loss;
      
      (* Backward pass *)
      let grad = Tensor.grad batch_loss in
      gradients := grad :: !gradients
    ) batch;
    
    (* Update parameters *)
    let avg_loss = !loss /. float_of_int (List.length batch) in
    let avg_grad = 
      List.fold_left Tensor.add 
        (List.hd !gradients) 
        (List.tl !gradients) in
    
    let updated_memory = {
      model.memory with
      parameters = 
        Tensor.sub model.memory.parameters 
          (Tensor.mul_scalar avg_grad config.learning_rate);
      velocity =
        Tensor.add
          (Tensor.mul_scalar model.memory.velocity config.momentum)
          (Tensor.mul_scalar avg_grad config.learning_rate);
    } in
    
    {model with memory = updated_memory}, avg_loss
end
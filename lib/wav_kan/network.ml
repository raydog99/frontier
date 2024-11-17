open Torch

module WavKANNetwork = struct

  type config = {
    input_dim: int;
    output_dim: int;
    hidden_dims: int list;
    activation_learnable: bool;
    use_skip_connections: bool;
    batch_norm_layers: bool;
  }

  module Layer = struct
    type t = {
      weights: Tensor.t;
      bias: Tensor.t;
      wavelet_fn: float -> float;
      learnable_params: Tensor.t list;
      batch_norm: BatchNorm.t option;
      skip_connection: Tensor.t option;
    }

    let create in_dim out_dim config =
      let weights = Tensor.randn [out_dim; in_dim] ~requires_grad:true in
      let bias = Tensor.zeros [out_dim] ~requires_grad:true in
      let wavelet_fn = Function.ricker in
      let learnable_params = 
        if config.activation_learnable then
          [Tensor.ones [out_dim] ~requires_grad:true;  (* scale *)
           Tensor.zeros [out_dim] ~requires_grad:true] (* translation *)
        else []
      in
      let batch_norm = 
        if config.batch_norm_layers then
          Some (BatchNorm.create out_dim ())
        else None
      in
      let skip_connection =
        if config.use_skip_connections && in_dim = out_dim then
          Some (Tensor.eye in_dim)
        else None
      in
      { weights; bias; wavelet_fn; learnable_params; 
        batch_norm; skip_connection }

    let forward layer input ~training =
      let linear = Tensor.mm input layer.weights +! layer.bias in
      let activated = 
        if List.length layer.learnable_params > 0 then
          let scale = List.nth layer.learnable_params 0 in
          let translation = List.nth layer.learnable_params 1 in
          Tensor.map (fun x ->
            let x' = (Tensor.float_value x - Tensor.float_value translation) /.
                    Tensor.float_value scale in
            Function.to_tensor_op layer.wavelet_fn x'
          ) linear
        else
          Tensor.map (Function.to_tensor_op layer.wavelet_fn) linear
      in
      let normalized = match layer.batch_norm with
        | Some bn -> BatchNorm.forward bn activated ~training
        | None -> activated
      in
      match layer.skip_connection with
        | Some skip -> Tensor.(normalized + (input * skip))
        | None -> normalized
  end

  type t = {
    layers: Layer.t list;
    config: config;
    mutable training: bool;
  }

  let create config =
    let layer_dims = config.input_dim :: 
                    config.hidden_dims @ 
                    [config.output_dim] in
    let layers = List.map2 
      (fun in_dim out_dim -> Layer.create in_dim out_dim config)
      (List.init (List.length layer_dims - 1) (fun i -> List.nth layer_dims i))
      (List.tl layer_dims)
    in
    { layers; config; training = true }

  let forward network input =
    List.fold_left (fun acc layer ->
      Layer.forward layer acc ~training:network.training
    ) input network.layers

  let parameters network =
    List.concat_map (fun layer ->
      layer.Layer.weights :: 
      layer.Layer.bias :: 
      layer.Layer.learnable_params @
      (match layer.Layer.batch_norm with
       | Some bn -> [bn.BatchNorm.weight; bn.BatchNorm.bias]
       | None -> [])
    ) network.layers

  let train network mode =
    network.training <- mode
end
open Torch

module MultiHeadAttention = struct
  type t = {
    w_q: Nn.t;
    w_k: Nn.t;
    w_v: Nn.t;
    w_o: Nn.t;
    num_heads: int;
    head_dim: int;
  }

  let create dim num_heads =
    let head_dim = dim / num_heads in
    {
      w_q = Nn.linear dim dim;
      w_k = Nn.linear dim dim;
      w_v = Nn.linear dim dim;
      w_o = Nn.linear dim dim;
      num_heads;
      head_dim;
    }

  let forward t x =
    let open Tensor in
    let batch_size = shape x |> List.hd in
    
    let reshape_heads x =
      reshape x [batch_size; -1; t.num_heads; t.head_dim]
      |> transpose ~dim0:1 ~dim1:2 in
    
    let q = Nn.forward t.w_q x |> reshape_heads in
    let k = Nn.forward t.w_k x |> reshape_heads in
    let v = Nn.forward t.w_v x |> reshape_heads in
    
    let attn_weights = 
      matmul q (transpose k ~dim0:(-2) ~dim1:(-1))
      |> div_scalar (sqrt (float_of_int t.head_dim))
      |> softmax ~dim:(-1) in
    
    let output = 
      matmul attn_weights v
      |> transpose ~dim0:1 ~dim1:2
      |> reshape [-1; dim] in
    
    Nn.forward t.w_o output
end

module TransformerBlock = struct
  type t = {
    attention: MultiHeadAttention.t;
    norm1: Nn.t;
    norm2: Nn.t;
    ffn: Nn.t;
  }

  let create dim num_heads =
    {
      attention = MultiHeadAttention.create dim num_heads;
      norm1 = Nn.layer_norm [dim];
      norm2 = Nn.layer_norm [dim];
      ffn = Nn.sequential [
        Nn.linear dim (4 * dim);
        Nn.gelu ();
        Nn.linear (4 * dim) dim;
      ];
    }

  let forward t x =
    let open Tensor in
    let attn_out = MultiHeadAttention.forward t.attention x in
    let x = add x attn_out |> Nn.forward t.norm1 in
    let ffn_out = Nn.forward t.ffn x in
    add x ffn_out |> Nn.forward t.norm2
end

module FlowNetwork = struct
  type t = {
    embedding: Nn.t;
    transformer_blocks: TransformerBlock.t list;
    output: Nn.t;
  }

  let create ~input_dim ~num_layers ~num_heads =
    let embedding = Nn.linear input_dim input_dim in
    let transformer_blocks = 
      List.init num_layers (fun _ -> 
        TransformerBlock.create input_dim num_heads) in
    let output = Nn.linear input_dim input_dim in
    { embedding; transformer_blocks; output }

  let forward t x =
    let x = Nn.forward t.embedding x in
    let x = List.fold_left 
      (fun acc block -> TransformerBlock.forward block acc)
      x t.transformer_blocks in
    Nn.forward t.output x
end

module VAE = struct
  type t = {
    encoder: TransformerBlock.t list;
    mu_proj: Nn.t;
    logvar_proj: Nn.t;
    decoder: TransformerBlock.t list;
    output_proj: Nn.t;
    latent_dim: int;
  }

  let create ~input_dim ~latent_dim ~num_layers ~num_heads =
    {
      encoder = List.init num_layers (fun _ -> 
        TransformerBlock.create input_dim num_heads);
      mu_proj = Nn.linear input_dim latent_dim;
      logvar_proj = Nn.linear input_dim latent_dim;
      decoder = List.init num_layers (fun _ -> 
        TransformerBlock.create input_dim num_heads);
      output_proj = Nn.linear input_dim input_dim;
      latent_dim;
    }

  let reparameterize mu logvar =
    let open Tensor in
    let std = exp (mul_scalar logvar 0.5) in
    let eps = randn_like std in
    add mu (mul eps std)

  let forward t x =
    let open Tensor in
    let encoded = List.fold_left 
      (fun acc block -> TransformerBlock.forward block acc) 
      x t.encoder in
    let mu = Nn.forward t.mu_proj encoded in
    let logvar = Nn.forward t.logvar_proj encoded in
    let z = reparameterize mu logvar in
    let decoded = List.fold_left 
      (fun acc block -> TransformerBlock.forward block acc) 
      z t.decoder in
    let recon = Nn.forward t.output_proj decoded in
    recon, mu, logvar

  let loss ~recon ~input ~mu ~logvar =
    let open Tensor in
    let recon_loss = mse_loss recon input in
    let kld_loss = 
      mul (add (add (mul mu mu) (exp logvar)) (neg_scalar logvar)) 
        (scalar (-0.5))
      |> sum ~dim:[1]
      |> mean in
    add recon_loss kld_loss
end
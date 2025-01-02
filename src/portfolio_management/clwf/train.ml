open Torch

let train_vae ~vae ~data ~config =
  let optimizer = Optimizer.adam (Nn.parameters vae) ~lr:config.learning_rate in
  
  for epoch = 1 to config.max_epochs do
    let loader = DataLoader.create ~data ~batch_size:config.batch_size ~shuffle:true in
    
    let rec train_epoch () =
      match DataLoader.next_batch loader with
      | None -> ()
      | Some batch ->
          let recon, mu, logvar = VAE.forward vae batch.data in
          let loss = VAE.loss ~recon ~input:batch.data ~mu ~logvar in
          
          Optimizer.zero_grad optimizer;
          backward loss;
          Optimizer.step optimizer;
          train_epoch ()
    in
    train_epoch ()
  done

let train_flow ~flow ~vae ~config ~data =
  let optimizer = Optimizer.adam (Nn.parameters flow) ~lr:config.learning_rate in
  
  for epoch = 1 to config.max_epochs do
    let loader = DataLoader.create ~data ~batch_size:config.batch_size ~shuffle:true in
    
    let rec train_epoch () =
      match DataLoader.next_batch loader with
      | None -> ()
      | Some batch ->
          (* Sample source and target *)
          let source = Tensor.randn [config.batch_size; data.dimensions; data.sequence_length] 
            |> Tensor.mul_scalar config.initial_noise_std in
          let target = batch.data in
          
          (* Sample OT maps *)
          let source, target = Transport.sample_ot_maps ~source ~target 
            ~epsilon:0.01 ~num_iters:10 in
          
          (* Sample time and compute loss *)
          let t = Random.float config.terminal_time in
          let conditional = target in (* Use target as conditional *)
          
          let loss = Flow.flow_matching_loss ~model:flow ~source ~target 
            ~conditional ~t ~config in
          
          Optimizer.zero_grad optimizer;
          backward loss;
          Optimizer.step optimizer;
          train_epoch ()
    in
    train_epoch ()
  done
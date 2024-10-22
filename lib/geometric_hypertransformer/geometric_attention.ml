open Torch

type t = {
  num_heads: int;
  head_dim: int;
  qas_space: (module QASSpace.QAS_SPACE);
  temperature: float;
  geometric_structure: InformationGeometry.geometric_structure;
}

let create ~num_heads ~head_dim ~qas_space =
  let module QAS = (val qas_space) in
  {
    num_heads;
    head_dim;
    qas_space;
    temperature = 1. /. sqrt (float_of_int head_dim);
    geometric_structure = InformationGeometry.create_statistical_manifold head_dim;
  }

let forward t x ?mask =
  let module QAS = (val t.qas_space) in
  
  (* Split into multiple heads *)
  let batch_size = Tensor.size x 0 in
  let seq_len = Tensor.size x 1 in
  
  let split_heads x =
    Tensor.reshape x [batch_size; seq_len; t.num_heads; t.head_dim]
    |> fun x -> Tensor.transpose x ~dim0:1 ~dim1:2
  in
  
  (* Compute Q, K, V projections with geometric structure *)
  let q = split_heads x in
  let k = split_heads x in
  let v = split_heads x in

  (* Compute geometric attention scores *)
  let scores = Array.init t.num_heads (fun h ->
    let q_h = Tensor.select q ~dim:1 ~index:h in
    let k_h = Tensor.select k ~dim:1 ~index:h in
    
    let scores = Tensor.zeros [batch_size; seq_len; seq_len] in
    for i = 0 to seq_len - 1 do
      for j = 0 to seq_len - 1 do
        let qi = Tensor.select q_h ~dim:1 ~index:i in
        let kj = Tensor.select k_h ~dim:1 ~index:j in
        
        (* Use Riemannian metric for attention *)
        let metric = t.geometric_structure.metric qi kj in
        let dist = Tensor.float_value metric in
        let score = -.(dist /. t.temperature) in
        
        Tensor.set_ scores [|i; j|] score
      done
    done;
    scores
  ) |> fun x -> Tensor.stack (Array.to_list x) ~dim:1 in
  
  (* Apply mask if provided *)
  let masked_scores = match mask with
    | Some m -> Tensor.masked_fill scores m Float.neg_infinity
    | None -> scores
  in

  (* Apply attention with parallel transport *)
  let weights = Tensor.softmax masked_scores ~dim:(-1) in
  
  (* Transport values along geodesics *)
  let transported_values = Array.init t.num_heads (fun h ->
    let v_h = Tensor.select v ~dim:1 ~index:h in
    let w_h = Tensor.select weights ~dim:1 ~index:h in
    
    let result = Tensor.zeros_like v_h in
    for i = 0 to seq_len - 1 do
      let base_point = Tensor.select q ~dim:1 ~index:i in
      
      (* Parallel transport along geodesics *)
      let transported = Array.init seq_len (fun j ->
        let target = Tensor.select k ~dim:1 ~index:j in
        let value = Tensor.select v_h ~dim:1 ~index:j in
        let weight = Tensor.get w_h [|i; j|] in
        
        let transported_value = 
          InformationGeometry.parallel_transport
            t.geometric_structure base_point target value in
        Tensor.mul_scalar transported_value weight
      ) |> Array.fold_left Tensor.add (Tensor.zeros_like v_h) in
      
      Tensor.copy_ result ~src:transported ~dst_dim:1 ~dst_idx:i
    done;
    result
  ) |> fun x -> Tensor.stack (Array.to_list x) ~dim:1 in

  (* Final mixing in QAS space *)
  Array.init batch_size (fun b ->
    Array.init t.num_heads (fun h ->
      let values = Tensor.select transported_values ~dim:0 ~index:b
                  |> fun x -> Tensor.select x ~dim:0 ~index:h in
      let weights = Tensor.select weights ~dim:0 ~index:b
                   |> fun x -> Tensor.select x ~dim:0 ~index:h in
      
      QAS.mix QAS.create_mixing 
        (Tensor.to_float1 weights) [|values|]
    ) |> fun x -> Tensor.stack (Array.to_list x) ~dim:0
  ) |> fun x -> Tensor.stack (Array.to_list x) ~dim:0
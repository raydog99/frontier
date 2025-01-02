open Torch

type point = {
  support: Tensor.t;      (* [batch x points x dim] *)
  weights: Tensor.t;      (* [batch x points] *)
  dimension: int;
}

type t = point

type mixing_function = {
  mix: float array -> point array -> point;
  constant: float;
  power: int;
}

let create_mixing constant power =
  let mix weights points =
    let batch_size = Tensor.size (Array.get points 0).support 0 in
    let dim = (Array.get points 0).dimension in
    
    (* Solve multi-marginal optimal transport *)
    let cost_matrix = Array.make_matrix 
      (Array.length points) 
      (Array.length points) 
      (Tensor.zeros [1]) in
    
    for i = 0 to Array.length points - 1 do
      for j = i + 1 to Array.length points - 1 do
        let cost = Tensor.cdist 
          points.(i).support points.(j).support ~p:2. in
        cost_matrix.(i).(j) <- cost;
        cost_matrix.(j).(i) <- cost
      done
    done;

    (* Sinkhorn algorithm for barycenter computation *)
    let rec sinkhorn_iterate potentials iter =
      if iter >= 100 then potentials
      else
        let new_potentials = Array.mapi (fun i pot ->
          let sum = Array.fold_lefti (fun acc j other_pot ->
            if i = j then acc
            else
              let scaled_cost = 
                Tensor.mul cost_matrix.(i).(j) other_pot in
              Tensor.add acc scaled_cost
          ) (Tensor.zeros [batch_size]) potentials in
          
          Tensor.div points.(i).weights sum
        ) potentials in
        
        sinkhorn_iterate new_potentials (iter + 1)
    in
    
    let initial_potentials = 
      Array.map (fun p -> p.weights) points in
    let optimal_potentials = 
      sinkhorn_iterate initial_potentials 0 in
    
    (* Compute barycenter *)
    let support = ref (Tensor.zeros [batch_size; 1; dim]) in
    let weights = ref (Tensor.zeros [batch_size; 1]) in
    
    Array.iteri (fun i point ->
      let transport = Array.init (Array.length points) (fun j ->
        if i = j then 
          Tensor.eye (Tensor.size point.support 1)
        else
          let cost = cost_matrix.(i).(j) in
          let scaled_cost = Tensor.mul cost 
            (Tensor.mul optimal_potentials.(i) 
               optimal_potentials.(j)) in
          Tensor.softmax scaled_cost ~dim:1
      ) in
      
      let combined_support = 
        Tensor.cat [!support; point.support] ~dim:1 in
      let combined_weights = Tensor.cat
        [!weights; 
         Tensor.mul_scalar point.weights weights.(i)] ~dim:1 in
      
      support := combined_support;
      weights := combined_weights
    ) points;
    
    {
      support = !support;
      weights = !weights;
      dimension = dim;
    }
  in
  {mix; constant; power}

let distance p1 p2 =
  let cost = Tensor.cdist p1.support p2.support ~p:2. in
  let transport = 
    AdaptedWasserstein.compute_optimal_transport
      p1.weights p2.weights cost in
  Tensor.sum (Tensor.mul cost transport) 
  |> Tensor.float_value

let metric_capacity epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let quantization_modulus epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let mix = fun m -> m.mix

let quantize q tensor =
  let batch_size = Tensor.size tensor 0 in
  let dim = Tensor.size tensor (-1) in
  
  let support = Tensor.reshape tensor [batch_size; q; dim] in
  let weights = 
    Tensor.ones [batch_size; q] 
    |> fun x -> Tensor.div_scalar x (float_of_int q) in
  
  {support; weights; dimension = dim}

let verify_simplicial mixing points =
  Array.for_all (fun point ->
    let mixed = mixing.mix 
      (Array.make (Array.length points) 
         (1. /. float_of_int (Array.length points))) 
      points in
    distance mixed point <= 
      mixing.constant *. 
      (Array.fold_left (fun acc p ->
         max acc (distance point p)
       ) 0. points) ** float_of_int mixing.power
  ) points

let compress point epsilon =
  let k = int_of_float (ceil (1. /. epsilon)) in
  let kmeans = KMeans.fit 
    ~n_clusters:k 
    ~points:point.support in
  
  let new_support = kmeans.centroids in
  let new_weights = kmeans.cluster_weights in
  
  {point with 
    support = new_support;
    weights = new_weights}

let decompress point = point
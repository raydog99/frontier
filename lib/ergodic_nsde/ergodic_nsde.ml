open Torch
open Base

type node_id = int
type edge = node_id * node_id

type graph = {
  nodes: node_id list;
  edges: edge list;
  adjacency: (node_id * node_id) list
}

type parameters = {
  alpha: float Tensor.t;  
  beta: float Tensor.t;   
  weights: float Tensor.t;
}

type state = {
  x: float Tensor.t;     
  time: float;           
}

type estimation_options = {
  delta: float;          
  threshold: float;      
  learning_rate: float;  
  max_iter: int;        
  tolerance: float;      
  sparsity_threshold: float;
  n_samples: int;       
  regularization: float;
  dt: float;           
}

let diag tensor =
  let size = Tensor.size tensor [0] in
  let d = Tensor.zeros [size; size] in
  for i = 0 to size - 1 do
    Tensor.set_elem2d d i i (Tensor.get_elem1d tensor i)
  done;
  d

let adjacency_matrix graph =
  let n = List.length graph.nodes in
  let adj = Tensor.zeros [n; n] in
  List.iter (fun (i,j) -> 
    Tensor.set_elem2d adj i j 1.0
  ) graph.adjacency;
  adj

let normalize_adjacency adj =
  let n = Tensor.size adj 0 in
  let degrees = Tensor.sum adj ~dim:[1] in
  let d_inv = Tensor.(pow degrees (-1.0)) in
  let d_mat = diag d_inv in
  Tensor.(matmul d_mat adj)

let laplacian graph =
  let adj = adjacency_matrix graph in
  let n = List.length graph.nodes in
  Tensor.(eye n - adj)

let drift state params graph =
  let x = state.x in
  let beta = params.beta in
  let lap = laplacian graph in
  let b = Tensor.(matmul lap beta) in
  Tensor.(matmul b x)

let diffusion state params =
  let x = state.x in
  let alpha = params.alpha in
  Tensor.(matmul (diag alpha) x)

let step state params graph dt =
  let x = state.x in
  let b = drift state params graph in
  let sigma = diffusion state params in
  let dw = Tensor.randn (Tensor.size x) in
  let dx = Tensor.(
    b * float_tensor dt + 
    matmul sigma dw * float_tensor (sqrt dt)
  ) in
  { x = Tensor.(x + dx);
    time = state.time +. dt }

let simulate init_state params graph n_steps dt =
  let rec loop state acc = function
    | 0 -> List.rev acc
    | n -> 
        let next_state = step state params graph dt in
        loop next_state (next_state :: acc) (n-1)
  in
  loop init_state [init_state] n_steps

let logdet_stable m =
  let u, s, _ = Tensor.svd m in
  Tensor.sum (Tensor.log s)

let score trajectory params graph dt =
  let states = List.map (fun s -> s.x) trajectory in
  let diffs = List.map2 Tensor.(-) (List.tl states) states in
  
  List.fold_left2 (fun acc dx x ->
    let state = {x; time=0.} in
    let c = Core.diffusion state params in
    let c_inv = Tensor.inverse c in
    let b = Core.drift state params graph in
    
    let d_alpha = Tensor.(
      -0.5 /. dt *. 
      (matmul c_inv dx - matmul b (float_tensor dt)) ** (of_int2 2) +
      -0.5 *. logdet_stable c
    ) in
    
    let d_beta = Tensor.(
      matmul c_inv (dx - matmul b (float_tensor dt)) * x
    ) in
    
    {
      alpha = Tensor.(acc.alpha + d_alpha);
      beta = Tensor.(acc.beta + d_beta);
      weights = acc.weights
    }
  ) 
    {
      alpha=Tensor.zeros_like params.alpha;
      beta=Tensor.zeros_like params.beta;
      weights=Tensor.zeros_like params.weights
    } 
    diffs states

let quasi_likelihood trajectory params graph dt =
  let states = List.map (fun s -> s.x) trajectory in
  let diffs = List.map2 Tensor.(-) (List.tl states) states in
  
  List.fold_left2 (fun acc dx x ->
    let state = {x; time=0.} in
    let c = Core.diffusion state params in
    let b = Core.drift state params graph in
    
    let term1 = Tensor.(
      -0.5 /. dt *. sum (
        matmul (inverse c) (dx - matmul b (float_tensor dt)) ** (of_int2 2)
      )
    ) in
    let term2 = -0.5 *. logdet_stable c in
    
    acc +. term1 +. term2
  ) 0. diffs states


let estimate_adjacency weights threshold =
  let n = Tensor.size weights 0 in
  let adj = Tensor.zeros [n; n] in
  
  for i = 0 to n-1 do
    for j = 0 to n-1 do
      if i <> j then
        let w_ij = Tensor.get_float2 weights i j in
        if abs_float w_ij > threshold then
          Tensor.set_float2 adj i j 1.0
    done
  done;
  adj

let reconstruct_graph params threshold =
  let adj = estimate_adjacency params.weights threshold in
  let n = Tensor.size adj 0 in
  
  let edges = ref [] in
  for i = 0 to n-1 do
    for j = 0 to n-1 do
      if Tensor.get_float2 adj i j > 0.5 then
        edges := (i,j) :: !edges
    done
  done;
  
  {
    nodes = List.init n (fun i -> i);
    edges = !edges;
    adjacency = !edges
  }

let is_subgraph estimated true_graph =
  List.for_all (fun edge ->
    List.mem edge true_graph.edges
  ) estimated.edges

let check_lyapunov_stability params graph =
  let n = List.length graph.nodes in
  
  (* Construct Jacobian matrix *)
  let jacobian x =
    let j = Tensor.zeros [n; n] in
    for i = 0 to n-1 do
      (* Self dynamics *)
      Tensor.set_float2 j i i (
        Tensor.get_float1 params.beta i
      );
      (* Network effects *)
      List.iter (fun (src,dst) ->
        if src = i then
          let w = Tensor.get_float2 params.weights i dst in
          Tensor.set_float2 j i dst w
      ) graph.adjacency
    done;
    j
  in
  
  (* Check equilibrium point *)
  let equilibrium = Tensor.zeros [n] in
  let j_eq = jacobian equilibrium in
  
  (* All eigenvalues should have negative real parts *)
  let eigenvals = Tensor.eigenvals j_eq in
  let real_parts = Tensor.real eigenvals in
  Tensor.(all (real_parts < float_tensor 0.0))

let verify_ergodicity trajectory params graph =
  let states = List.map (fun s -> s.x) trajectory in
  
  (* Test recurrence *)
  let is_recurrent =
    let max_norm = List.fold_left (fun acc x ->
      max acc (Tensor.get_float0 (Tensor.norm x))
    ) 0. states in
    max_norm < 1000.0
  in
  
  (* Test irreducibility *)
  let state_space_coverage =
    let bins = 10 in
    let visited = Array.make bins false in
    List.iter (fun x ->
      let norm = Tensor.get_float0 (Tensor.norm x) in
      let bin = min (bins-1) (int_of_float (norm *. float_of_int bins)) in
      visited.(bin) <- true
    ) states;
    Array.for_all (fun x -> x) visited
  in
  
  is_recurrent && state_space_coverage

let block_optimize trajectory graph params =
  let n = List.length graph.nodes in
  let block_size = min 100 (n/10) in
  
  (* Split parameters into blocks *)
  let create_blocks tensor size =
    let total_size = Tensor.size tensor 0 in
    List.init (1 + (total_size-1)/size) (fun i ->
      let start_idx = i * size in
      let end_idx = min (start_idx + size) total_size in
      Tensor.narrow tensor 0 start_idx (end_idx - start_idx)
    )
  in
  
  let alpha_blocks = create_blocks params.alpha block_size in
  let beta_blocks = create_blocks params.beta block_size in
  
  (* Optimize blocks *)
  let optimize_blocks current_params =
    let new_alpha = List.mapi (fun i block ->
      let block_params = {current_params with alpha = block} in
      let score = QuasiLikelihood.score trajectory block_params graph 0.01 in
      Tensor.(block - float_tensor 0.01 * score.alpha)
    ) alpha_blocks in
    
    let new_beta = List.mapi (fun i block ->
      let block_params = {current_params with beta = block} in
      let score = QuasiLikelihood.score trajectory block_params graph 0.01 in
      Tensor.(block - float_tensor 0.01 * score.beta)
    ) beta_blocks in
    
    {
      alpha = Tensor.cat new_alpha 0;
      beta = Tensor.cat new_beta 0;
      weights = current_params.weights
    }
  in
  
  let rec iterate params iter max_iter tol =
    if iter >= max_iter then params
    else
      let new_params = optimize_blocks params in
      let diff = Tensor.(
        mean (abs (new_params.alpha - params.alpha)) +
        mean (abs (new_params.beta - params.beta))
      ) in
      if Tensor.get_float0 diff < tol then new_params
      else iterate new_params (iter + 1) max_iter tol
  in
  
  iterate params 0 100 1e-6

let estimate_network_sde ?(known_graph=true) trajectory graph options =
  try
    if known_graph then
      Ok (estimate trajectory graph)
    else
      let adapted_params = AdaptiveLasso.estimate trajectory graph {
        lambda = options.regularization;
        delta = options.delta;
        threshold = options.threshold;
        learning_rate = options.learning_rate;
        max_iter = options.max_iter;
        tolerance = options.tolerance;
      } in
      
      let estimated_graph = reconstruct_graph 
        adapted_params options.threshold in
      
      if is_subgraph estimated_graph graph then
        Ok adapted_params
      else
        Error "Graph estimation failed consistency check"
  with
  | e -> Error (Exn.to_string e)

let verify_system trajectory params graph =
  try
    let stability_ok = check_lyapunov_stability params graph in
    let ergodicity_ok = verify_ergodicity trajectory params graph in
    
    let theoretical_check =
      let states = List.map (fun s -> s.x) trajectory in
      let max_norm = List.fold_left (fun acc x ->
        max acc (Tensor.get_float0 (Tensor.norm x))
      ) 0. states in
      max_norm < Float.infinity
    in
    
    stability_ok && ergodicity_ok && theoretical_check
  with
  | _ -> false

let optimize_params trajectory graph params learning_rate max_iter tol =
  let rec iterate current_params iter =
    if iter >= max_iter then current_params
    else
      let score = QuasiLikelihood.score trajectory current_params graph 0.01 in
      let new_params = {
        alpha = Tensor.(current_params.alpha - float_tensor learning_rate * score.alpha);
        beta = Tensor.(current_params.beta - float_tensor learning_rate * score.beta);
        weights = Tensor.(current_params.weights - float_tensor learning_rate * score.weights)
      } in
      
      let diff = Tensor.(
        sum (abs (new_params.alpha - current_params.alpha)) +
        sum (abs (new_params.beta - current_params.beta)) +
        sum (abs (new_params.weights - current_params.weights))
      ) in
      
      if Tensor.get_float0 diff < tol then new_params
      else iterate new_params (iter + 1)
  in
  iterate params 0

let estimate trajectory graph =
  let n = List.length graph.nodes in
  let init_params = {
    alpha = Tensor.ones [n];
    beta = Tensor.ones [n];
    weights = Tensor.ones [n; n]
  } in
  optimize_params trajectory graph init_params 0.01 1000 1e-6

module AdaptiveLasso = struct
  type lasso_params = {
    lambda: float;
    delta: float;
    threshold: float;
    learning_rate: float;
    max_iter: int;
    tolerance: float;
  }

  let compute_adaptive_weights init_params delta =
    let n = Tensor.size init_params.weights 0 in
    let weights = Tensor.zeros [n; n] in
    
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        if i <> j then
          let w_ij = Tensor.get_float2 init_params.weights i j in
          let adaptive_w = abs_float w_ij ** (-1. *. delta) in
          Tensor.set_float2 weights i j adaptive_w
      done
    done;
    weights

  let proximal_update params grad weights lambda lr =
    let soft_threshold x w =
      let sign = Tensor.sign x in
      Tensor.(sign * max (abs x - float_tensor (lambda *. w *. lr)) (zeros_like x))
    in
    
    {
      alpha = Tensor.(params.alpha - lr * grad.alpha);
      beta = soft_threshold 
        Tensor.(params.beta - lr * grad.beta)
        weights;
      weights = soft_threshold 
        Tensor.(params.weights - lr * grad.weights)
        weights
    }

  let estimate trajectory graph options =
    (* Initial non-regularized estimate *)
    let init_estimate = estimate trajectory graph in
    
    (* Compute adaptive weights *)
    let adaptive_weights = compute_adaptive_weights init_estimate options.delta in
    
    let rec optimize current_params iter prev_loss =
      if iter >= options.max_iter then current_params
      else
        let loss = QuasiLikelihood.quasi_likelihood 
          trajectory current_params graph options.dt in
        let penalty = Tensor.(sum (
          adaptive_weights * abs current_params.weights
        )) in
        let total_loss = loss +. options.lambda *. penalty in
        
        if abs_float (total_loss -. prev_loss) < options.tolerance 
        then current_params
        else
          let grad = QuasiLikelihood.score trajectory current_params graph options.dt in
          let new_params = proximal_update 
            current_params grad adaptive_weights options.lambda options.learning_rate in
          optimize new_params (iter + 1) total_loss
    in
    
    optimize init_estimate 0 Float.infinity
end
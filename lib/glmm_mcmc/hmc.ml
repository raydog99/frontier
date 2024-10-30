open Torch
open Types

type hmc_state = {
  position: model_params;
  momentum: model_params;
  log_prob: float;
}

let kinetic_energy state =
  Tensor.(
    add
      (add
        (sum (mul state.momentum.beta state.momentum.beta))
        (sum (mul state.momentum.u state.momentum.u)))
      (sum (mul state.momentum.lambda state.momentum.lambda))
  )

let leapfrog_step data state epsilon =
  let grad = Gradients.log_posterior data state.position state.prior_params in
  
  (* Half step for momentum *)
  let half_momentum = {
    beta = Tensor.(add state.momentum.beta 
      (mul grad.beta (float_tensor [epsilon /. 2.0])));
    u = Tensor.(add state.momentum.u 
      (mul grad.u (float_tensor [epsilon /. 2.0])));
    lambda = Tensor.(add state.momentum.lambda 
      (mul grad.lambda (float_tensor [epsilon /. 2.0])));
  } in
  
  (* Full step for position *)
  let new_position = {
    beta = Tensor.(add state.position.beta 
      (mul half_momentum.beta (float_tensor [epsilon])));
    u = Tensor.(add state.position.u 
      (mul half_momentum.u (float_tensor [epsilon])));
    lambda = Tensor.(add state.position.lambda 
      (mul half_momentum.lambda (float_tensor [epsilon])));
  } in
  
  (* Half step for momentum using new gradient *)
  let new_grad = Gradients.log_posterior data new_position state.prior_params in
  let new_momentum = {
    beta = Tensor.(add half_momentum.beta 
      (mul new_grad.beta (float_tensor [epsilon /. 2.0])));
    u = Tensor.(add half_momentum.u 
      (mul new_grad.u (float_tensor [epsilon /. 2.0])));
    lambda = Tensor.(add half_momentum.lambda 
      (mul new_grad.lambda (float_tensor [epsilon /. 2.0])));
  } in
  
  {position = new_position; momentum = new_momentum; log_prob = state.log_prob}

let step data state epsilon l =
  (* Generate new momentum *)
  let init_momentum = {
    beta = Tensor.randn (Tensor.size state.position.beta);
    u = Tensor.randn (Tensor.size state.position.u);
    lambda = Tensor.randn (Tensor.size state.position.lambda);
  } in
  
  let init_state = {state with momentum = init_momentum} in
  let init_energy = Float.neg (Tensor.float_value (kinetic_energy init_state)) +. state.log_prob in
  
  (* Perform leapfrog steps *)
  let final_state = List.fold_left
    (fun s _ -> leapfrog_step data s epsilon)
    init_state
    (List.init l (fun _ -> ())) in
    
  let final_energy = Float.neg (Tensor.float_value (kinetic_energy final_state)) +. final_state.log_prob in
  
  (* Accept/reject based on energy difference *)
  if Random.float 1.0 < Float.exp (final_energy -. init_energy) then
    final_state.position
  else
    state.position

module NUTS = struct
  type tree = {
    position: model_params;
    momentum: model_params;
    log_prob: float;
    energy: float;
    depth: int;
  }

  let build_tree data init_tree epsilon direction max_depth =
    let rec build depth state =
      if depth = max_depth then
        state
      else
        let next_state = leapfrog_step data 
          {position = state.position; 
           momentum = state.momentum; 
           log_prob = state.log_prob} 
          (float_of_int direction *. epsilon) in
          
        let next_energy = Float.neg (Tensor.float_value (kinetic_energy next_state)) 
                         +. next_state.log_prob in
        
        (* Check U-turn condition *)
        let momentum_sum = {
          beta = Tensor.(add state.momentum.beta next_state.momentum.beta);
          u = Tensor.(add state.momentum.u next_state.momentum.u);
          lambda = Tensor.(add state.momentum.lambda next_state.momentum.lambda);
        } in
        
        let position_diff = {
          beta = Tensor.(sub next_state.position.beta state.position.beta);
          u = Tensor.(sub next_state.position.u state.position.u);
          lambda = Tensor.(sub next_state.position.lambda state.position.lambda);
        } in
        
        let no_uturn = 
          Tensor.float_value (Tensor.dot momentum_sum.beta position_diff.beta) > 0.0 &&
          Tensor.float_value (Tensor.dot momentum_sum.u position_diff.u) > 0.0 in
          
        if no_uturn then
          build (depth + 1) {next_state with energy = next_energy; depth = depth + 1}
        else
          state
    in
    build 0 init_tree

  let nuts_step data state epsilon max_depth =
    let init_momentum = {
      beta = Tensor.randn (Tensor.size state.position.beta);
      u = Tensor.randn (Tensor.size state.position.u);
      lambda = Tensor.randn (Tensor.size state.position.lambda);
    } in
    
    let init_tree = {
      position = state.position;
      momentum = init_momentum;
      log_prob = state.log_prob;
      energy = Float.neg (Tensor.float_value (kinetic_energy state)) +. state.log_prob;
      depth = 0;
    } in
    
    (* Build trees in both directions *)
    let forward_tree = build_tree data init_tree epsilon 1 max_depth in
    let backward_tree = build_tree data init_tree epsilon (-1) max_depth in
    
    (* Choose tree with better energy *)
    if forward_tree.energy > backward_tree.energy then
      forward_tree.position
    else
      backward_tree.position
end
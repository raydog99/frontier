open Torch
open Optimal_execution

type config = {
  state_dim: int;
  action_dim: int;
  hidden_dim: int;
  num_layers: int;
  learning_rate: float;
  discount_factor: float;
  replay_buffer_size: int;
  batch_size: int;
  target_update_freq: int;
}

type state = {
  model: Module.t;
  target_model: Module.t;
  optimizer: Optimizer.t;
  replay_buffer: (Tensor.t * Tensor.t * float * Tensor.t * bool) list;
  step: int;
}

let create_config ~state_dim ~action_dim ~hidden_dim ~num_layers ~learning_rate ~discount_factor ~replay_buffer_size ~batch_size ~target_update_freq =
  { state_dim; action_dim; hidden_dim; num_layers; learning_rate; discount_factor; replay_buffer_size; batch_size; target_update_freq }

let create_network config =
  let open Layer in
  let layers = 
    sequential
      (linear ~in_dim:config.state_dim ~out_dim:config.hidden_dim () ::
       List.init (config.num_layers - 1) (fun _ -> linear ~in_dim:config.hidden_dim ~out_dim:config.hidden_dim ()) @
       [linear ~in_dim:config.hidden_dim ~out_dim:config.action_dim ()])
  in
  Module.create layers

let init config =
  let model = create_network config in
  let target_model = create_network config in
  Module.copy_ ~src:model ~dst:target_model;
  let optimizer = Optimizer.adam (Module.parameters model) ~lr:config.learning_rate in
  { model; target_model; optimizer; replay_buffer = []; step = 0 }

let select_action state input =
  Module.eval state.model;
  let action = Module.forward state.model input in
  Module.train state.model;
  action

let update state =
  if List.length state.replay_buffer < state.config.batch_size then
    state
  else
    let batch = List.take state.config.batch_size state.replay_buffer in
    let states, actions, rewards, next_states, dones = List.split5 batch in
    let states = Tensor.stack states ~dim:0 in
    let actions = Tensor.stack actions ~dim:0 in
    let rewards = Tensor.of_float1 (Array.of_list rewards) in
    let next_states = Tensor.stack next_states ~dim:0 in
    let dones = Tensor.of_int1 (Array.of_list (List.map (fun d -> if d then 1 else 0) dones)) in

    let q_values = Module.forward state.model states in
    let next_q_values = Module.forward state.target_model next_states in
    let max_next_q_values = Tensor.max next_q_values ~dim:1 |> fst in
    let expected_q_values = Tensor.add rewards (Tensor.mul (Tensor.sub (Tensor.ones_like dones) dones) (Tensor.mul (Tensor.of_float state.config.discount_factor) max_next_q_values)) in

    let loss = Tensor.mse_loss q_values expected_q_values in
    Optimizer.zero_grad state.optimizer;
    Tensor.backward loss;
    Optimizer.step state.optimizer;

    if state.step mod state.config.target_update_freq = 0 then
      Module.copy_ ~src:state.model ~dst:state.target_model;

    { state with step = state.step + 1 }

let train t num_episodes state =
  let rec train_episode episode state =
    if episode >= num_episodes then
      state
    else
      let initial_state = Optimal_execution.generate_market_scenarios t 1 |> Array.get 0 in
      let rec step current_state total_reward state =
        if Tensor.all (Tensor.eq current_state Tensor.zeros_like current_state) then
          (total_reward, state)
        else
          let action = select_action state current_state in
          let next_state, reward, done_ = Optimal_execution.step t current_state action in
          let new_replay_buffer = (current_state, action, reward, next_state, done_) :: state.replay_buffer in
          let new_replay_buffer = 
            if List.length new_replay_buffer > state.config.replay_buffer_size then
              List.take state.config.replay_buffer_size new_replay_buffer
            else
              new_replay_buffer
          in
          let updated_state = update { state with replay_buffer = new_replay_buffer } in
          if done_ then
            (total_reward +. reward, updated_state)
          else
            step next_state (total_reward +. reward) updated_state
      in
      let episode_reward, new_state = step initial_state 0. state in
      Printf.printf "Episode %d, Reward: %f\n" episode episode_reward;
      train_episode (episode + 1) new_state
  in
  train_episode 0 state
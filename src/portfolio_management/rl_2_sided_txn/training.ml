open Torch
open Rl_environment
open Sac_agent
open Benchmark_models
open Performance_evaluation

let train_and_evaluate agent env benchmark_models num_episodes eval_interval =
  let episode_rewards = ref [] in
  let episode_portfolio_values = ref [] in

  let rec train_loop episode total_reward =
    if episode > num_episodes then
      ()
    else
      let state = Rl_environment.get_state env in
      let action = Sac_agent.select_action agent state in
      let reward, next_portfolio_value, done = Rl_environment.step env action in
      let next_state = Rl_environment.get_state env in
      Sac_agent.update agent (state, action, reward, next_state, done);
      
      let total_reward = total_reward +. reward in
      
      if done then
        begin
          episode_rewards := total_reward :: !episode_rewards;
          episode_portfolio_values := next_portfolio_value :: !episode_portfolio_values;
          
          Printf.printf "Episode %d: Total Reward: %.4f, Final Portfolio Value: %.4f\n"
            episode total_reward next_portfolio_value;
          
          if episode mod eval_interval = 0 then
            begin
              let rl_performance = evaluate_agent agent env in
              List.iter (fun model ->
                let benchmark_performance = evaluate_benchmark model env in
                compare_performance rl_performance benchmark_performance
              ) benchmark_models;
              
              Sac_agent.save agent (Printf.sprintf "sac_agent_episode_%d.pt" episode);
            end;
          
          Rl_environment.reset env;
          train_loop (episode + 1) 0.0
        end
      else
        train_loop episode total_reward
  in
  train_loop 1 0.0;
  
  (List.rev !episode_rewards, List.rev !episode_portfolio_values)

let evaluate_agent agent env =
  let num_evaluation_episodes = 10 in
  let returns = ref [] in
  for _ = 1 to num_evaluation_episodes do
    Rl_environment.reset env;
    let rec eval_loop total_return =
      let state = Rl_environment.get_state env in
      let action = Sac_agent.select_action agent state in
      let reward, _, done = Rl_environment.step env action in
      let total_return = total_return +. reward in
      if done then total_return else eval_loop total_return
    in
    let episode_return = eval_loop 0.0 in
    returns := episode_return :: !returns
  done;
  let mean_return = List.fold_left (+.) 0.0 !returns /. float num_evaluation_episodes in
  let std_return = 
    let squared_diff = List.map (fun r -> (r -. mean_return) ** 2.0) !returns in
    sqrt (List.fold_left (+.) 0.0 squared_diff /. float num_evaluation_episodes)
  in
  { 
    return = mean_return; 
    risk = std_return; 
    sharpe_ratio = mean_return /. std_return;
    max_drawdown = calculate_max_drawdown !returns;
  }

let evaluate_benchmark model env =
  Rl_environment.reset env;
  let rec eval_loop total_return portfolio_values =
    let state = Rl_environment.get_state env in
    model.update state;
    let action = model.weights in
    let reward, portfolio_value, done = Rl_environment.step env action in
    let new_total_return = total_return +. reward in
    let new_portfolio_values = portfolio_value :: portfolio_values in
    if done then (new_total_return, new_portfolio_values)
    else eval_loop new_total_return new_portfolio_values
  in
  let total_return, portfolio_values = eval_loop 0.0 [] in
  let risk_free_rate = 0.02 in  (* Assume a 2% risk-free rate *)
  evaluate_performance (List.rev portfolio_values) risk_free_rate

let compare_performance rl_perf benchmark_perf =
  Printf.printf "RL Agent Performance:\n";
  Printf.printf "  Return: %.4f\n" rl_perf.return;
  Printf.printf "  Risk: %.4f\n" rl_perf.risk;
  Printf.printf "  Sharpe Ratio: %.4f\n" rl_perf.sharpe_ratio;
  Printf.printf "  Max Drawdown: %.4f\n" rl_perf.max_drawdown;
  
  Printf.printf "\nBenchmark Model Performance:\n";
  Printf.printf "  Return: %.4f\n" benchmark_perf.return;
  Printf.printf "  Risk: %.4f\n" benchmark_perf.risk;
  Printf.printf "  Sharpe Ratio: %.4f\n" benchmark_perf.sharpe_ratio;
  Printf.printf "  Max Drawdown: %.4f\n" benchmark_perf.max_drawdown;
  
  Printf.printf "\nPerformance Comparison:\n";
  Printf.printf "  Return Difference: %.4f\n" (rl_perf.return -. benchmark_perf.return);
  Printf.printf "  Risk Difference: %.4f\n" (rl_perf.risk -. benchmark_perf.risk);
  Printf.printf "  Sharpe Ratio Difference: %.4f\n" (rl_perf.sharpe_ratio -. benchmark_perf.sharpe_ratio);
  Printf.printf "  Max Drawdown Difference: %.4f\n" (rl_perf.max_drawdown -. benchmark_perf.max_drawdown)

let run_experiment data_file num_episodes eval_interval =
  let data = Data_preprocessing.load_and_preprocess_csv data_file in
  let num_assets = Tensor.size data |> List.hd in
  
  let env = Rl_environment.create data 0.001 0.02 in  (* 0.1% transaction fee, 2% interest rate *)
  let agent = Sac_agent.create (4 * num_assets) num_assets 0.001 in
  
  let mv_model = MeanVariance.create num_assets in
  let mad_model = MeanAbsoluteDeviation.create num_assets in
  let cvar_model = ConditionalValueAtRisk.create num_assets 0.95 in
  
  let benchmark_models = [mv_model; mad_model; cvar_model] in
  
  train_and_evaluate agent env benchmark_models num_episodes eval_interval
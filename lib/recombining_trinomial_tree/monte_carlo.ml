let european_option_price option num_simulations =
  let sum = ref 0. in
  for _ = 1 to num_simulations do
    let s_T = option.tree.s0 *. exp((option.tree.r -. 0.5 *. option.tree.sigma ** 2.) *. option.tree.t +. 
                                    option.tree.sigma *. sqrt(option.tree.t) *. Random.float 1.) in
    sum := !sum +. Option.payoff option s_T
  done;
  exp(-. option.tree.r *. option.tree.t) *. !sum /. float_of_int num_simulations

let american_option_price option num_simulations num_steps =
  let dt = option.tree.t /. float_of_int num_steps in
  let paths = Array.init num_simulations (fun _ ->
    Stochastic_process.simulate (Geometric_Brownian_Motion(option.tree.r, option.tree.sigma)) dt num_steps option.tree.s0
  ) in
  let exercise_values = Array.map (fun path ->
    Array.map (Option.payoff option) path
  ) paths in
  let continuation_values = Array.make_matrix num_simulations (num_steps + 1) 0. in
  for i = num_steps - 1 downto 0 do
    let x = Array.map (fun path -> path.(i)) paths in
    let y = Array.map (fun values -> values.(i+1)) continuation_values in
    let regression = Regression.least_squares x y in
    for j = 0 to num_simulations - 1 do
      continuation_values.(j).(i) <- Regression.predict regression paths.(j).(i)
    done
  done;
  let option_values = Array.mapi (fun i path ->
    let rec traverse j acc =
      if j = num_steps then acc
      else
        let exercise_value = exercise_values.(i).(j) in
        let continuation_value = continuation_values.(i).(j) in
        if exercise_value > continuation_value
        then exercise_value
        else traverse (j + 1) (acc *. exp(-. option.tree.r *. dt))
    in
    traverse 0 0.
  ) paths in
  Array.fold_left (+.) 0. option_values /. float_of_int num_simulations
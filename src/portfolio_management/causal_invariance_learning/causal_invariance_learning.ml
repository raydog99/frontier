open Torch

type environment = {
  id: int;                  (* Unique environment identifier *)
  covariates: Tensor.t;     (* Input features *)
  outcomes: Tensor.t;       (* Observed outcomes *)
  sample_size: int;         (* Number of samples *)
  intervention_strength: float; (* Strength of potential interventions *)
  metadata: (string * string) list; (* Flexible metadata *)
}

type intervention = 
  | NoIntervention
  | DirectIntervention of {
      target_variable: int;
      strength: float;
      distribution: [`Gaussian | `Uniform | `Custom]
    }
  | IndirectIntervention of {
      mediating_variables: int list;
      effect_size: float
    }

type model_parameters = {
  causal_coefficients: Tensor.t;  (* Causal effect coefficients *)
  noise_variance: float;           (* Model noise variance *)
  complexity_penalty: float;       (* Regularization strength *)
}

type causal_graph = {
  adjacency_matrix: Tensor.t;  (* Directed graph representation *)
  variable_names: string array;
  has_cycles: bool;
}

type statistical_summary = {
  mean: Tensor.t;
  variance: Tensor.t;
  skewness: Tensor.t;
  kurtosis: Tensor.t;
}

let compute_moments tensor =
  let mean = Tensor.mean tensor in
  let variance = Tensor.var tensor in
  let skewness = 
    let centered = Tensor.(sub tensor mean) in
    let third_moment = Tensor.mean (Tensor.pow centered (Tensor.scalar 3.0)) in
    let std = Tensor.sqrt variance in
    Tensor.(div third_moment (pow std (Tensor.scalar 3.0)))
  in
  let kurtosis =
    let centered = Tensor.(sub tensor mean) in
    let fourth_moment = Tensor.mean (Tensor.pow centered (Tensor.scalar 4.0)) in
    let std_fourth = Tensor.pow variance (Tensor.scalar 2.0) in
    Tensor.(div fourth_moment std_fourth) 
  in
  {
    mean;
    variance;
    skewness;
    kurtosis;
  }

let detect_distribution_shift env1 env2 =
  let summary1 = compute_moments env1.covariates in
  let summary2 = compute_moments env2.covariates in
  
  let kl_divergence = 
    Tensor.(
      sum (mul 
        (log (div summary1.variance summary2.variance)) 
        (scalar 0.5)
      )
    ) |> Tensor.float_value
  in
  {
    environment_1_id = env1.id;
    environment_2_id = env2.id;
    kl_divergence;
    is_significant_shift = kl_divergence > 1.0;
  }

let partial_correlations tensor =
  let n, p = Tensor.shape tensor in
  let corr_matrix = Tensor.zeros [p; p] in
  
  for i = 0 to p - 1 do
    for j = i + 1 to p - 1 do
      let xi = Tensor.select tensor ~dim:1 ~index:i in
      let xj = Tensor.select tensor ~dim:1 ~index:j in
      
      (* Partial correlation computation *)
      let partial_corr = 
        try 
          let residuals_xi = 
            Tensor.run_backward xi [xj] |> List.hd in
          let residuals_xj = 
            Tensor.run_backward xj [xi] |> List.hd in
          
          Tensor.(
            div 
              (mean (mul residuals_xi residuals_xj)) 
              (mul 
                (Tensor.std residuals_xi) 
                (Tensor.std residuals_xj)
              )
          ) |> Tensor.float_value
        with _ -> 0.0
      in
      
      Tensor.set corr_matrix [i; j] (Tensor.scalar partial_corr);
      Tensor.set corr_matrix [j; i] (Tensor.scalar partial_corr)
    done
  done;
  
  corr_matrix

type nonconvex_dro_model = {
  parameters: model_parameters;
  environments: environment list;
  causal_graph: causal_graph;
  training_history: float list;
}

type training_config = {
  max_iterations: int;
  learning_rate: float;
  regularization_strength: float;
  early_stopping_tolerance: float;
  intervention_strategy: intervention;
}

let default_training_config = {
  max_iterations = 1000;
  learning_rate = 0.01;
  regularization_strength = 1e-4;
  early_stopping_tolerance = 1e-6;
  intervention_strategy = NoIntervention;
}

let create_model environments ?initial_graph () =
  let dim = 
    match environments with
    | [] -> failwith "No environments provided"
    | hd :: _ -> Tensor.size hd.covariates |> List.hd
  in
  
  let default_graph = {
    adjacency_matrix = Tensor.zeros [dim; dim];
    variable_names = Array.init dim (fun i -> Printf.sprintf "x%d" i);
    has_cycles = false;
  } in
  
  let causal_graph = Option.value initial_graph ~default:default_graph in
  
  let parameters = {
    causal_coefficients = Tensor.randn [dim];
    noise_variance = 1.0;
    complexity_penalty = 1e-4;
  } in
  
  {
    parameters;
    environments;
    causal_graph;
    training_history = [];
  }

let compute_environment_loss model env =
  let pred = Tensor.mm env.covariates model.parameters.causal_coefficients in
  Tensor.(
    mean (pow (sub pred env.outcomes) (scalar 2.0)) +
    (scalar model.parameters.complexity_penalty *. 
     norm model.parameters.causal_coefficients)
  )

let compute_environment_weights model =
  let losses = List.map (compute_environment_loss model) model.environments in
  let max_loss = List.fold_left Tensor.maximum (List.hd losses) (List.tl losses) in
  
  List.map (fun loss ->
    if Tensor.(equal loss max_loss) then
      1.0 +. model.parameters.complexity_penalty
    else
      -.model.parameters.complexity_penalty
  ) losses

let compute_training_loss model =
  let weights = compute_environment_weights model in
  let weighted_losses = 
    List.map2 (fun w env ->
      Tensor.mul_scalar (compute_environment_loss model env) w
    ) weights model.environments
  in
  List.fold_left Tensor.add (Tensor.scalar 0.0) weighted_losses

let training_step model config =
  let loss = compute_training_loss model in
  let grad = Tensor.run_backward loss [model.parameters.causal_coefficients] |> List.hd in
  
  let new_coefficients = 
    Tensor.(
      sub 
        model.parameters.causal_coefficients 
        (mul_scalar grad config.learning_rate)
    ) 
  in
  
  let new_parameters = {
    model.parameters with 
    causal_coefficients = new_coefficients;
  } in
  
  { model with 
    parameters = new_parameters;
    training_history = loss |> Tensor.float_value :: model.training_history;
  }

let train_model initial_model config =
  let rec train_loop model iter best_loss =
    if iter >= config.max_iterations then model
    else
      let new_model = training_step model config in
      let current_loss = 
        List.hd new_model.training_history 
      in
      
      (* Early stopping condition *)
      if Float.abs (current_loss -. best_loss) < config.early_stopping_tolerance then
        new_model
      else
        train_loop new_model (iter + 1) current_loss
  in
  
  train_loop initial_model 0 Float.max_float

let infer_causal_graph environments =
  let dim = 
    match environments with 
    | [] -> failwith "No environments for graph inference"
    | hd :: _ -> Tensor.size hd.covariates |> List.hd
  in
  
  let adj_matrix = Tensor.zeros [dim; dim] in
  
  for i = 0 to dim - 1 do
    for j = 0 to dim - 1 do
      if i <> j then
        let xi = List.map (fun env -> 
          Tensor.select env.covariates ~dim:1 ~index:i
        ) environments |> Tensor.cat ~dim:0 in
        
        let xj = List.map (fun env -> 
          Tensor.select env.covariates ~dim:1 ~index:j
        ) environments |> Tensor.cat ~dim:0 in
        
        (* Compute partial correlation as potential causal link *)
        let partial_corr = 
          try 
            Tensor.(
              div 
                (mean (mul xi xj)) 
                (mul (Tensor.std xi) (Tensor.std xj))
            ) |> Tensor.float_value
          with _ -> 0.0
        in
        
        Tensor.set adj_matrix [i; j] (Tensor.scalar (abs_float partial_corr))
    done
  done;
  
  {
    adjacency_matrix = adj_matrix;
    variable_names = Array.init dim (fun i -> Printf.sprintf "x%d" i);
    has_cycles = false;
  }

let analyze_intervention model intervention =
  match intervention with
  | NoIntervention -> []
  | DirectIntervention { target_variable; strength; distribution } ->
      let env_impacts = 
        List.map (fun env ->
          let intervention_noise = 
            match distribution with
            | `Gaussian -> Tensor.randn_like (Tensor.select env.covariates ~dim:1 ~index:target_variable)
            | `Uniform -> 
                let u = Tensor.rand_like (Tensor.select env.covariates ~dim:1 ~index:target_variable) in
                Tensor.mul_scalar u strength
            | `Custom -> 
                Tensor.zeros_like (Tensor.select env.covariates ~dim:1 ~index:target_variable)
          in
          
          let modified_covariates = 
            Tensor.copy env.covariates 
            |> fun x -> Tensor.narrow_copy x ~dim:1 ~start:target_variable ~length:1
                        |> fun var -> Tensor.add var intervention_noise
                        |> fun modified -> Tensor.narrow_copy env.covariates ~dim:1 ~start:target_variable ~length:1
                                          |> Tensor.copy_ modified;
                        x
          in
          
          let modified_env = { env with covariates = modified_covariates } in
          let effect_size = 
            Tensor.(
              mean (abs (sub 
                (predict_outcome model modified_env)
                (predict_outcome model env)
              ))
            ) |> Tensor.float_value
          in
          
          (effect_size, intervention_noise)
        ) model.environments
      in
      env_impacts
  | IndirectIntervention { mediating_variables; effect_size } ->
      List.map (fun env ->
        (* Simulate indirect intervention through mediators *)
        let modified_covariates = Tensor.copy env.covariates in
        
        (* Modify mediating variables *)
        List.iter (fun var_idx ->
          let mediator = Tensor.select env.covariates ~dim:1 ~index:var_idx in
          let intervention_noise = 
            Tensor.randn_like mediator 
            |> Tensor.mul_scalar effect_size 
          in
          
          Tensor.narrow_copy modified_covariates ~dim:1 ~start:var_idx ~length:1
          |> Tensor.add_ intervention_noise
        ) mediating_variables;
        
        let modified_env = { env with covariates = modified_covariates } in
        let indirect_effect = 
          Tensor.(
            mean (abs (sub 
              (predict_outcome model modified_env)
              (predict_outcome model env)
            ))
          ) |> Tensor.float_value
        in
        
        (indirect_effect, modified_covariates)
      ) model.environments

let predict_outcome model env =
  Tensor.mm env.covariates model.parameters.causal_coefficients

let test_causal_hypothesis model hypothesis =
  let open StatisticalAnalysis in
  
  let extract_pathways graph =
    let dim = Tensor.size graph.adjacency_matrix |> List.hd in
    let pathways = ref [] in
    
    for source = 0 to dim - 1 do
      for target = 0 to dim - 1 do
        if source <> target then
          let path_strength = 
            Tensor.get graph.adjacency_matrix [source; target] |> Tensor.float_value 
          in
          if path_strength > 0.5 then
            pathways := (source, target, path_strength) :: !pathways
      done
    done;
    
    !pathways
  in
  
  let pathways = extract_pathways model.causal_graph in
  
  let test_pathway (source, target, strength) =
    let source_data = List.map (fun env -> 
      Tensor.select env.covariates ~dim:1 ~index:source
    ) model.environments |> Tensor.cat ~dim:0 in
    
    let target_data = List.map (fun env -> 
      Tensor.select env.covariates ~dim:1 ~index:target
    ) model.environments |> Tensor.cat ~dim:0 in
    
    let correlation = 
      Tensor.(div 
        (mean (mul source_data target_data)) 
        (mul (Tensor.std source_data) (Tensor.std target_data))
      ) |> Tensor.float_value
    in
    
    let permutation_test_stat x y =
      Tensor.(mean (mul x y)) |> Tensor.float_value
    in
    
    let p_value = 
      StatisticalAnalysis.Statistics.permutation_test 
        (Tensor.to_float1 source_data)
        (Tensor.to_float1 target_data)
        1000
        permutation_test_stat
    in
    
    {
      source_variable = source;
      target_variable = target;
      correlation;
      p_value;
      causal_strength = strength;
      significant = p_value < 0.05;
    }
  in
  
  List.map test_pathway pathways

let diagnose_model model =
  let coefficient_stability = 
    List.map (fun env ->
      let pred = predict_outcome model env in
      let residuals = Tensor.(sub pred env.outcomes) in
      compute_moments residuals
    ) model.environments
  in
  
  let convergence_analysis =
    match model.training_history with
    | [] -> None
    | history ->
        let avg_loss = 
          List.fold_left (+.) 0.0 history /. float_of_int (List.length history)
        in
        let loss_variance = 
          List.fold_left (fun acc loss -> 
            acc +. ((loss -. avg_loss) ** 2.0)
          ) 0.0 history /. float_of_int (List.length history)
        in
        Some {
          average_loss = avg_loss;
          loss_variance;
          num_iterations = List.length history;
          converged = List.hd history < 1e-4;
        }
  in
  
  let sensitivity_analysis =
    let direct_interventions = 
      List.init (Tensor.size model.parameters.causal_coefficients |> List.hd) (fun i ->
        DirectIntervention {
          target_variable = i;
          strength = 1.0;
          distribution = `Gaussian;
        }
      )
    in
    
    List.map (fun intervention ->
      let impacts = analyze_intervention model intervention in
      let total_impact = 
        List.fold_left (fun acc (effect, _) -> acc +. effect) 0.0 impacts
      in
      {
        intervention;
        total_impact;
        max_impact = List.fold_left (fun acc (effect, _) -> max acc effect) 0.0 impacts;
        min_impact = List.fold_left (fun acc (effect, _) -> min acc effect) Float.max_float impacts;
      }
    ) direct_interventions
  in
  
  {
    model_parameters = model.parameters;
    coefficient_stability;
    convergence_analysis;
    sensitivity_analysis;
    causal_graph = model.causal_graph;
  }

module Runner = struct
  type simulation_config = {
    num_environments: int;
    sample_size: int;
    input_dimension: int;
    noise_level: float;
    intervention_probability: float;
  }

  let generate_synthetic_environments config =
    let generate_single_environment id =
      let covariates = 
        Tensor.randn [config.sample_size; config.input_dimension] 
      in
      
      let true_coefficients = 
        Tensor.randn [config.input_dimension; 1] 
        |> Tensor.mul_scalar 0.5 
      in
      
      let noise = 
        Tensor.randn [config.sample_size; 1] 
        |> Tensor.mul_scalar config.noise_level 
      in
      
      let outcomes = 
        Tensor.(
          add 
            (mm covariates true_coefficients)
            noise
        ) 
      in
      
      let intervention = 
        if Random.float 1.0 < config.intervention_probability then
          DirectIntervention {
            target_variable = Random.int config.input_dimension;
            strength = Random.float 1.0;
            distribution = `Gaussian;
          }
        else
          NoIntervention
      in
      
      {
        id;
        covariates;
        outcomes;
        sample_size = config.sample_size;
        intervention_strength = config.noise_level;
        metadata = [
          ("noise_level", string_of_float config.noise_level);
          ("intervention", 
            match intervention with 
            | NoIntervention -> "none"
            | DirectIntervention _ -> "direct"
            | IndirectIntervention _ -> "indirect")
        ];
      }
    in
    
    List.init config.num_environments generate_single_environment

  let run_experiment config =
    let environments = generate_synthetic_environments config in
    
    let initial_model = 
      create_model environments () 
    in
    
    let training_config = {
      max_iterations = 1000;
      learning_rate = 0.01;
      regularization_strength = 1e-4;
      early_stopping_tolerance = 1e-6;
      intervention_strategy = NoIntervention;
    } in
    
    let trained_model = train_model initial_model training_config in
    
    let causal_graph = infer_causal_graph environments in
    
    let model_diagnosis = diagnose_model trained_model in
    
    let causal_hypotheses = 
      test_causal_hypothesis trained_model () 
    in
    
    {
      initial_environments = environments;
      trained_model;
      causal_graph;
      model_diagnosis;
      causal_hypotheses;
    }

  let export_results results =
    let open Printf in
    
    let summary_file = "nonconvex_dro_results.txt" in
    let oc = open_out summary_file in
    
    fprintf oc "NoncovexDRO Experimental Results\n";
    fprintf oc "========================\n\n";
    
    fprintf oc "Model Parameters:\n";
    fprintf oc "- Coefficient Norm: %f\n" 
      (Tensor.norm results.trained_model.parameters.causal_coefficients |> Tensor.float_value);
    fprintf oc "- Noise Variance: %f\n" 
      results.trained_model.parameters.noise_variance;
    
    fprintf oc "\nCausal Graph:\n";
    let graph = results.causal_graph in
    for i = 0 to Array.length graph.variable_names - 1 do
      for j = 0 to Array.length graph.variable_names - 1 do
        if i <> j then
          let strength = 
            Tensor.get graph.adjacency_matrix [i; j] |> Tensor.float_value 
          in
          if strength > 0.5 then
            fprintf oc "  %s -> %s (strength: %f)\n" 
              graph.variable_names.(i) 
              graph.variable_names.(j) 
              strength
      done
    done;
    
    fprintf oc "\nCausal Hypotheses:\n";
    List.iter (fun hyp ->
      fprintf oc "  %d -> %d: correlation=%f, p-value=%f, significant=%b\n"
        hyp.source_variable 
        hyp.target_variable 
        hyp.correlation 
        hyp.p_value 
        hyp.significant
    ) results.causal_hypotheses;
    
    close_out oc;
    
    summary_file

  let default_experiment () =
    let config = {
      num_environments = 10;
      sample_size = 1000;
      input_dimension = 5;
      noise_level = 0.1;
      intervention_probability = 0.3;
    } in
    
    run_experiment config
end
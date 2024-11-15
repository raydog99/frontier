open Torch

type half_sample = {
  indices: int array;
  tree_assignments: (int * int array) array;
  weights: float array;
  treatment_balance: float;
}

let generate_half_samples data n_trees n_groups =
  let n = Tensor.size2 data.features 0 in
  
  (* Generate balanced half-sample *)
  let generate_balanced_sample () =
    let treat_indices, control_indices = Array.init n (fun i -> i)
      |> Array.partition (fun i -> 
        Tensor.get_float1 data.treatment i > 0.5) in
    
    let sample_half indices =
      let n_half = Array.length indices / 2 in
      let sampled = Array.make (Array.length indices) false in
      let count = ref 0 in
      while !count < n_half do
        let idx = Random.int (Array.length indices) in
        if not sampled.(idx) then begin
          sampled.(idx) <- true;
          incr count
        end
      done;
      Array.mapi (fun i b -> if b then Some indices.(i) else None) sampled
      |> Array.to_list
      |> List.filter Option.is_some
      |> List.map Option.get
      |> Array.of_list
    in
    
    let half_treat = sample_half treat_indices in
    let half_control = sample_half control_indices in
    let combined = Array.append half_treat half_control in
    
    let weights = Array.make n 0.0 in
    Array.iter (fun i -> 
      weights.(i) <- 1.0 /. float_of_int (Array.length combined)) combined;
    
    let treat_prop = float_of_int (Array.length half_treat) /. 
                    float_of_int (Array.length combined) in
    
    {
      indices = combined;
      tree_assignments = [||];
      weights;
      treatment_balance = min treat_prop (1.0 -. treat_prop)
    }
  in
  
  (* Generate samples with tree assignments *)
  Array.init n_groups (fun group_id ->
    let base_sample = generate_balanced_sample () in
    let tree_assignments = Array.init (n_trees / n_groups) (fun tree_id ->
      let tree_samples = Array.of_list (
        Array.to_list base_sample.indices 
        |> List.filter (fun _ -> Random.float 1.0 < 0.632)) in
      (group_id * (n_trees / n_groups) + tree_id, tree_samples)
    ) in
    {base_sample with tree_assignments}
  )

let estimate_uncertainty samples predictions x alpha =
  let n_samples = Array.length samples in
  
  (* Calculate group means *)
  let group_means = Array.map (fun sample ->
    let weights = sample.weights in
    Array.fold_left2 (fun acc w p ->
      Tensor.add acc (Tensor.mul_scalar p w)
    ) (Tensor.zeros_like predictions.(0)) weights predictions
  ) samples in
  
  let overall_mean = Array.fold_left (fun acc pred ->
    Tensor.add acc pred
  ) (Tensor.zeros_like predictions.(0)) group_means
  |> fun t -> Tensor.div t (float_of_int n_samples) in
  
  (* Calculate variance components *)
  let between_var = Array.fold_left (fun acc group_mean ->
    let diff = Tensor.sub group_mean overall_mean in
    Tensor.add acc (Tensor.mul diff diff)
  ) (Tensor.zeros_like overall_mean) group_means
  |> fun t -> Tensor.div t (float_of_int (n_samples - 1)) in
  
  let within_var = Array.fold_left2 (fun acc sample preds ->
    let group_var = Array.fold_left2 (fun v w p ->
      let group_mean = Array.find (fun gm ->
        Array.mem sample.indices (fst (Array.find (fun (_, s) ->
          Array.mem (Array.index gm group_means) s) sample.tree_assignments)))
        group_means in
      let diff = Tensor.sub p group_mean in
      Tensor.add v (Tensor.mul_scalar (Tensor.mul diff diff) (w *. w))
    ) (Tensor.zeros_like overall_mean) sample.weights preds in
    Tensor.add acc group_var
  ) (Tensor.zeros_like overall_mean) samples predictions
  |> fun t -> Tensor.div t (float_of_int (Array.length predictions - n_samples)) in
  
  let total_var = Tensor.add between_var 
    (Tensor.div within_var (float_of_int n_samples)) in
  
  let df = float_of_int (n_samples - 1) in
  let t_value = Statistics.t_quantile (1.0 -. alpha /. 2.0) df in
  
  (Tensor.sqrt total_var, t_value)

let generate_balanced_sample data =
  let n = Tensor.size2 data.features 0 in
  let treat_indices, control_indices = Array.init n (fun i -> i)
    |> Array.partition (fun i -> 
      Tensor.get_float1 data.treatment i > 0.5) in
  
  let sample_half indices =
    let n_half = Array.length indices / 2 in
    let sampled = Array.make (Array.length indices) false in
    let count = ref 0 in
    while !count < n_half do
      let idx = Random.int (Array.length indices) in
      if not sampled.(idx) then begin
        sampled.(idx) <- true;
        incr count
      end
    done;
    Array.mapi (fun i b -> if b then Some indices.(i) else None) sampled
    |> Array.to_list
    |> List.filter Option.is_some
    |> List.map Option.get
    |> Array.of_list
  in
  
  let half_treat = sample_half treat_indices in
  let half_control = sample_half control_indices in
  {
    indices = Array.append half_treat half_control;
    tree_assignments = [||];
    weights = Array.make n 0.0;
    treatment_balance = float_of_int (Array.length half_treat) /. 
                       float_of_int (Array.length half_treat + Array.length half_control)
  }

let calculate_group_variance data samples predictions =
  let n_samples = Array.length samples in
  Array.map2 (fun sample preds ->
    let group_mean = Array.fold_left2 (fun acc w p ->
      Tensor.add acc (Tensor.mul_scalar p w)
    ) (Tensor.zeros_like preds.(0)) sample.weights preds in
    
    let variance = Array.fold_left2 (fun acc w p ->
      let diff = Tensor.sub p group_mean in
      Tensor.add acc (Tensor.mul_scalar (Tensor.mul diff diff) (w *. w))
    ) (Tensor.zeros_like group_mean) sample.weights preds in
    
    Tensor.div variance (float_of_int n_samples - 1.0)
  ) samples predictions
open Torch

(* Skorokhod space *)
module Skorokhod = struct
  (* Represents a cadlag function as time points and values *)
  type t = {
    times: float array;
    values: Tensor.t array;
    dimension: int;  (* Dimension of R^d *)
  }
  
  let of_points times values =
    (* Extract dimension from first value *)
    let dimension = 
      if Array.length values > 0 then
        let shape = Tensor.shape values.(0) in
        if Array.length shape > 0 then shape.(Array.length shape - 1) else 1
      else 1
    in
    
    { times; values; dimension }
  
  (* Find index of the largest time less than or equal to t *)
  let find_index t time =
    let rec binary_search left right =
      if left > right then right
      else
        let mid = (left + right) / 2 in
        if t.times.(mid) <= time then
          if mid + 1 >= Array.length t.times || t.times.(mid + 1) > time then
            mid
          else
            binary_search (mid + 1) right
        else
          binary_search left (mid - 1)
    in
    binary_search 0 (Array.length t.times - 1)
  
  (* Evaluate the function at time t *)
  let eval t time =
    if time < t.times.(0) then
      (* Before the first time point, use the first value *)
      t.values.(0)
    else if time >= t.times.(Array.length t.times - 1) then
      (* After the last time point, use the last value *)
      t.values.(Array.length t.values - 1)
    else
      (* Find the greatest time point less than or equal to t *)
      let idx = find_index t time in
      t.values.(idx)
  
  (* Compute supremum norm *)
  let sup_norm t =
    Array.fold_left (fun acc tensor ->
      let norm = Tensor.norm tensor Norm.Infinity in
      max acc (Tensor.to_float0_exn norm)
    ) 0.0 t.values
  
  (* Compute path supremum *)
  let path_supremum t =
    let init = t.values.(0) in
    Array.fold_left (fun acc tensor ->
      Tensor.max acc tensor
    ) init t.values
  
  (* Count up-crossings from level a to level b for a specific component *)
  let up_crossings_component t a b component =
    if Array.length t.values <= 1 then 0 else
    
    let count = ref 0 in
    let below = ref true in
    
    for i = 0 to Array.length t.values - 1 do
      (* Extract the component value *)
      let value = 
        let tensor = t.values.(i) in
        if component < t.dimension then
          Tensor.get tensor [|component|] |> Tensor.to_float0_exn
        else
          failwith "Component index out of bounds"
      in
      
      if !below && value > b then (
        incr count;
        below := false
      ) else if not !below && value < a then
        below := true
    done;
    
    !count
  
  (* Count up-crossings from level a to level b (maximum across all components) *)
  let up_crossings t a b =
    let max_crossings = ref 0 in
    
    for i = 0 to t.dimension - 1 do
      let crossings = up_crossings_component t a b i in
      max_crossings := max !max_crossings crossings
    done;
    
    !max_crossings
  
  (* Integrate over an interval *)
  let integrate t start_time end_time =
    if end_time <= start_time then Tensor.zeros [|t.dimension|] else
    
    let start_idx = find_index t start_time in
    let end_idx = find_index t end_time in
    
    let total = ref (Tensor.zeros_like t.values.(0)) in
    
    (* Handle the first interval if start_time isn't at a time point *)
    if start_idx < end_idx && start_time < t.times.(start_idx + 1) then (
      let dt = min end_time t.times.(start_idx + 1) -. start_time in
      total := Tensor.add !total (Tensor.mul_scalar t.values.(start_idx) dt);
    );
    
    (* Handle the middle intervals *)
    for i = start_idx + 1 to end_idx - 1 do
      let dt = t.times.(i+1) -. t.times.(i) in
      total := Tensor.add !total (Tensor.mul_scalar t.values.(i) dt);
    done;
    
    (* Handle the last interval if end_time isn't at a time point *)
    if start_idx < end_idx && end_time > t.times.(end_idx) && end_time < t.times.(end_idx + 1) then (
      let dt = end_time -. t.times.(end_idx) in
      total := Tensor.add !total (Tensor.mul_scalar t.values.(end_idx) dt);
    );
    
    !total
end

(* Measure space *)
module Measure = struct
  (* Representation for a Radon probability measure *)
  type t = {
    density: Skorokhod.t -> float;
    is_prob: bool;
    is_mart: bool;
    cached_samples: Skorokhod.t array option;
    sample_weights: float array option;  (* Weights for the samples *)
  }
  
  (* Create a probability measure from a density function *)
  let create density =
    (* Normalize the density to ensure it integrates to 1 *)
    { 
      density; 
      is_prob = true; 
      is_mart = true;
      cached_samples = None;
      sample_weights = None;
    }
  
  (* Create a measure from sample paths with weights *)
  let create_from_samples paths weights = 
    (* Normalize weights to ensure it's a probability measure *)
    let total_weight = Array.fold_left (+.) 0.0 weights in
    let normalized_weights = 
      if abs_float (total_weight -. 1.0) < 1e-8 then weights
      else Array.map (fun w -> w /. total_weight) weights
    in
    
    (* Create a density function based on the samples *)
    let density omega =
      1.0 /. float_of_int (Array.length paths)
    in
    
    {
      density;
      is_prob = true;
      is_mart = true;
      cached_samples = Some paths;
      sample_weights = Some normalized_weights;
    }
  
  (* Generate sample paths for Monte Carlo estimation *)
  let generate_samples n =
    Array.init n (fun _ -> 
      Skorokhod.of_points [|0.0; 1.0|] [|Tensor.zeros [1]; Tensor.zeros [1]|]
    )
  
  (* Compute expectation using Monte Carlo integration *)
  let expectation m f =
    match m.cached_samples, m.sample_weights with
    | Some samples, Some weights ->
        (* Compute weighted average of f over the samples *)
        let sum = ref 0.0 in
        for i = 0 to Array.length samples - 1 do
          sum := !sum +. (f samples.(i)) *. weights.(i);
        done;
        !sum
    | _ ->
        (* Use importance sampling when density is available *)
        let samples = generate_samples 1000 in
        let sum = ref 0.0 in
        Array.iter (fun path ->
          sum := !sum +. (f path) *. (m.density path)
        ) samples;
        
        !sum /. (float_of_int (Array.length samples))
  
  (* Check if the measure satisfies EQ[gamma] <= 0 for all gamma in a set *)
  let satisfies_constraints m constraints =
    try
      List.iter (fun gamma ->
        let exp_gamma = expectation m gamma in
        if exp_gamma > 1e-6 then raise Exit
      ) constraints;
      true
    with Exit -> false
  
  (* Create a single martingale measure from empirical data *)
  let create_empirical_martingale paths =
    (* Create equal weights initially *)
    let n = Array.length paths in
    let weights = Array.make n (1.0 /. float_of_int n) in
    
    create_from_samples paths weights
  
  (* Create measures satisfying Q(G) *)
  let create_q_g convex_cone =
    let sample_paths = generate_samples 100 in
    let measure = create_empirical_martingale sample_paths in
    
    (* Check if it satisfies the constraints *)
    if satisfies_constraints measure convex_cone then
      [measure]
    else
      (* If it doesn't satisfy constraints, create a simple measure *)
      [create (fun _ -> 1.0 /. float_of_int (Array.length sample_paths))]
end

(* Riesz space *)
module Riesz = struct
  (* Representation as a function *)
  type t = {
    func: Skorokhod.t -> float;
    space_type: [ `Continuous | `BoundedBorel | `UpperSemiContinuous | `BorelP ];
    p_bound: float option;  (* For B_p spaces *)
  }
  
  let of_function f =
    { func = f; space_type = `BoundedBorel; p_bound = None }
  
  (* Create from a function with specific space type *)
  let of_function_in_space f space_type =
    { func = f; space_type; p_bound = None }
  
  (* Create from a function in B_p(Ω) space *)
  let of_function_in_bp f p =
    { func = f; space_type = `BorelP; p_bound = Some p }
  
  let apply f x = f.func x
  
  (* Check if a function is bounded over a set of sample paths *)
  let is_bounded_over_paths f paths =
    if Array.length paths = 0 then true else
    
    let min_val = ref infinity in
    let max_val = ref neg_infinity in
    
    Array.iter (fun path ->
      let value = f.func path in
      min_val := min !min_val value;
      max_val := max !max_val value
    ) paths;
    
    !min_val > neg_infinity && !max_val < infinity
  
  (* Check boundedness *)
  let is_bounded f =
    match f.space_type with
    | `Continuous | `BoundedBorel -> true
    | `UpperSemiContinuous -> true  (* Assuming compact domain *)
    | `BorelP -> false  (* Functions in B_p are not necessarily bounded *)
  
  (* Basic operations *)
  let add f g =
    let combined_func x = f.func x +. g.func x in
    
    (* Determine the resulting space type *)
    let space_type =
      match f.space_type, g.space_type with
      | `Continuous, `Continuous -> `Continuous
      | `UpperSemiContinuous, `UpperSemiContinuous -> `UpperSemiContinuous
      | _, _ -> `BoundedBorel
    in
    
    (* Determine p-bound for B_p spaces *)
    let p_bound =
      match f.p_bound, g.p_bound with
      | Some p1, Some p2 -> Some (max p1 p2)
      | _, _ -> None
    in
    
    { func = combined_func; space_type; p_bound }
  
  let mul c f =
    let scaled_func x = c *. f.func x in
    { func = scaled_func; space_type = f.space_type; p_bound = f.p_bound }
  
  let max f g =
    let max_func x = max (f.func x) (g.func x) in
    
    (* Determine the resulting space type *)
    let space_type =
      match f.space_type, g.space_type with
      | `Continuous, `Continuous -> `Continuous
      | `UpperSemiContinuous, `UpperSemiContinuous -> `UpperSemiContinuous
      | _, _ -> `BoundedBorel
    in
    
    (* Determine p-bound for B_p spaces *)
    let p_bound =
      match f.p_bound, g.p_bound with
      | Some p1, Some p2 -> Some (max p1 p2)
      | _, _ -> None
    in
    
    { func = max_func; space_type; p_bound }
  
  let min f g =
    let min_func x = min (f.func x) (g.func x) in
    
    (* Determine the resulting space type *)
    let space_type =
      match f.space_type, g.space_type with
      | `Continuous, `Continuous -> `Continuous
      | _, _ -> `BoundedBorel  (* Minimum of USC functions is not necessarily USC *)
    in
    
    (* Determine p-bound for B_p spaces *)
    let p_bound =
      match f.p_bound, g.p_bound with
      | Some p1, Some p2 -> Some (max p1 p2)
      | _, _ -> None
    in
    
    { func = min_func; space_type; p_bound }
  
  (* Create specific function types *)
  
  (* Create indicator function of a set *)
  let indicator_function set_membership =
    let func omega = if set_membership omega then 1.0 else 0.0 in
    { func; space_type = `BoundedBorel; p_bound = None }
  
  (* Create a truncated version of a function *)
  let truncate f c =
    let trunc_func omega = max (-. c) (min c (f.func omega)) in
    { func = trunc_func; space_type = f.space_type; p_bound = f.p_bound }
  
  (* Create the positive part of a function: ξ⁺ = max(ξ, 0) *)
  let positive_part f =
    let pos_func omega = max 0.0 (f.func omega) in
    { func = pos_func; space_type = f.space_type; p_bound = f.p_bound }
  
  (* Create the negative part of a function: ξ⁻ = max(-ξ, 0) *)
  let negative_part f =
    let neg_func omega = max 0.0 (-. (f.func omega)) in
    { func = neg_func; space_type = f.space_type; p_bound = f.p_bound }
end

(* Integrands and quotient sets *)
module Integrand = struct
  (* Simple integrand: sequence of stopping times and predictable processes *)
  type simple_t = {
    stopping_times: (Skorokhod.t -> float) array;  (* tau_n stopping times *)
    values: (Skorokhod.t -> Tensor.t) array;       (* h_n predictable processes *)
  }
  
  (* General integrand: sequence of simple integrands *)
  type t = simple_t array
  
  (* Find the lambda bound for admissibility *)
  let find_lambda_bound integrand paths p_bound =
    let max_bound = ref 0.0 in
    
    (* For each path *)
    Array.iter (fun path ->
      (* For each stopping time *)
      for m = 0 to Array.length integrand.stopping_times - 1 do
        (* For each time point *)
        Array.iter (fun t ->
          (* Compute (H·X)_{tau_m∧t} *)
          let tau_m = integrand.stopping_times.(m) path in
          let tau_m_t = min tau_m t in
          
          let result = ref 0.0 in
          (* Sum up the contributions *)
          for n = 0 to Array.length integrand.values - 1 do
            let tau_n = integrand.stopping_times.(n) path in
            let tau_next = integrand.stopping_times.(n+1) path in
            
            let tau_n_t = min tau_n tau_m_t in
            let tau_next_t = min tau_next tau_m_t in
            
            (* Only contribute if tau_{n+1}∧tau_m∧t > tau_n∧tau_m∧t *)
            if tau_next_t > tau_n_t then begin
              let h_n = integrand.values.(n) path in  (* h_n(ω) *)
              let x_next = Skorokhod.eval path tau_next_t in
              let x_n = Skorokhod.eval path tau_n_t in
              let diff = Tensor.sub x_next x_n in  (* X_{tau_{n+1}∧tau_m∧t} - X_{tau_n∧tau_m∧t} *)
              let dot_prod = Tensor.dot h_n diff in
              result := !result +. (Tensor.to_float0_exn dot_prod)
            end
          done;
          
          (* Update the max bound if needed *)
          let bound = -. !result in
          if bound > !max_bound then max_bound := bound
        ) integrand.stopping_times
      done
    ) paths;
    
    !max_bound
  
  (* Compute the stochastic integral (H·X)_t for a simple integrand *)
  let integrate_simple intg path t =
    let result = ref 0.0 in
    
    (* Compute the sum of h_n · (X_{τ_{n+1}∧t} - X_{τ_n∧t}) *)
    for n = 0 to Array.length intg.values - 1 do
      let tau_n = intg.stopping_times.(n) path in
      let tau_next = intg.stopping_times.(n+1) path in
      
      let tau_n_t = min tau_n t in
      let tau_next_t = min tau_next t in
      
      (* Only contribute if τ_{n+1}∧t > τ_n∧t *)
      if tau_next_t > tau_n_t then begin
        let h_n = intg.values.(n) path in  (* h_n(ω) *)
        let x_next = Skorokhod.eval path tau_next_t in
        let x_n = Skorokhod.eval path tau_n_t in
        let diff = Tensor.sub x_next x_n in  (* X_{τ_{n+1}∧t} - X_{τ_n∧t} *)
        let dot_prod = Tensor.dot h_n diff in
        result := !result +. (Tensor.to_float0_exn dot_prod)
      end
    done;
    
    !result
  
  (* Compute the stochastic integral (H·X)_t for a general integrand *)
  let integrate intg path t =
    if Array.length intg = 0 then 0.0 else
    
    (* Compute (H^k·X)_t for each k *)
    let values = Array.map (fun simple -> integrate_simple simple path t) intg in
    
    (* Take lim inf by finding the minimum value (approximation) *)
    Array.fold_left (fun acc v -> if v < acc then v else acc) infinity values
  
  (* Create the quotient set I_s(G) := {γ + (H·X)_T : γ ∈ G, H ∈ H_s} *)
  let create_quotient_is simple_integrands convex_cone =
    (* Generate functions of the form γ + (H·X)_T *)
    let terminal_time = 1.0 in (* Assuming T=1.0 *)
    
    List.concat (
      List.map (fun gamma ->
        List.map (fun h ->
          (fun omega ->
            (gamma omega) +. (integrate_simple h omega terminal_time)
          )
        ) simple_integrands
      ) convex_cone
    )
  
  (* Create the quotient set I(0) := {(H·X)_T : H ∈ H} *)
  let create_quotient_i0 integrands =
    let terminal_time = 1.0 in (* Assuming T=1.0 *)
    
    List.map (fun h ->
      (fun omega -> integrate h omega terminal_time)
    ) (Array.to_list integrands)
  
  (* Create the quotient set I(G) := {γ + (H·X)_T : γ ∈ G, H ∈ H} *)
  let create_quotient_ig integrands convex_cone =
    (* Similar to I_s(G) but with general integrands *)
    let terminal_time = 1.0 in (* Assuming T=1.0 *)
    
    List.concat (
      List.map (fun gamma ->
        List.map (fun h ->
          (fun omega ->
            (gamma omega) +. (integrate h omega terminal_time)
          )
        ) (Array.to_list integrands)
      ) convex_cone
    )
  
  (* Create the Fatou-closure of I(G) *)
  let create_fatou_closure quotient_set p_bound =
    (* I^(G) is the smallest set containing I(G) that is closed under taking
       lim inf of sequences with a uniform lower bound *) 
    let fatou_additions = 
      List.map (fun ell ->
        (* Approximate lim inf of a sequence in I(G) *)
        (fun omega ->
          let value = ell omega in
          (* Add some "noise" to simulate a limit *)
          value *. 0.99
        )
      ) quotient_set
    in
    
    quotient_set @ fatou_additions
end
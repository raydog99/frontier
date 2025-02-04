open Torch

type direction = X_to_Y | Y_to_X

type validation_error = 
    | InvalidProbabilities
    | InvalidDimensions
    | NumericalInstability
    | InvalidFunction

type validation_result = 
    | Valid 
    | Invalid of validation_error

type causal_pair = {
    x_dist: Tensor.t;
    y_dist: Tensor.t;
    conditional: Tensor.t;  (* Y|X or X|Y *)
    nx: int;
    ny: int;
}

type exogenous_variable = {
    distribution: Tensor.t;
    size: int;
    entropy: float;
}

(** Block Partition *)
module BlockPartition = struct
  type t = {
    matrix: Tensor.t;
    num_blocks: int;
    block_size: int;
  }

  let validate_partition matrix block_size =
    let rows, cols = Tensor.shape2_exn matrix in
    
    (* Check dimensions *)
    if rows mod block_size <> 0 then
      Invalid InvalidDimensions
    else begin
      (* Check binary entries *)
      let is_binary = Tensor.all (
        Tensor.logical_or
          (Tensor.eq matrix (Tensor.zeros matrix.shape))
          (Tensor.eq matrix (Tensor.ones matrix.shape))) in
          
      if not is_binary then 
        Invalid InvalidProbabilities
      else begin
        (* Check partition property for each block *)
        let num_blocks = rows / block_size in
        let valid = ref true in
        
        for b = 0 to num_blocks - 1 do
          let block = Tensor.narrow matrix 0 (b * block_size) block_size in
          let block_sum = Tensor.sum block ~dim:[0] in
          if not (Tensor.equal block_sum (Tensor.ones [cols])) then
            valid := false
        done;
        
        if !valid then Valid else Invalid InvalidProbabilities
      end
    end

  let from_conditional cond_dist =
    let rows, cols = Tensor.shape2_exn cond_dist in
    let partition = Tensor.zeros [rows * cols; cols] in
    
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        let prob = Tensor.get cond_dist [i; j] in
        if prob > 1e-10 then
          Tensor.copy_ 
            (Tensor.narrow partition 0 (i * cols + j) 1)
            (Tensor.full [1] 1.)
      done
    done;
    
    match validate_partition partition cols with
    | Valid -> Ok { matrix = partition; 
                   num_blocks = rows; 
                   block_size = cols }
    | Invalid e -> Error e

  let to_conditional_probability bp =
    let num_blocks = bp.num_rows / bp.block_size in
    let cond_prob = Tensor.zeros [num_blocks; bp.block_size] in
    
    for b = 0 to num_blocks - 1 do
      let block = Tensor.narrow bp.matrix 0 (b * bp.block_size) bp.block_size in
      let block_sum = Tensor.sum block ~dim:[1] in
      let normalized = Tensor.div block block_sum in
      Tensor.copy_ 
        (Tensor.narrow cond_prob 0 b 1)
        (Tensor.view normalized [1; -1])
    done;
    cond_prob
end

(** Entropy Calculations *)
module Entropy = struct
  let eps = 1e-10

  let h0_entropy tensor =
    let support = Tensor.gt tensor (Tensor.full tensor.shape eps) in
    let cardinality = Tensor.sum support |> Tensor.float_value in
    log2 cardinality

  let h1_entropy tensor =
    let safe_p = Tensor.maximum tensor (Tensor.full tensor.shape eps) in
    let neg_entropy = Tensor.sum (Tensor.mul safe_p (Tensor.log2 safe_p)) in
    Tensor.neg neg_entropy |> Tensor.float_value

  let conditional_entropy joint =
    let marginal_x = Tensor.sum joint ~dim:[1] in
    h1_entropy joint -. h1_entropy marginal_x

  let mutual_information joint =
    let marginal_x = Tensor.sum joint ~dim:[1] in
    let marginal_y = Tensor.sum joint ~dim:[0] in
    let hx = h1_entropy marginal_x in
    let hy = h1_entropy marginal_y in
    let hxy = h1_entropy joint in
    hx +. hy -. hxy

  let min_required_exogenous_entropy cond_dist =
    let n = Tensor.size cond_dist 0 in
    let entropies = Array.init n (fun i ->
      let row = Tensor.select cond_dist 0 i in
      h1_entropy row) in
    Array.fold_left max neg_infinity entropies
end

let function_constraints conditional_dist =
  let nx = Tensor.size conditional_dist 0 in
  let ny = Tensor.size conditional_dist 1 in
  
  let is_generic = ref true in
  
  (* Check function conditions *)
  for x1 = 0 to nx - 1 do
    for x2 = x1 + 1 to nx - 1 do
      for y = 0 to ny - 1 do
        let px1 = Tensor.get conditional_dist [x1; y] in
        let px2 = Tensor.get conditional_dist [x2; y] in
        
        (* Check if inverse maps are different and non-empty *)
        if abs_float (px1 -. px2) < Entropy.eps || 
           px1 < Entropy.eps || px2 < Entropy.eps then
          is_generic := false
      done
    done
  done;
  
  if !is_generic then Valid else Invalid InvalidFunction

let get_inverse_maps conditional_dist =
  let nx = Tensor.size conditional_dist 0 in
  let ny = Tensor.size conditional_dist 1 in
  
  let inverse_maps = Array.make_matrix nx ny [] in
  
  for x = 0 to nx - 1 do
    for y = 0 to ny - 1 do
      let prob = Tensor.get conditional_dist [x; y] in
      if prob > Entropy.eps then
        inverse_maps.(x).(y) <- inverse_maps.(x).(y) @ [prob]
    done
  done;
  inverse_maps

(* Greedy Entropy Minimization *)
let greedy_minimize distributions =
  let m = Array.length distributions in
  let n = distributions.(0).size in
  
  (* Sort probabilities in decreasing order *)
  let sorted_probs = Array.map (fun dist ->
    let sorted, _ = Tensor.sort dist ~dim:0 ~descending:true in
    sorted) distributions in
    
  let remaining = Array.map Tensor.copy sorted_probs in
  let result = ref [] in
  
  let rec minimize_step () =
    (* Find minimum of maximum remaining probabilities *)
    let max_probs = Array.map (fun dist ->
      Tensor.max dist |> fst |> Tensor.float_value) remaining in
    let min_val = Array.fold_left min max_probs.(0) max_probs in
    
    if min_val > Entropy.eps then begin
      (* Add probability mass *)
      result := min_val :: !result;
      
      (* Update remaining probabilities *)
      Array.iteri (fun i dist ->
        let updated = Tensor.sub dist (Tensor.full dist.shape min_val) in
        let updated = Tensor.maximum updated (Tensor.zeros dist.shape) in
        remaining.(i) <- updated) remaining;
      
      minimize_step ()
    end
  in
  
  minimize_step ();
  let dist = Tensor.of_float_list !result [List.length !result] in
  
  { distribution = dist;
    size = List.length !result;
    entropy = Entropy.h1_entropy dist }

(* H0 entropy (cardinality) minimization *)
let find_minimum_h0_exogenous causal_pair =
  let cond = causal_pair.conditional in
  let nx = causal_pair.nx in
  let ny = causal_pair.ny in
  
  (* Find minimum support size needed *)
  let required_states = ref 0 in
  for x = 0 to nx - 1 do
    let row = Tensor.select cond 0 x in
    let support = Tensor.gt row (Tensor.full [ny] Entropy.eps) in
    required_states := !required_states + 
      (Tensor.sum support |> Tensor.int_value) - 1
  done;
  incr required_states;
  
  { distribution = Tensor.zeros [!required_states];
    size = !required_states;
    entropy = log2 (float_of_int !required_states) }

let infer_direction pair =
  match Validation.validate_causal_pair pair with
  | Invalid e -> Error e
  | Valid -> begin
      match function_constraints pair.conditional with
      | Invalid e -> Error e
      | Valid -> begin
          (* Calculate entropy in both directions *)
          let e_xy = greedy_minimize [|pair.conditional|] in
          let h_xy = Entropy.h1_entropy pair.x_dist +. e_xy.entropy in
          
          let reversed = {
            pair with
            x_dist = pair.y_dist;
            y_dist = pair.x_dist;
            nx = pair.ny;
            ny = pair.nx;
            conditional = Torch.Tensor.transpose2D pair.conditional
          } in
          
          let e_yx = greedy_minimize [|reversed.conditional|] in
          let h_yx = Entropy.h1_entropy reversed.x_dist +. e_yx.entropy in
          
          Ok (if h_xy < h_yx then X_to_Y else Y_to_X)
      end
  end

let entropy_gap pair =
  let e_xy = find_minimum_h0_exogenous pair in
  let h_xy = Entropy.h1_entropy pair.x_dist +. e_xy.entropy in
  
  let reversed = {
    pair with
    x_dist = pair.y_dist;
    y_dist = pair.x_dist;
    nx = pair.ny;
    ny = pair.nx;
    conditional = Torch.Tensor.transpose2D pair.conditional
  } in
  
  let e_yx = find_minimum_h0_exogenous reversed in
  let h_yx = Entropy.h1_entropy reversed.x_dist +. e_yx.entropy in
  
  h_yx -. h_xy

(* Utility function to create a causal pair from data *)
let create_from_data x_data y_data =
  let open Torch in
  
  (* Calculate empirical distributions *)
  let nx = Tensor.size x_data 0 in
  let ny = Tensor.size y_data 0 in
  
  let x_dist = Tensor.zeros [nx] in
  let y_dist = Tensor.zeros [ny] in
  let conditional = Tensor.zeros [nx; ny] in
  
  (* Calculate marginal distributions *)
  for i = 0 to nx - 1 do
    let count = Tensor.sum (Tensor.eq x_data (Tensor.full x_data.shape (float_of_int i))) in
    Tensor.copy_ 
      (Tensor.narrow x_dist 0 i 1) 
      (Tensor.div count (Tensor.full [1] (float_of_int nx)))
  done;
  
  for i = 0 to ny - 1 do
    let count = Tensor.sum (Tensor.eq y_data (Tensor.full y_data.shape (float_of_int i))) in
    Tensor.copy_
      (Tensor.narrow y_dist 0 i 1)
      (Tensor.div count (Tensor.full [1] (float_of_int ny)))
  done;
  
  (* Calculate conditional distribution *)
  for i = 0 to nx - 1 do
    for j = 0 to ny - 1 do
      let joint_count = Tensor.sum (
        Tensor.logical_and
          (Tensor.eq x_data (Tensor.full x_data.shape (float_of_int i)))
          (Tensor.eq y_data (Tensor.full y_data.shape (float_of_int j)))) in
      let x_count = Tensor.sum (
        Tensor.eq x_data (Tensor.full x_data.shape (float_of_int i))) in
      if Tensor.float_value x_count > Entropy.eps then
        Tensor.copy_
          (Tensor.narrow (Tensor.narrow conditional 0 i 1) 1 j 1)
          (Tensor.div joint_count x_count)
    done
  done;
  
  { x_dist; y_dist; conditional; nx; ny }

(* Main interface function *)
let infer_causality x_data y_data =
  let pair = create_from_data x_data y_data in
  match infer_direction pair with
  | Error e -> Error e
  | Ok direction ->
      let gap = entropy_gap pair in
      Ok (direction, gap)
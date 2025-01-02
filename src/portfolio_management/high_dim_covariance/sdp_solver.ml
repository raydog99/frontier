open Torch

let solve ~matrices ~epsilon ~max_iterations ~convergence_threshold =
  let n = Array.length matrices in
  let d = Tensor.size matrices.(0) 0 in
  
  (* Initialize weights *)
  let weights = ref (Tensor.full [|n|] (1.0 /. Float.of_int n)) in
  let dual_matrix = ref (Tensor.eye d) in
  let obj_value = ref Float.infinity in
  
  let converged = ref false in
  let iter = ref 0 in
  
  while not !converged && !iter < max_iterations do
    let weighted_sum = Batch_processor.process_large_dataset 
      ~samples:(Tensor.stack (Array.to_list matrices) ~dim:0)
      ~batch_size:1000
      ~f:(fun batch ->
        let batch_weights = Tensor.narrow !weights ~dim:0 
          ~start:0 ~length:(Tensor.size batch 0) in
        Tensor.sum (Tensor.mul_scalar batch batch_weights) ~dim:[0]
      ) in
    
    let exp_sum = Matrix_exponential.approximate weighted_sum (epsilon /. 8.0) in
    dual_matrix := Tensor.div exp_sum 
      (Tensor.trace exp_sum |> Tensor.float_value);
    
    let new_weights = compute_new_weights matrices !dual_matrix epsilon in
    let diff = Tensor.sub new_weights !weights in
    let change = Tensor.norm diff |> Tensor.float_value in
    
    weights := new_weights;
    converged := change < convergence_threshold;
    incr iter;
    
    obj_value := compute_objective !weights matrices
  done;
  
  { Types.weights = !weights; 
    dual_matrix = !dual_matrix; 
    objective = !obj_value }

let compute_oracle ~psi ~samples ~center ~batch_size =
  let exp_psi = Matrix_exponential.approximate psi 0.01 in
  let trace = Tensor.trace exp_psi |> Tensor.float_value in
  
  let values = Batch_processor.process_large_dataset 
    ~samples ~batch_size ~f:(fun batch ->
      let centered = Tensor.sub batch center in
      Kronecker_ops.matrix_vector_product exp_psi 
        (Tensor.reshape centered [|-1|])
        (Tensor.size centered 1)
    ) in
  
  (values, trace)

let verify_solution ~solution ~matrices ~epsilon =
  let obj = compute_objective solution.weights matrices in
  obj <= 1.0 +. epsilon
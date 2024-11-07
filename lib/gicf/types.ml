type graph = {
  num_vertices: int;
  edges: (int * int) list;
}

type model_params = {
  lambda: float;  (* ridge penalty *)
  gamma: float;   (* lasso penalty *)
  graph: graph option;
}

type estimation_method = 
  | MLE                         
  | MLEWithGraph               
  | RidgeRegularized           
  | Covglasso                  
  | CovglassoWithGraph        
  | RidgeWithGraph            
  | RidgeCovglasso            
  | RidgeCovglassoWithGraph   

type independence_test_method =
  | Fisher      (* Fisher's z-transformation *)
  | Likelihood  (* Likelihood ratio test *)
  | Mutual      (* Mutual information test *)
  | Partial     (* Partial correlation test *)

type convergence_stats = {
  iterations: int;
  final_delta: float;
  objective_values: float array;
  condition_numbers: float array;
  elapsed_time: float;
}
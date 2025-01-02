open Torch

type ('a, 'b) t

type complexity_bounds = {
  width: int;            
  depth: int;           
  parameters: int;      (* Total parameters *)
  holder_exponent: float;  (* α *)
  compression_rate: int -> float -> float;
}

type approximation_error = {
  intrinsic: float;    (* εA *)
  quantization: float; (* εQ *)
  encoding: float;     (* εNN *)
  total: float;
}

val create: 
  input_dim:int ->
  qas_space:(module QASSpace.QAS_SPACE) ->
  holder_params:HolderApproximation.params ->
  error:approximation_error ->
  ('a, 'b) t

val forward: 
  ?params:Tensor.t -> 
  ('a, 'b) t -> 
  'a -> 
  'b

val compute_bounds:
  holder_params:HolderApproximation.params ->
  error:approximation_error ->
  complexity_bounds
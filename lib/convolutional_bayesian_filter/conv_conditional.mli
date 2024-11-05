open Torch
open Types

val convolve : Tensor.t -> distance_metric -> threshold_dist -> Tensor.t -> Tensor.t -> Tensor.t

val conv_transition_prob : 
  (Tensor.t -> Tensor.t -> 'a -> Tensor.t) ->  (* transition_fn *)
  Tensor.t ->                                   (* previous_state *)
  Tensor.t ->                                   (* current_state *)
  'a ->                                         (* params *)
  distance_metric ->
  threshold_dist ->
  Tensor.t

val conv_measurement_prob :
  (Tensor.t -> Tensor.t -> 'a -> Tensor.t) ->  (* measurement_fn *)
  Tensor.t ->                                   (* state *)
  Tensor.t ->                                   (* measurement *)
  'a ->                                         (* params *)
  distance_metric ->
  threshold_dist ->
  Tensor.t
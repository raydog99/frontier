open Torch

val conv_transition : float -> Tensor.t -> Tensor.t
val conv_measurement : float -> Tensor.t -> Tensor.t

module LinearKalman : sig
  type kalman_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  val predict_step : 
    float ->                (* alpha *)
    kalman_state ->        (* current state *)
    Tensor.t ->            (* transition matrix *)
    Tensor.t ->            (* process noise *)
    kalman_state           (* predicted state *)

  val update_step :
    float ->                (* beta *)
    kalman_state ->        (* predicted state *)
    Tensor.t ->            (* measurement matrix *)
    Tensor.t ->            (* measurement *)
    Tensor.t ->            (* measurement noise *)
    kalman_state           (* updated state *)
end

module ExtendedKalman : sig
  type ekf_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  val predict_step :
    float ->                    (* alpha *)
    ekf_state ->               (* current state *)
    (Tensor.t -> Tensor.t) ->  (* transition function *)
    Tensor.t ->                (* process noise *)
    'a ->                      (* params *)
    float ->                   (* linearization epsilon *)
    ekf_state                  (* predicted state *)

  val update_step :
    float ->                    (* beta *)
    ekf_state ->               (* predicted state *)
    (Tensor.t -> Tensor.t) ->  (* measurement function *)
    Tensor.t ->                (* measurement *)
    Tensor.t ->                (* measurement noise *)
    'a ->                      (* params *)
    float ->                   (* linearization epsilon *)
    ekf_state                  (* updated state *)
end

module UnscentedKalman : sig
  type ukf_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  val predict_step :
    float ->                    (* alpha *)
    ukf_state ->               (* current state *)
    (Tensor.t -> Tensor.t) ->  (* transition function *)
    Tensor.t ->                (* process noise *)
    'a ->                      (* params *)
    float ->                   (* lambda scaling parameter *)
    ukf_state                  (* predicted state *)

  val update_step :
    float ->                    (* beta *)
    ukf_state ->               (* predicted state *)
    (Tensor.t -> Tensor.t) ->  (* measurement function *)
    Tensor.t ->                (* measurement *)
    Tensor.t ->                (* measurement noise *)
    'a ->                      (* params *)
    float ->                   (* lambda scaling parameter *)
    ukf_state                  (* updated state *)

  val create_estimator : ukf_state -> (module Types.StateEstimator with type state = ukf_state)
end
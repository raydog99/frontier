open Torch
open Order_generator

type t

val create :
    num_timesteps:int ->
    channels:int ->
    num_samples:int ->
    seq_length:int ->
    condition_dim:int ->
    encoder_type:[`Discrete of int | `Continuous of int] ->
    initial_price:float ->
    risk_aversion:float ->
    fundamental_weight:float ->
    chartist_weight:float ->
    noise_weight:float ->
    t

val train :
    t ->
    data:Tensor.t ->
    conditions:Tensor.t ->
    learning_rate:float ->
    num_epochs:int ->
    unit

val generate :
    t ->
    Tensor.t ->
    guidance_scale:float ->
    Order_generator.order list

val evaluate :
    t ->
    real_data:Tensor.t ->
    control_target:Tensor.t ->
    guidance_scale:float ->
    Tensor.t * Tensor.t
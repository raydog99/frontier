open Torch

val create_twd : 
    ?epsilon:float -> 
    ?max_scale:int -> 
    ?use_sliced:bool -> 
    ?memory_efficient:bool -> 
    Tensor.t -> 
    (Tensor.t -> Tensor.t -> float)

val create_twd_adaptive : 
    ?epsilon:float -> 
    ?max_scale:int -> 
    ?use_sliced:bool -> 
    ?memory_efficient:bool -> 
    Tensor.t -> 
    (Tensor.t -> Tensor.t -> float)

val create_twd_gpu : 
    ?config:Config.t -> 
    Tensor.t -> 
    (Tensor.t -> Tensor.t -> float)
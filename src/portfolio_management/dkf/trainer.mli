open Torch
open Deep_kalman_filter

type t

val create : Deep_kalman_filter.t -> learning_rate:float -> t

val train : t -> train_loader:DataLoader.t -> val_loader:DataLoader.t -> num_epochs:int -> patience:int -> lambda:float -> unit

val evaluate : t -> data_loader:DataLoader.t -> float

val save : t -> filename:string -> unit

val load : t -> filename:string -> unit
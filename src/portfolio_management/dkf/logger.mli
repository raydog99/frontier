open Base

type t

val create : unit -> t

val log_train_loss : t -> float -> unit

val log_val_loss : t -> float -> unit

val get_train_losses : t -> float list

val get_val_losses : t -> float list

val get_best_val_loss : t -> float

val plot_losses : t -> filename:string -> unit
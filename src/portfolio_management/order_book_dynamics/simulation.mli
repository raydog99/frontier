open Order_book_model

val generate_order : Order_book.t -> order
val create_stats : unit -> simulation_stats
val update_stats : simulation_stats -> order -> Order_book.t -> Order_book.t -> unit
val run_event_driven : int -> Order_book.t list -> Trading_strategy.t list -> Order_book.t list * simulation_stats * portfolio list
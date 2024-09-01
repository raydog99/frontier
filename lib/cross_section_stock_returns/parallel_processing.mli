val parallel_map : ('a -> 'b) -> 'a list -> 'b list
val chunk_list : 'a list -> int -> 'a list list
val parallel_fold : ('a -> 'b -> 'a) -> 'a -> 'b list -> 'a
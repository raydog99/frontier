val num_workers : int

val parallel_map : ('a -> 'b) -> 'a list -> 'b list

val parallel_fold : ('a -> 'b -> 'a) -> 'a -> 'b list -> 'a

val parallel_filter : ('a -> bool) -> 'a list -> 'a list
val memoize : ('a -> 'b) -> 'a -> 'b
val parallel_map : ('a -> 'b) -> 'a list -> 'b list
val time_it : ('a -> 'b) -> 'a -> 'b * float
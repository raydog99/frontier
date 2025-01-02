type node = { id: int; address: string }
type cluster = { nodes: node list; master: node }

val create_cluster : string -> string list -> cluster
val distribute_work : cluster -> 'a list -> (node * 'a) list
val collect_results : (node * (unit -> 'a)) list -> (node * 'a) list Lwt.t
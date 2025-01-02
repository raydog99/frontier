module Node : sig
  type t = int
  val compare : t -> t -> int
end

module NodeSet : Set.S with type elt = Node.t
module NodeMap : Map.S with type key = Node.t

type t = {
  nodes: NodeSet.t;
  parents: NodeSet.t NodeMap.t;
  max_depth: int;
  max_in_degree: int;
  effective_max_in_degree: int;
  effective_max_depth: int;
}

val create : NodeSet.t -> NodeSet.t NodeMap.t -> t
val get_parents : t -> Node.t -> NodeSet.t
val validate_topological_order : NodeSet.t NodeMap.t -> bool
val compute_effective_bounds : t -> Node.t -> int * int
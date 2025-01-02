type t

val create : FunctionallyGeneratedPortfolio.t list -> t
val update : t -> Market.t list -> int -> t
val get_distribution : t -> (Portfolio.t * float) list
val covers_universal_portfolio : t -> Market.t list -> int -> Portfolio.t
val wealth_concentration : t -> int
val wealth_entropy : t -> float
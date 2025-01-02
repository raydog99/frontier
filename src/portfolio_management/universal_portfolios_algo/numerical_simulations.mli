val simulate_market : float array -> float -> float -> float -> float -> int -> int -> Market.t list list
val verify_asymptotic_universality : FunctionallyGeneratedPortfolio.t list -> Market.t list list -> bool
val verify_relative_arbitrage : Market.t list list -> float -> bool
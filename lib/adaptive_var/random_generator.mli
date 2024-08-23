module type Generator = sig
  val uniform : unit -> float
  val normal : unit -> float
end

module DefaultGenerator : Generator
module MersenneTwister : Generator
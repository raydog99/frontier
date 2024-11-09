open Utils

type word = int list

val apply_word : Measure.t -> word -> float -> float

module PrefractalGraph : sig
  type vertex = {
    position: float;
    level: int;
    word: int list;
  }
  type t
  val make : Measure.t -> int -> t
end

module HarmonicSpline : sig
  type t
  type basis_function
  val make : Grid.t -> int -> t
  val evaluate : t -> int -> float -> float
end
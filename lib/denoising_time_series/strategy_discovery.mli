open Config
open Data_processing
open Mutual_learning

type t

val create : Config.t -> t
val run : t -> (float array array * float array, string) result
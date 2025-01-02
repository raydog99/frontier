open Config
open Multi_asset_twma

type t

val create : Config.t -> Multi_asset_twma.t -> t
val run : t -> float array array -> float list
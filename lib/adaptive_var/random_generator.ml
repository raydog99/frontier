module type Generator = sig
  val uniform : unit -> float
  val normal : unit -> float
end

module DefaultGenerator : Generator = struct
  let uniform () = Random.float 1.0
  let normal () =
    let u1 = Random.float 1.0 in
    let u2 = Random.float 1.0 in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end

module MersenneTwister : Generator = struct
  let state = Random.State.make_self_init ()
  let uniform () = Random.State.float state 1.0
  let normal () =
    let u1 = Random.State.float state 1.0 in
    let u2 = Random.State.float state 1.0 in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end
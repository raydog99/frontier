open Torch

module type Utility = sig
  val evaluate : Tensor.t -> Tensor.t
  val derivative : Tensor.t -> Tensor.t
end

module PowerUtility = struct
  let create p =
    let module M = struct
      let evaluate x = Tensor.(pow x (float_vec [|p|]))
      let derivative x = Tensor.(pow x (float_vec [|p -. 1.|]) * (float_vec [|p|]))
    end in
    (module M : Utility)
end

module LogUtility = struct
  let create () =
    let module M = struct
      let evaluate x = Tensor.log x
      let derivative x = Tensor.reciprocal x
    end in
    (module M : Utility)
end

module ExponentialUtility = struct
  let create alpha =
    let module M = struct
      let evaluate x = Tensor.(neg (exp (neg (float_vec [|alpha|] * x))) / float_vec [|alpha|])
      let derivative x = Tensor.(exp (neg (float_vec [|alpha|] * x)))
    end in
    (module M : Utility)
end

module SShapedUtility = struct
  let create u1 u2 k =
    let module M = struct
      let evaluate x =
        let module U1 = (val u1 : Utility) in
        let module U2 = (val u2 : Utility) in
        Tensor.(where (x >= float_vec [|0.|])
                  (U1.evaluate x)
                  (neg (Tensor.mul_scalar (U2.evaluate (neg x)) k)))

      let derivative x =
        let module U1 = (val u1 : Utility) in
        let module U2 = (val u2 : Utility) in
        Tensor.(where (x >= float_vec [|0.|])
                  (U1.derivative x)
                  (Tensor.mul_scalar (U2.derivative (neg x)) k))
    end in
    (module M : Utility)
end

module ConcaveEnvelope = struct
  let create u1 u2 k =
    let module U1 = (val u1 : Utility) in
    let module U2 = (val u2 : Utility) in
    
    let rec find_eta r =
      let f eta = 
        Tensor.float_value (U1.evaluate (Tensor.of_float1 [|eta -. r|])) +. 
        k *. Tensor.float_value (U2.evaluate (Tensor.of_float1 [|r|])) -. 
        eta *. Tensor.float_value (U1.derivative (Tensor.of_float1 [|eta -. r|]))
      in
      let rec binary_search low high =
        if high -. low < 1e-6 then (low +. high) /. 2.
        else
          let mid = (low +. high) /. 2. in
          if f mid > 0. then binary_search low mid
          else binary_search mid high
      in
      binary_search r (r +. 100.)
    in

    let module M = struct
      let evaluate x =
        let r = Tensor.float_value x in
        let eta = find_eta r in
        Tensor.(where (x < float_vec [|eta|])
                  (neg (U2.evaluate x) + (x * U1.derivative (float_vec [|eta -. r|])))
                  (U1.evaluate (x - float_vec [|r|])))

      let derivative x =
        let r = Tensor.float_value x in
        let eta = find_eta r in
        Tensor.(where (x < float_vec [|eta|])
                  (U1.derivative (float_vec [|eta -. r|]))
                  (U1.derivative (x - float_vec [|r|])))
    end in
    (module M : Utility)
end
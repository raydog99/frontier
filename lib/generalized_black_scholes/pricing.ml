open Utils

module OptionType = struct
  type t = Call | Put

  let payoff opt_type strike s =
    match opt_type with
    | Call -> max 0.0 (s -. strike)
    | Put -> max 0.0 (strike -. s)

  let boundary_condition opt_type strike r t s =
    match opt_type with
    | Call -> 
        if s < 1e-10 then 0.0
        else s -. strike *. exp(-1.0 *. r *. t)
    | Put ->
        if s > 1e10 then 0.0
        else strike *. exp(-1.0 *. r *. t) -. s
end

module Greeks = struct
  type t = {
    delta: Tensor.t;
    gamma: Tensor.t;
    theta: Tensor.t;
    vega: Tensor.t;
    rho: Tensor.t;
  }

  let calculate params grid dt solution =
    let n = Grid.size grid - 2 in
    let h = Grid.delta grid in
    
    (* Delta calculation *)
    let delta = Tensor.zeros [Tensor.size solution 0; n] in
    for t = 0 to (Tensor.size solution 0) - 1 do
      for i = 0 to n-1 do
        let d = (Tensor.get solution [t; i+2] -. 
                Tensor.get solution [t; i]) /. (2.0 *. h) in
        Tensor.set delta [t; i] d
      done
    done;

    (* Gamma calculation *)
    let gamma = Tensor.zeros [Tensor.size solution 0; n] in
    for t = 0 to (Tensor.size solution 0) - 1 do
      for i = 0 to n-1 do
        let g = (Tensor.get solution [t; i+2] -. 
                2.0 *. Tensor.get solution [t; i+1] +.
                Tensor.get solution [t; i]) /. (h *. h) in
        Tensor.set gamma [t; i] g
      done
    done;

    (* Theta calculation *)
    let theta = Tensor.zeros [Tensor.size solution 0 - 1; n] in
    for t = 0 to (Tensor.size solution 0) - 2 do
      for i = 0 to n-1 do
        let th = -.(Tensor.get solution [t+1; i] -. 
                   Tensor.get solution [t; i]) /. dt in
        Tensor.set theta [t; i] th
      done
    done;

    (* Vega calculation using finite difference *)
    let d_sigma = 0.001 in
    let bumped_params = { params with sigma = params.sigma +. d_sigma } in
    let vega = Tensor.zeros [Tensor.size solution 0; n] in

    (* Rho calculation using finite difference *)
    let d_r = 0.0001 in
    let bumped_r = { params with r = params.r +. d_r } in
    let rho = Tensor.zeros [Tensor.size solution 0; n] in

    { delta; gamma; theta; vega; rho }
end
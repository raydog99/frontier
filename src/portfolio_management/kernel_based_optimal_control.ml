open Torch

type state = Tensor.t
type control = Tensor.t
type time = float

module Dataset = struct
  type t = {
    states: Tensor.t;
    derivatives: Tensor.t;
    controls: Tensor.t option;
  }

  let create ~states ~derivatives ?controls () = {
    states;
    derivatives;
    controls;
  }
end

module Measure = struct
  type t = {
    density: Tensor.t -> float Tensor.t;
    support: (float * float) array;
  }

  let create ~density ~support = {
    density;
    support;
  }

  let in_support t x =
    let dims = Tensor.size x 0 in
    let rec check_dims i =
      if i >= dims then true
      else
        let xi = Tensor.get x [|i|] in
        let (lower, upper) = t.support.(i) in
        if xi >= lower && xi <= upper then
          check_dims (i + 1)
        else
          false
    in
    check_dims 0

  let evaluate t x =
    if in_support t x then
      t.density x
    else
      Tensor.zeros []
end

let gaussian_k ~sigma x y =
  let diff = Tensor.(x - y) in
  let sqnorm = Tensor.(sum (diff * diff)) in
  Tensor.exp (Tensor.(~-sqnorm / (2.0 *. sigma *. sigma)))

let gaussian_grad_k ~sigma x y =
  let diff = Tensor.(x - y) in
  let k_val = gaussian_k ~sigma x y in
  Tensor.(k_val * diff / (~-.(sigma *. sigma)))

let gaussian_hessian_k ~sigma x y =
  let diff = Tensor.(x - y) in
  let k_val = gaussian_k ~sigma x y in
  let outer = Tensor.mm (Tensor.unsqueeze diff 1) (Tensor.unsqueeze diff 0) in
  let identity = Tensor.eye (Tensor.size diff).[0] in
  Tensor.((k_val / (sigma *. sigma)) * (outer / (sigma *. sigma) - identity))

let third_derivative_k ~sigma x y =
  let diff = Tensor.(x - y) in
  let k_val = gaussian_k ~sigma x y in
  let hess = gaussian_hessian_k ~sigma x y in
  let scale = 1.0 /. (sigma *. sigma *. sigma) in
  Tensor.(k_val * scale * diff)

let mixed_derivatives_k ~sigma x y =
  let nx = Tensor.size x 0 in
  Array.init nx (fun i ->
    let ei = Tensor.zeros [nx] in
    Tensor.set ei [|i|] 1.0;
    Tensor.gradient (gaussian_hessian_k ~sigma x y) [ei]
  )

module Dynamics = struct
  type t = {
    nx: int;
    nu: int;
    drift: Tensor.t -> Tensor.t;
    diffusion: float;
  }

  let create ~nx ~nu ~drift ~diffusion = {
    nx;
    nu;
    drift;
    diffusion;
  }
end

let control_matrix x = Tensor.zeros [1]

let controlled_drift x u =
  let drift = Dynamics.create
    ~nx:(Tensor.size x 0)
    ~nu:(Tensor.size u 0)
    ~drift:(fun _ -> Tensor.zeros [1])
    ~diffusion:0.0 in
  Tensor.(drift.drift x + matmul (control_matrix x) u)

module Cost = struct
  type t = {
    stage_cost: Tensor.t -> float Tensor.t;
    control_penalty: Tensor.t -> float Tensor.t;
  }

  let create ~stage_cost ~control_penalty = {
    stage_cost;
    control_penalty;
  }
end

module HilbertOperator = struct
  type t = {
    forward: Tensor.t -> Tensor.t;
    adjoint: Tensor.t -> Tensor.t;
    domain_dim: int;
    range_dim: int;
  }

  let create ~forward ~adjoint ~domain_dim ~range_dim = {
    forward;
    adjoint;
    domain_dim;
    range_dim;
  }

  let compose a b =
    if a.domain_dim <> b.range_dim then
      invalid_arg "Incompatible operator dimensions";
    {
      forward = (fun x -> a.forward (b.forward x));
      adjoint = (fun x -> b.adjoint (a.adjoint x));
      domain_dim = b.domain_dim;
      range_dim = a.range_dim;
    }
end

module GeneratorOperator = struct
  type t = {
    sigma: float;  (* For gaussian kernel *)
    epsilon: float;
  }

  let create ~epsilon = {
    sigma = 1.0;  (* Default value *)
    epsilon;
  }

  let forward t x =
    let grad = gaussian_grad_k ~sigma:t.sigma x x in
    let hess = gaussian_hessian_k ~sigma:t.sigma x x in
    Tensor.(grad + (t.epsilon * trace hess))

  let adjoint t x =
    let grad = gaussian_grad_k ~sigma:t.sigma x x in
    let hess = gaussian_hessian_k ~sigma:t.sigma x x in
    Tensor.(neg grad + (t.epsilon * trace hess))
end

module FPK = struct
  type t = {
    sigma: float;
    dynamics: Dynamics.t;
    epsilon: float;
  }

  let create ~dynamics ~epsilon = {
    sigma = 1.0;
    dynamics;
    epsilon;
  }

  let forward_evolution t density dt =
    let n = Tensor.size density 0 in
    let fpk_op = Tensor.zeros [n; n] in

    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let xi = Tensor.select density 0 i in
        let xj = Tensor.select density 0 j in
        
        let drift = t.dynamics.drift xi in
        let grad_k = gaussian_grad_k ~sigma:t.sigma xi xj in
        let hess_k = gaussian_hessian_k ~sigma:t.sigma xi xj in
        
        let drift_term = Tensor.(sum (grad_k * drift)) in
        let diff_term = t.epsilon *. Tensor.(trace hess_k |> to_float0_exn) in
        
        Tensor.set fpk_op [|i; j|] (Tensor.to_float0_exn drift_term +. diff_term)
      done
    done;
    
    let lhs = Tensor.(eye n + (dt * fpk_op)) in
    Tensor.mm (Tensor.inverse lhs) density

  let backward_evolution t density dt =
    let n = Tensor.size density 0 in
    let adj_op = Tensor.zeros [n; n] in

    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let xi = Tensor.select density 0 i in
        let xj = Tensor.select density 0 j in
        
        let drift = t.dynamics.drift xi in
        let grad_k = gaussian_grad_k ~sigma:t.sigma xi xj in
        let hess_k = gaussian_hessian_k ~sigma:t.sigma xi xj in
        
        let drift_term = Tensor.(sum (neg grad_k * drift)) in
        let diff_term = t.epsilon *. Tensor.(trace hess_k |> to_float0_exn) in
        
        Tensor.set adj_op [|i; j|] (Tensor.to_float0_exn drift_term +. diff_term)
      done
    done;
    
    let lhs = Tensor.(eye n + (dt * adj_op)) in
    Tensor.mm (Tensor.inverse lhs) density
end

module HJB = struct
  type t = {
    fpk: FPK.t;
    cost: Cost.t;
    final_time: float;
    dt: float;
  }

  let create ~fpk ~cost ~final_time ~dt = {
    fpk;
    cost;
    final_time;
    dt;
  }

  let solve t initial_density =
    let n_steps = int_of_float (t.final_time /. t.dt) in
    let density = ref initial_density in
    let value_fn = ref (Tensor.zeros_like initial_density) in
    
    for i = 0 to n_steps - 1 do
      (* Forward FPK step *)
      density := FPK.forward_evolution t.fpk !density t.dt;
      
      (* Backward HJB step *)
      let cost = t.cost.stage_cost !density in
      value_fn := FPK.backward_evolution t.fpk (Tensor.(cost + !value_fn)) t.dt
    done;
    
    (!value_fn, !density)
end

module GeneratorRegression = struct
  type t = {
    sigma: float;
    epsilon: float;
    reg_param: float;
  }

  let create ~epsilon ~reg_param = {
    sigma = 1.0;
    epsilon;
    reg_param;
  }

  let learn_generator t data test_x =
    let n = Tensor.size data.Dataset.states 0 in
    
    (* Compute Gram matrix *)
    let gram = Tensor.zeros [n; n] in
    for i = 0 to n-1 do
      let xi = Tensor.select data.states 0 i in
      for j = 0 to n-1 do
        let xj = Tensor.select data.states 0 j in
        let kij = gaussian_k ~sigma:t.sigma xi xj |> Tensor.to_float0_exn in
        Tensor.set gram [|i; j|] kij
      done
    done;
    
    (* Add regularization *)
    let reg_gram = Tensor.(gram + (t.reg_param * eye n)) in
    
    (* Compute coefficients *)
    let coeffs = Tensor.mm (Tensor.inverse reg_gram) data.derivatives in
    
    (* Evaluate at test point *)
    let k_test = Tensor.zeros [n] in
    for i = 0 to n-1 do
      let xi = Tensor.select data.states 0 i in
      let ki = gaussian_k ~sigma:t.sigma test_x xi |> Tensor.to_float0_exn in
      Tensor.set k_test [|i|] ki
    done;
    
    Tensor.mv coeffs k_test

  let learn_controlled_generator t data =
    match data.Dataset.controls with
    | None -> invalid_arg "Control data required"
    | Some controls ->
        let autonomous_gen = learn_generator t data in
        let n = Tensor.size data.states 0 in
        let nu = Tensor.size controls 1 in
        
        let control_gens = Array.init nu (fun i ->
          let control_data = Dataset.create
            ~states:data.states
            ~derivatives:Tensor.(select controls 1 i |> unsqueeze 1)
            () in
          learn_generator t control_data
        ) in
        
        (autonomous_gen, fun x -> Array.map (fun g -> g x) control_gens)
end

module DualityMeasure = struct
  type dual_pair = {
    primal: Measure.t;
    dual: Measure.t;
    pairing: Tensor.t -> Tensor.t -> float Tensor.t;
  }

  let create_dual_pair ~primal ~dual ~pairing = {
    primal;
    dual;
    pairing;
  }

  let verify_strong_duality t =
    let test_points = Tensor.randn [100; 1] in
    let max_gap = ref 0.0 in
    
    Tensor.iter (fun x ->
      let y = Tensor.randn [1] in
      let gap = t.pairing x y |> Tensor.to_float0_exn |> abs_float in
      max_gap := max !max_gap gap
    ) test_points;
    
    if !max_gap < 1e-6 then
      Ok ()
    else
      Error "Duality gap too large"

  let optimal_values t primal dual =
    let pv = t.pairing primal dual in
    let dv = t.pairing dual primal in
    let gap = Tensor.(abs (pv - dv)) in
    (pv, dv, gap)
end
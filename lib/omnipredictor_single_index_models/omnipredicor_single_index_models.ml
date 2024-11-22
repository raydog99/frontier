open Torch

module ModelParams = struct
  type t = {
    d: int;                (* Dimension *)
    l_bar: float;          (* Lower bound on second moment *)
    l: float;              (* Upper bound on features *)
    r: float;              (* Radius of weight vectors *)
    eps: float;            (* Error parameter *)
    eta: float;            (* Learning rate *)
  }

  let create ~d ~l_bar ~l ~r ~eps ~eta =
    if d <= 0 then invalid_arg "d must be positive";
    if l_bar <= 0. || l <= 0. || r <= 0. then 
      invalid_arg "bounds must be positive";
    if l_bar > l then invalid_arg "l_bar must be <= l";
    if eps <= 0. || eps >= 1. then invalid_arg "eps must be in (0,1)";
    if eta <= 0. then invalid_arg "eta must be positive";
    {d; l_bar; l; r; eps; eta}

  let scale_invariant t =
    {t with 
     l = t.l /. t.r;
     l_bar = t.l_bar /. t.r;
     r = 1.0}
end

module Utils = struct
  let l2_project tensor radius =
    let norm = Tensor.norm tensor ~p:(Scalar 2) ~dim:[0] ~keepdim:true in
    let scale = Tensor.min (Tensor.ones_like norm) 
                 (Tensor.div (Tensor.full_like norm radius) norm) in
    Tensor.mul tensor scale

  let squared_divergence pred target =
    let diff = Tensor.sub pred target in
    Tensor.mean (Tensor.mul diff diff)

  let gradient_norm grad =
    Tensor.float_value (Tensor.norm grad ~p:(Scalar 2))

  let clamp tensor ~min ~max =
    Tensor.min (Tensor.max tensor (Tensor.full_like tensor min))
               (Tensor.full_like tensor max)
end

module type LinkFunction = sig
  type t
  val create : ?alpha:float -> ?beta:float -> unit -> t
  val apply : t -> Tensor.t -> Tensor.t
  val inverse : t -> Tensor.t -> Tensor.t
  val derivative : t -> Tensor.t -> Tensor.t
  val is_lipschitz : t -> float -> bool
  val is_anti_lipschitz : t -> bool
  val integrate : t -> Tensor.t -> Tensor.t -> Tensor.t
end

module LogisticLink : LinkFunction = struct
  type t = {
    alpha: float;  (* Anti-Lipschitz parameter *)
    beta: float;   (* Lipschitz parameter *)
  }

  let create ?(alpha=0.25) ?(beta=0.25) () = {alpha; beta}

  let apply t x =
    let open Tensor in
    let neg_x = neg x in
    div ones_like x (add (ones_like x) (exp neg_x))

  let inverse t y =
    let open Tensor in
    let y_clip = Utils.clamp y ~min:1e-6 ~max:(1.-.1e-6) in
    neg (log (div (sub (ones_like y) y_clip) y_clip))

  let derivative t x =
    let open Tensor in
    let sig_x = apply t x in
    mul sig_x (sub (ones_like sig_x) sig_x)

  let is_lipschitz t beta = beta >= t.beta
  let is_anti_lipschitz t = true

  let integrate t start stop =
    let open Tensor in
    let primitive x = 
      let sig_x = apply t x in
      add (mul x sig_x) (log (add (ones_like x) (exp (neg x))))
    in
    sub (primitive stop) (primitive start)
end

module ReLULink : LinkFunction = struct
  type t = {
    alpha: float;
    beta: float;
  }

  let create ?(alpha=1.0) ?(beta=1.0) () = {alpha; beta}

  let apply t x =
    Tensor.(clamp (relu x) ~min:0. ~max:1.)

  let inverse t y =
    Utils.clamp y ~min:0. ~max:1.

  let derivative t x =
    let open Tensor in
    map (fun v -> if v > 0. then 1. else 0.) x

  let is_lipschitz t beta = beta >= t.beta
  let is_anti_lipschitz t = true

  let integrate t start stop =
    let open Tensor in
    let primitive x =
      let activated = relu x in
      mul activated activated |> div_scalar 2.
    in
    sub (primitive stop) (primitive start)
end

module DivergenceMeasures = struct
  type divergence = {
    compute: Tensor.t -> Tensor.t -> Tensor.t;
    gradient: Tensor.t -> Tensor.t -> Tensor.t;
    is_proper: bool;
  }

  let matching_divergence (link: LinkFunction.t) = 
    let compute pred target =
      let open Tensor in
      let integral = link.integrate (zeros_like pred) pred in
      sub integral (mul pred target)
    in
    let gradient pred target =
      let open Tensor in
      sub (link.apply pred) target
    in
    {compute; gradient; is_proper = false}

  let proper_divergence (link: LinkFunction.t) =
    let compute pred target =
      let open Tensor in
      let unlinked = link.inverse pred in
      let matched = matching_divergence link in
      matched.compute unlinked target
    in
    let gradient pred target =
      let open Tensor in
      let unlinked = link.inverse pred in
      let matched = matching_divergence link in
      let deriv = link.derivative pred in
      mul (matched.gradient unlinked target) deriv
    in
    {compute; gradient; is_proper = true}

  let omnigap_divergence ~pred_link ~target_link =
    let compute pred target = 
      let open Tensor in
      let linked = pred_link.apply pred in
      let diff = sub linked target in
      let inverted = target_link.inverse linked in
      mean (mul diff inverted)
    in
    {compute; gradient = compute; is_proper = false}
end

module IsotonicRegression = struct
  let solve y =
    let n = Tensor.size y 0 in
    let out = Tensor.clone y in
    let rec merge_pools start =
      if start >= n - 1 then out
      else
        let v1 = Tensor.get out start in
        let v2 = Tensor.get out (start + 1) in
        if v1 <= v2 then merge_pools (start + 1)
        else begin
          let avg = (v1 +. v2) /. 2. in
          Tensor.set out start avg;
          Tensor.set out (start + 1) avg;
          if start > 0 then merge_pools (start - 1)
          else merge_pools (start + 1)
        end
    in
    merge_pools 0

  let fit_link (link: LinkFunction.t) w x y =
    let open Tensor in
    let wx = mm x w in
    let sorted_indices = argsort wx 0 in
    let sorted_y = index_select y 0 sorted_indices in
    let fitted = solve sorted_y in
    let unsorted_indices = argsort sorted_indices 0 in
    index_select fitted 0 unsorted_indices
end

module SIM (P : sig val params : ModelParams.t end) = struct
  type t = {
    weights: Tensor.t;
    link: LogisticLink.t;
    divergence: DivergenceMeasures.divergence;
    iteration: int;
    momentum: Tensor.t option;
  }

  let create () = {
    weights = Tensor.zeros [P.params.d];
    link = LogisticLink.create ();
    divergence = DivergenceMeasures.proper_divergence (LogisticLink.create ());
    iteration = 0;
    momentum = None;
  }

  let predict t x =
    let linear = Tensor.mm x t.weights in
    t.link.apply linear

  let divergence t x y =
    t.divergence.compute (predict t x) y

  let gradient t x y =
    let pred = predict t x in
    t.divergence.gradient pred y

  let update ?(use_momentum=true) t grad =
    let open Tensor in
    let new_momentum = match t.momentum, use_momentum with
      | Some m, true -> 
          Some (add (mul_scalar m 0.9) grad)
      | _, _ -> None in
    
    let update_vec = match new_momentum with
      | Some m -> m
      | None -> grad in
    
    let new_weights = sub t.weights (mul_scalar update_vec P.params.eta) in
    let projected = Utils.l2_project new_weights P.params.r in
    
    {t with 
     weights = projected;
     momentum = new_momentum;
     iteration = t.iteration + 1}
end

module Isotron (P : sig val params : ModelParams.t end) = struct
  module Base = SIM(P)
  
  type t = {
    model: Base.t;
    best_weights: Tensor.t option;
    best_divergence: float;
    total_iter: int;
  }

  let create () = {
    model = Base.create ();
    best_weights = None;
    best_divergence = Float.infinity;
    total_iter = 0;
  }

  let step t x y =
    let fitted = IsotonicRegression.fit_link 
      t.model.link t.model.weights x y in
    
    let grad = t.model.gradient x fitted in
    let model' = Base.update t.model grad in
    
    let curr_divergence = Tensor.float_value (Base.divergence model' x y) in
    let (best_weights, best_divergence) =
      if curr_divergence < t.best_divergence then
        (Some model'.weights, curr_divergence)
      else
        (t.best_weights, t.best_divergence)
    in
    
    {model = model';
     best_weights;
     best_divergence;
     total_iter = t.total_iter + 1}

  let train ?(max_iter=100) t x y =
    let rec loop t iter =
      if iter >= max_iter then t
      else
        let t' = step t x y in
        loop t' (iter + 1)
    in
    loop t 0

  let predict t = Base.predict t.model
end

module MultiIndexModel (P : sig val params : ModelParams.t end) = struct
  type predictor = {
    weights: Tensor.t;
    link: LogisticLink.t;
    iteration: int;
  }

  type t = {
    predictors: predictor list;
    max_models: int;
    total_iter: int;
  }

  let create ?(max_models=100) () = {
    predictors = [];
    max_models;
    total_iter = 0;
  }

  let add_predictor t weights =
    let pred = {
      weights;
      link = LogisticLink.create ();
      iteration = t.total_iter;
    } in
    if List.length t.predictors >= t.max_models then
      {t with 
       predictors = pred :: List.tl t.predictors;
       total_iter = t.total_iter + 1}
    else
      {t with 
       predictors = pred :: t.predictors;
       total_iter = t.total_iter + 1}

  let predict t x target_link =
    let open Tensor in
    match t.predictors with
    | [] -> zeros [size x 0]
    | preds ->
        let predictions = List.map (fun p ->
          let raw = mm x p.weights in
          let linked = p.link.apply raw in
          target_link.inverse linked
        ) preds in
        let n = float_of_int (List.length predictions) in
        List.fold_left (fun acc p ->
          add acc (div p n)
        ) (zeros_like (List.hd predictions)) predictions
end

module Omnitron (P : sig val params : ModelParams.t end) = struct
  module Base = Isotron(P)
  module Model = MultiIndexModel(P)

  type t = {
    base: Base.t;
    model: Model.t;
    stats: stats option;
  }
  and stats = {
    empirical_error: float;
    population_error: float option;
    sample_complexity: int;
    iteration_complexity: int;
  }

  let create ?(max_models=100) () = {
    base = Base.create ();
    model = Model.create ~max_models ();
    stats = None;
  }

  let step t x y =
    let base' = Base.step t.base x y in
    
    let model' = Model.add_predictor t.model base'.model.weights in
    
    {t with base = base'; model = model'}

  let train ?(max_iter=100) t x y =
    let rec loop t iter =
      if iter >= max_iter then t
      else
        let t' = step t x y in
        loop t' (iter + 1)
    in
    loop t 0

  let predict t x target_link =
    Model.predict t.model x target_link

  let omnigap t x y target_link target_weights =
    let pred = predict t x target_link in
    let divergence = DivergenceMeasures.omnigap_divergence 
      ~pred_link:t.base.model.link
      ~target_link in
    divergence.compute pred y
end
type option_type = Call | Put

type style = European | American | Bermudan of float list

type barrier_type = UpAndOut | UpAndIn | DownAndOut | DownAndIn

type exotic_type =
  | Vanilla
  | Barrier of barrier_type * float
  | AsianFixed of int
  | Lookback

type t = {
  tree: Tree.t;
  option_type: option_type;
  strike: float;
  style: style;
  exotic: exotic_type;
}

type multi_asset_option = {
  trees: Tree.multi_asset_t;
  option_type: option_type;
  strike: float;
  style: style;
  exotic: exotic_type;
}

let create tree option_type strike style exotic =
  { tree; option_type; strike; style; exotic }

let create_multi_asset trees option_type strike style exotic =
  { trees; option_type; strike; style; exotic }

let payoff option s =
  match option.option_type with
  | Call -> max 0. (s -. option.strike)
  | Put -> max 0. (option.strike -. s)

let is_barrier_breached barrier_type barrier_level s =
  match barrier_type with
  | UpAndOut -> s >= barrier_level
  | UpAndIn -> s < barrier_level
  | DownAndOut -> s <= barrier_level
  | DownAndIn -> s > barrier_level

let price_vanilla option =
  let tree = option.tree in
  let n = tree.n in
  let df = Tree.discount_factor tree in
  let qu, qm, qd = Tree.risk_neutral_probabilities tree in
  
  let values = Array.make (n + 1) 0. in
  for j = 0 to n do
    let s = Tree.stock_price tree j (n - j) in
    values.(j) <- payoff option s
  done;

  for i = n - 1 downto 0 do
    for j = 0 to i do
      let s = Tree.stock_price tree j (i - j) in
      let v_up = values.(j)
      and v_mid = values.(j + 1)
      and v_down = values.(j + 2) in
      let continuation_value = qu *. v_up +. qm *. v_mid +. qd *. v_down in
      values.(j) <- match option.style with
        | European -> df *. continuation_value
        | American -> max (payoff option s) (df *. continuation_value)
        | Bermudan dates ->
            if List.mem (float_of_int i *. tree.dt) dates
            then max (payoff option s) (df *. continuation_value)
            else df *. continuation_value
    done
  done;

  values.(0)

let price_barrier option barrier_type barrier_level =
  let tree = option.tree in
  let n = tree.n in
  let df = Tree.discount_factor tree in
  let qu, qm, qd = Tree.risk_neutral_probabilities tree in
  
  let values = Array.make (n + 1) 0. in
  for j = 0 to n do
    let s = Tree.stock_price tree j (n - j) in
    values.(j) <- if is_barrier_breached barrier_type barrier_level s
                  then payoff option s
                  else 0.
  done;

  for i = n - 1 downto 0 do
    for j = 0 to i do
      let s = Tree.stock_price tree j (i - j) in
      if is_barrier_breached barrier_type barrier_level s then
        let v_up = values.(j)
        and v_mid = values.(j + 1)
        and v_down = values.(j + 2) in
        let continuation_value = qu *. v_up +. qm *. v_mid +. qd *. v_down in
        values.(j) <- df *. continuation_value
      else
        values.(j) <- 0.
    done
  done;

  values.(0)

let price_asian_fixed option fixing_dates =
  let tree = option.tree in
  let n = tree.n in
  let df = Tree.discount_factor tree in
  let qu, qm, qd = Tree.risk_neutral_probabilities tree in
  
  let values = Array.make_matrix (n + 1) (fixing_dates + 1) 0. in
  for j = 0 to n do
    let s = Tree.stock_price tree j (n - j) in
    for k = 0 to fixing_dates do
      values.(j).(k) <- payoff option (s *. float_of_int k /. float_of_int fixing_dates)
    done
  done;

  for i = n - 1 downto 0 do
    for j = 0 to i do
      let s = Tree.stock_price tree j (i - j) in
      for k = 0 to fixing_dates do
        let v_up = values.(j).(k)
        and v_mid = values.(j + 1).(k)
        and v_down = values.(j + 2).(k) in
        let continuation_value = qu *. v_up +. qm *. v_mid +. qd *. v_down in
        if i mod (n / fixing_dates) = 0 && k < fixing_dates then
          values.(j).(k + 1) <- df *. continuation_value
        else
          values.(j).(k) <- df *. continuation_value
      done
    done
  done;

  values.(0).(0)

let price_lookback option =
  let tree = option.tree in
  let n = tree.n in
  let df = Tree.discount_factor tree in
  let qu, qm, qd = Tree.risk_neutral_probabilities tree in
  
  let values = Array.make_matrix (n + 1) (n + 1) 0. in
  for j = 0 to n do
    let s = Tree.stock_price tree j (n - j) in
    for k = 0 to j do
      let max_s = Tree.stock_price tree k 0 in
      values.(j).(k) <- payoff option (max s max_s)
    done
  done;

  for i = n - 1 downto 0 do
    for j = 0 to i do
      let s = Tree.stock_price tree j (i - j) in
      for k = 0 to j do
        let max_s = Tree.stock_price tree k 0 in
        let v_up = values.(j).(max k (j + 1))
        and v_mid = values.(j + 1).(max k (j + 1))
        and v_down = values.(j + 2).(max k (j + 2)) in
        let continuation_value = qu *. v_up +. qm *. v_mid +. qd *. v_down in
        values.(j).(k) <- df *. continuation_value
      done
    done
  done;

  values.(0).(0)

let price option =
  match option.exotic with
  | Vanilla -> price_vanilla option
  | Barrier (barrier_type, barrier_level) -> price_barrier option barrier_type barrier_level
  | AsianFixed fixing_dates -> price_asian_fixed option fixing_dates
  | Lookback -> price_lookback option

let price_with_control_variate option =
  let analytical_price = Black_Scholes.european_option_price option.tree.s0 option.strike option.tree.r option.tree.sigma option.tree.t option.option_type in
  let tree_price = price option in
  let control_variate = analytical_price -. tree_price in
  tree_price +. control_variate

let price_with_importance_sampling option num_samples =
  let importance_sample () =
    let u = Random.float 1. in
    option.tree.s0 *. exp((option.tree.r -. 0.5 *. option.tree.sigma ** 2.) *. option.tree.t +. 
                          option.tree.sigma *. sqrt(option.tree.t) *. sqrt(-2. *. log u) *. cos(2. *. Float.pi *. Random.float 1.))
  in
  let payoffs = Array.init num_samples (fun _ -> 
    let s_T = importance_sample () in
    payoff option s_T
  ) in
  let mean_payoff = Array.fold_left (+.) 0. payoffs /. float_of_int num_samples in
  exp(-. option.tree.r *. option.tree.t) *. mean_payoff

let delta option s t =
  let h = 0.01 *. s in
  let v_up = price { option with tree = { option.tree with s0 = s +. h } }
  and v_down = price { option with tree = { option.tree with s0 = s -. h } } in
  (v_up -. v_down) /. (2. *. h)

let gamma option s t =
  let h = 0.01 *. s in
  let v_up = price { option with tree = { option.tree with s0 = s +. h } }
  and v = price option
  and v_down = price { option with tree = { option.tree with s0 = s -. h } } in
  (v_up -. 2. *. v +. v_down) /. (h ** 2.)

let theta option s t =
  let h = 0.01 *. option.tree.t in
  let v_future = price { option with tree = { option.tree with t = t +. h } } in
  (v_future -. price option) /. h

let vega option s t =
  let h = 0.01 *. option.tree.sigma in
  let v_up = price { option with tree = { option.tree with sigma = option.tree.sigma +. h } }
  and v_down = price { option with tree = { option.tree with sigma = option.tree.sigma -. h } } in
  (v_up -. v_down) /. (2. *. h)

let rho option s t =
  let h = 0.01 *. option.tree.r in
  let v_up = price { option with tree = { option.tree with r = option.tree.r +. h } }
  and v_down = price { option with tree = { option.tree with r = option.tree.r -. h } } in
  (v_up -. v_down) /. (2. *. h)

let finite_difference_greeks option =
  let s = option.tree.s0 in
  let t = option.tree.t in
  let h_s = 0.01 *. s in
  let h_t = 0.01 *. t in
  let h_sigma = 0.0001 in
  let h_r = 0.0001 in
  
  let price_up_s = price { option with tree = { option.tree with s0 = s +. h_s } } in
  let price_down_s = price { option with tree = { option.tree with s0 = s -. h_s } } in
  let price_up_t = price { option with tree = { option.tree with t = t +. h_t } } in
  let price_mid = price option in
  let price_up_sigma = price { option with tree = { option.tree with sigma = option.tree.sigma +. h_sigma } } in
  let price_down_sigma = price { option with tree = { option.tree with sigma = option.tree.sigma -. h_sigma } } in
  let price_up_r = price { option with tree = { option.tree with r = option.tree.r +. h_r } } in
  let price_down_r = price { option with tree = { option.tree with r = option.tree.r -. h_r } } in

  {
    delta = (price_up_s -. price_down_s) /. (2. *. h_s);
    gamma = (price_up_s -. 2. *. price_mid +. price_down_s) /. (h_s ** 2.);
    theta = (price_mid -. price_up_t) /. h_t;
    vega = (price_up_sigma -. price_down_sigma) /. (2. *. h_sigma);
    rho = (price_up_r -. price_down_r) /. (2. *. h_r);
  }
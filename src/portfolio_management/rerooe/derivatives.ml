type option_type = Call | Put

type t =
  | EuropeanOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
    }
  | AmericanOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
    }
  | AsianOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
      averaging_period: float;
    }

let black_scholes_d1 s k r sigma t =
  (log (s /. k) +. (r +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t)

let black_scholes_d2 d1 sigma t =
  d1 -. sigma *. sqrt t

let normal_cdf x =
  0.5 *. (1. +. erf (x /. sqrt 2.))

let binomial_tree s k r sigma t steps =
  let dt = t /. float_of_int steps in
  let u = exp (sigma *. sqrt dt) in
  let d = 1. /. u in
  let p = (exp (r *. dt) -. d) /. (u -. d) in
  
  let tree = Array.make_matrix (steps + 1) (steps + 1) 0. in
  
  for i = 0 to steps do
    tree.(steps).(i) <- max 0. (s *. (u ** float_of_int (steps - i)) *. (d ** float_of_int i) -. k)
  done;
  
  for i = steps - 1 downto 0 do
    for j = 0 to i do
      let continuation_value = 
        exp (-. r *. dt) *. (p *. tree.(i+1).(j) +. (1. -. p) *. tree.(i+1).(j+1)) in
      let exercise_value = 
        max 0. (s *. (u ** float_of_int (i - j)) *. (d ** float_of_int j) -. k) in
      tree.(i).(j) <- max continuation_value exercise_value
    done
  done;
  
  tree.(0).(0)

let monte_carlo_asian s k r sigma t averaging_period num_paths num_steps =
  let dt = t /. float_of_int num_steps in
  let drift = (r -. 0.5 *. sigma ** 2.) *. dt in
  let vol = sigma *. sqrt dt in
  
  let sum_payoffs = ref 0. in
  for _ = 1 to num_paths do
    let path = Array.make (num_steps + 1) s in
    let sum_prices = ref s in
    for j = 1 to num_steps do
      let z = Random.float 1. |> Stats.gaussian_ppf ~mu:0. ~sigma:1. in
      path.(j) <- path.(j-1) *. exp (drift +. vol *. z);
      if float_of_int j >. (t -. averaging_period) *. float_of_int num_steps /. t then
        sum_prices := !sum_prices +. path.(j)
    done;
    let avg_price = !sum_prices /. (float_of_int num_steps *. averaging_period /. t +. 1.) in
    sum_payoffs := !sum_payoffs +. max 0. (avg_price -. k)
  done;
  
  exp (-. r *. t) *. !sum_payoffs /. float_of_int num_paths

let price derivative s r sigma =
  match derivative with
  | EuropeanOption { underlying=_; option_type; strike; expiry } ->
      let d1 = black_scholes_d1 s strike r sigma expiry in
      let d2 = black_scholes_d2 d1 sigma expiry in
      match option_type with
      | Call -> s *. normal_cdf d1 -. strike *. exp (-. r *. expiry) *. normal_cdf d2
      | Put -> strike *. exp (-. r *. expiry) *. normal_cdf (-. d2) -. s *. normal_cdf (-. d1)
  | AmericanOption _ ->
      let steps = 100 in  (* Number of steps in the binomial tree *)
      let price = binomial_tree s strike r sigma expiry steps in
      (match option_type with
       | Call -> price
       | Put -> price +. s -. strike *. exp (-. r *. expiry))
  | AsianOption { underlying=_; option_type; strike; expiry; averaging_period } ->
      let num_paths = 10000 in  (* Number of Monte Carlo paths *)
      let num_steps = 100 in    (* Number of time steps *)
      let price = monte_carlo_asian s strike r sigma expiry averaging_period num_paths num_steps in
      match option_type with
      | Call -> price
      | Put -> price +. s *. exp (-. r *. expiry) -. strike *. exp (-. r *. averaging_period)  (* Put-call parity for Asian options *)

let delta derivative s r sigma =
  match derivative with
  | EuropeanOption { underlying=_; option_type; strike; expiry } ->
      let d1 = black_scholes_d1 s strike r sigma expiry in
      match option_type with
      | Call -> normal_cdf d1
      | Put -> normal_cdf d1 -. 1.
  | AmericanOption _ ->
      failwith "American option delta not implemented"
  | AsianOption _ ->
      failwith "Asian option delta not implemented"

let gamma derivative s r sigma =
  match derivative with
  | EuropeanOption { underlying=_; option_type=_; strike; expiry } ->
      let d1 = black_scholes_d1 s strike r sigma expiry in
      exp (-. d1 ** 2. /. 2.) /. (s *. sigma *. sqrt expiry *. sqrt (2. *. Float.pi))
  | AmericanOption _ ->
      failwith "American option gamma not implemented"
  | AsianOption _ ->
      failwith "Asian option gamma not implemented"
let moving_average data window =
  Array.mapi (fun i _ ->
    if i < window - 1 then nan
    else
      let sum = ref 0. in
      for j = 0 to window - 1 do
        sum := !sum +. data.(i - j)
      done;
      !sum /. float_of_int window
  ) data

let exponential_moving_average data alpha =
  let ema = Array.copy data in
  for i = 1 to Array.length data - 1 do
    ema.(i) <- alpha *. data.(i) +. (1. -. alpha) *. ema.(i-1)
  done;
  ema

let bollinger_bands data window k =
  let ma = moving_average data window in
  let upper = Array.mapi (fun i x ->
    if i < window - 1 then nan
    else
      let sum_sq_diff = ref 0. in
      for j = 0 to window - 1 do
        sum_sq_diff := !sum_sq_diff +. (data.(i - j) -. x) ** 2.
      done;
      let std_dev = sqrt (!sum_sq_diff /. float_of_int window) in
      x +. k *. std_dev
  ) ma in
  let lower = Array.mapi (fun i x ->
    if i < window - 1 then nan
    else
      let sum_sq_diff = ref 0. in
      for j = 0 to window - 1 do
        sum_sq_diff := !sum_sq_diff +. (data.(i - j) -. x) ** 2.
      done;
      let std_dev = sqrt (!sum_sq_diff /. float_of_int window) in
      x -. k *. std_dev
  ) ma in
  (ma, upper, lower)

let relative_strength_index data window =
  let gains = Array.make (Array.length data) 0. in
  let losses = Array.make (Array.length data) 0. in
  for i = 1 to Array.length data - 1 do
    let diff = data.(i) -. data.(i-1) in
    if diff > 0. then gains.(i) <- diff else losses.(i) <- -.diff
  done;
  let avg_gain = exponential_moving_average gains (2. /. (float_of_int window +. 1.)) in
  let avg_loss = exponential_moving_average losses (2. /. (float_of_int window +. 1.)) in
  Array.mapi (fun i g ->
    if i < window then nan
    else
      let rs = g /. avg_loss.(i) in
      100. -. (100. /. (1. +. rs))
  ) avg_gain

let macd data short_window long_window signal_window =
  let short_ema = exponential_moving_average data (2. /. (float_of_int short_window +. 1.)) in
  let long_ema = exponential_moving_average data (2. /. (float_of_int long_window +. 1.)) in
  let macd_line = Array.map2 (-.) short_ema long_ema in
  let signal_line = exponential_moving_average macd_line (2. /. (float_of_int signal_window +. 1.)) in
  (macd_line, signal_line)
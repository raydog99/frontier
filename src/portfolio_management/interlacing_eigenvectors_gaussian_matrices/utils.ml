open Torch

let assert_positive_float name value =
  if value <= 0. then
    failwith (Printf.sprintf "%s must be positive" name)

let assert_in_range name value min max =
  if value < min || value > max then
    failwith (Printf.sprintf "%s must be in range [%f, %f]" name min max)

let integrate f a b eps =
  let rec simpson a b fa fb =
    let m = (a +. b) /. 2. in
    let fm = f m in
    let s = (fa +. 4. *. fm +. fb) *. (b -. a) /. 6. in
    if b -. a < eps then s
    else
      let s1 = simpson a m fa fm in
      let s2 = simpson m b fm fb in
      s1 +. s2
  in
  simpson a b (f a) (f b)

let find_root f a b eps =
  let rec bisect a b =
    let c = (a +. b) /. 2. in
    let fc = f c in
    if b -. a < eps || abs_float fc < eps then c
    else if fc *. f a > 0. then bisect c b
    else bisect a c
  in
  bisect a b

let parallel_map f lst =
  let open Lwt.Infix in
  Lwt_list.map_p (fun x -> Lwt.return (f x)) lst

let safe_div a b =
  let epsilon = 1e-10 in
  let safe_b = Tensor.where (Tensor.lt (Tensor.abs b) (Tensor.float_scalar epsilon))
                 (Tensor.full_like b epsilon)
                 b
  in
  Tensor.div a safe_b
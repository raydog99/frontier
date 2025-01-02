let multivariate_normal mean cov n =
  let d = Array.length mean in
  let l = Array.make_matrix d d 0.0 in
  
  (* Cholesky decomposition *)
  for i = 0 to d - 1 do
    for j = 0 to i do
      let s = ref cov.(i).(j) in
      for k = 0 to j - 1 do
        s := !s -. l.(i).(k) *. l.(j).(k)
      done;
      l.(i).(j) <- if i = j then sqrt(!s)
                   else if !s = 0.0 then 0.0
                   else !s /. l.(j).(j)
    done
  done;
  
  (* Generate samples *)
  Array.init n (fun _ ->
    let z = Array.init d (fun _ -> Random.float_gaussian ()) in
    let x = Array.make d 0.0 in
    for i = 0 to d - 1 do
      x.(i) <- mean.(i) +. 
        Array.fold_left (fun acc j -> 
          acc +. l.(i).(j) *. z.(j)
        ) 0.0 (Array.init (i + 1) (fun x -> x))
    done;
    x
  )

let gamma_rv shape scale =
  let d = shape -. 1.0 /. 3.0 in
  let c = 1.0 /. sqrt(9.0 *. d) in
  let rec generate () =
    let x = Random.float_gaussian () in
    let v = (1.0 +. c *. x) ** 3.0 in
    if v <= 0.0 then generate ()
    else
      let u = Random.float 1.0 in
      let x2 = x *. x in
      if u < 1.0 -. 0.0331 *. x2 *. x2 then v *. scale
      else if log u < 0.5 *. x2 +. d *. (1.0 -. v +. log v) then v *. scale
      else generate ()
  in
  generate ()
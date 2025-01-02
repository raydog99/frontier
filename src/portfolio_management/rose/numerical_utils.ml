let log_sum_exp arr =
  if Array.length arr = 0 then neg_infinity else
  let max_val = Array.fold_left max neg_infinity arr in
  let sum = Array.fold_left (fun acc x ->
    acc +. exp (x -. max_val)
  ) 0.0 arr in
  max_val +. log sum

let stable_variance arr =
  let n = Array.length arr in
  if n < 2 then 0.0 else
  let mean = ref 0.0 in
  let m2 = ref 0.0 in
  
  for i = 0 to n - 1 do
    let delta = arr.(i) -. !mean in
    mean := !mean +. delta /. float_of_int (i + 1);
    m2 := !m2 +. delta *. (arr.(i) -. !mean)
  done;
  
  !m2 /. float_of_int (n - 1)

let condition_number mat =
  let n = Array.length mat in
  let work = Array.make_matrix n n 0.0 in
  Array.iteri (fun i row ->
    Array.iteri (fun j x -> work.(i).(j) <- x) row
  ) mat;
  
  let max_eigen = ref 0.0 in
  let min_eigen = ref infinity in
  let v = Array.make n 1.0 in
  let temp = Array.make n 0.0 in
  
  for _ = 0 to 100 do
    for i = 0 to n - 1 do
      temp.(i) <- Array.fold_left2 (fun acc a b -> acc +. a *. b)
        0.0 work.(i) v
    done;
    
    let norm = sqrt (Array.fold_left (fun acc x -> acc +. x *. x) 0.0 temp) in
    Array.iteri (fun i x -> v.(i) <- x /. norm) temp;
    
    max_eigen := norm
  done;
  
  !max_eigen /. (if !min_eigen = infinity then 1e-10 else !min_eigen)

let matrix_inverse mat = [| [|0.0|] |]
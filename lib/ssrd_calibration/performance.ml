let memoize f =
  let table = Hashtbl.create 1024 in
  fun x ->
    try Hashtbl.find table x
    with Not_found ->
      let y = f x in
      Hashtbl.add table x y;
      y

let parallel_map f l =
  let num_threads = 4 in
  let rec split_list n lst =
    if n = 0 then ([], lst)
    else match lst with
      | [] -> ([], [])
      | x :: xs ->
          let (left, right) = split_list (n-1) xs in
          (x :: left, right)
  in
  let rec aux acc = function
    | [] -> List.concat (List.rev acc)
    | chunk :: chunks ->
        let thread = Thread.create (List.map f) chunk in
        aux (thread :: acc) chunks
  in
  let chunks = List.init num_threads (fun _ -> fst (split_list (List.length l / num_threads) l)) in
  aux [] chunks |> List.map Thread.join |> List.concat

let time_it f x =
  let start = Unix.gettimeofday () in
  let result = f x in
  let end_ = Unix.gettimeofday () in
  (result, end_ -. start)
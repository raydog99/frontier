open Torch

type cache_entry = {
  kernel_values: Tensor.t;
  last_access: float;
  size_bytes: int;
}

type t = {
  cache: (string, cache_entry) Hashtbl.t;
  max_size_bytes: int;
  current_size_bytes: int ref;
}

let create max_size_mb =
  {
    cache = Hashtbl.create 16;
    max_size_bytes = max_size_mb * 1024 * 1024;
    current_size_bytes = ref 0;
  }

let compute_key x y =
  let x_hash = Tensor.sum x |> Tensor.float_value in
  let y_hash = Tensor.sum y |> Tensor.float_value in
  Printf.sprintf "%f_%f" x_hash y_hash

let estimate_size tensor =
  let dims = Tensor.size_list tensor in
  List.fold_left ( * ) 8 dims  (* float64 *)

let evict cache required_bytes =
  let entries = Hashtbl.to_seq cache.cache
               |> List.of_seq
               |> List.sort (fun (_, a) (_, b) -> 
                   compare a.last_access b.last_access) in
  let rec evict_entries entries freed_bytes =
    match entries with
    | [] -> freed_bytes
    | (key, entry) :: rest ->
        Hashtbl.remove cache.cache key;
        let new_freed = freed_bytes + entry.size_bytes in
        if new_freed >= required_bytes then new_freed
        else evict_entries rest new_freed
  in
  evict_entries entries 0

let get_or_compute cache key compute_fn x y =
  match Hashtbl.find_opt cache.cache key with
  | Some entry ->
      let entry = {entry with last_access = Unix.time ()} in
      Hashtbl.replace cache.cache key entry;
      entry.kernel_values
  | None ->
      let result = compute_fn x y in
      let size = estimate_size result in
      if size > cache.max_size_bytes then
        result
      else begin
        if !cache.current_size_bytes + size > cache.max_size_bytes then
          cache.current_size_bytes := !cache.current_size_bytes - 
            evict cache size;
        let entry = {
          kernel_values = result;
          last_access = Unix.time ();
          size_bytes = size;
        } in
        Hashtbl.add cache.cache key entry;
        cache.current_size_bytes := !cache.current_size_bytes + size;
        result
      end
open Lwt

let parallel_map f lst =
  lst
  |> Lwt_list.map_p (fun x -> Lwt.return (f x))
  |> Lwt_main.run

let chunk_list lst chunk_size =
  let rec aux acc current rest =
    match rest with
    | [] -> List.rev (List.rev current :: acc)
    | _ :: _ as l ->
        if List.length current = chunk_size then
          aux (List.rev current :: acc) [] l
        else
          aux acc (List.hd l :: current) (List.tl l)
  in
  aux [] [] lst

let parallel_fold f init lst =
  let chunks = chunk_list lst 100 in
  let fold_chunk chunk =
    Lwt.return (List.fold_left f init chunk)
  in
  chunks
  |> Lwt_list.map_p fold_chunk
  |> Lwt_main.run
  |> List.fold_left f init
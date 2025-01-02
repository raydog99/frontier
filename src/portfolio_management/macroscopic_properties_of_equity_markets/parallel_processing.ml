open Lwt

let parallel_map f array =
  array
  |> Array.map (fun x -> Lwt.return (f x))
  |> Array.to_list
  |> Lwt.parallel_map_p Lwt.return
  |> Lwt_main.run
  |> Array.of_list

let parallel_fold f init array =
  array
  |> Array.to_list
  |> Lwt_list.fold_left_s (fun acc x -> Lwt.return (f acc x)) init
  |> Lwt_main.run
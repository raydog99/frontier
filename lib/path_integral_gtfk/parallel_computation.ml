open Lwt

let num_workers = 4

let parallel_map f xs =
  let pool = Lwt_pool.create num_workers (fun _ -> Lwt.return_unit) in
  Lwt_list.map_p (Lwt_pool.use pool (fun () -> Lwt.return (f ()))) xs
  |> Lwt_main.run
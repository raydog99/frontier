open Lwt.Infix

let parallel_map_chunked ~chunks ~f lst =
  let chunk_size = (List.length lst + chunks - 1) / chunks in
  let chunked_list = List.init chunks (fun i ->
    List.filteri (fun j _ -> j / chunk_size = i) lst
  ) in
  Lwt_list.map_p (fun chunk -> Lwt_list.map_p f chunk) chunked_list >|= List.flatten
open Lwt.Infix

type node = {
  id: int;
  address: string;
}

type cluster = {
  nodes: node list;
  master: node;
}

let create_cluster master_address slave_addresses =
  let master = { id = 0; address = master_address } in
  let slaves = List.mapi (fun i addr -> { id = i + 1; address = addr }) slave_addresses in
  { nodes = master :: slaves; master }

let distribute_work cluster work_items =
  let rec distribute acc = function
    | [] -> acc
    | item :: rest ->
        let node = List.nth cluster.nodes (List.length acc mod List.length cluster.nodes) in
        distribute ((node, item) :: acc) rest
  in
  List.rev (distribute [] work_items)

let collect_results distributed_work =
  Lwt_list.map_p
    (fun (node, work) ->
       Lwt.return (node, work ()))
    distributed_work
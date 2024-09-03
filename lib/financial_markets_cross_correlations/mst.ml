module PriorityQueue = struct
  type 'a t = {
    mutable heap: 'a array;
    mutable size: int;
    compare: 'a -> 'a -> int;
  }

  let create ?(initial_capacity = 16) compare =
    { heap = Array.make initial_capacity (Obj.magic 0); size = 0; compare }

  let parent i = (i - 1) / 2
  let left i = 2 * i + 1
  let right i = 2 * i + 2

  let swap heap i j =
    let temp = heap.(i) in
    heap.(i) <- heap.(j);
    heap.(j) <- temp

  let rec bubble_up heap compare i =
    let p = parent i in
    if i > 0 && compare heap.(i) heap.(p) < 0 then begin
      swap heap i p;
      bubble_up heap compare p
    end

  let rec bubble_down heap compare size i =
    let l = left i and r = right i in
    let smallest =
      if l < size && compare heap.(l) heap.(i) < 0 then l else i in
    let smallest =
      if r < size && compare heap.(r) heap.(smallest) < 0 then r else smallest in
    if smallest <> i then begin
      swap heap i smallest;
      bubble_down heap compare size smallest
    end

  let push q x =
    if q.size = Array.length q.heap then
      q.heap <- Array.append q.heap (Array.make (Array.length q.heap) (Obj.magic 0));
    q.heap.(q.size) <- x;
    bubble_up q.heap q.compare q.size;
    q.size <- q.size + 1

  let pop q =
    if q.size = 0 then raise (Invalid_argument "PriorityQueue is empty");
    let x = q.heap.(0) in
    q.size <- q.size - 1;
    q.heap.(0) <- q.heap.(q.size);
    bubble_down q.heap q.compare q.size 0;
    x

  let is_empty q = q.size = 0
end

let prim_mst matrix =
  let n = Array.length matrix in
  let mst = Array.make_matrix n n 0.0 in
  let visited = Array.make n false in
  let pq = PriorityQueue.create (fun (_, _, w1) (_, _, w2) -> compare w1 w2) in

  visited.(0) <- true;
  for j = 1 to n - 1 do
    PriorityQueue.push pq (0, j, matrix.(0).(j))
  done;

  while not (PriorityQueue.is_empty pq) do
    let (u, v, w) = PriorityQueue.pop pq in
    if not visited.(v) then begin
      visited.(v) <- true;
      mst.(u).(v) <- w;
      mst.(v).(u) <- w;
      for j = 0 to n - 1 do
        if not visited.(j) then
          PriorityQueue.push pq (v, j, matrix.(v).(j))
      done
    end
  done;
  mst

let normalized_tree_length mst =
  let n = Array.length mst in
  let total_weight = Array.fold_left (fun acc row ->
    acc +. Array.fold_left (+.) 0. row
  ) 0. mst in
  total_weight /. (2. *. float_of_int (n - 1))
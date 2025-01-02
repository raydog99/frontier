open Torch

type t = {
  data: (Tensor.t * Tensor.t) array;
  batch_size: int;
  shuffle: bool;
  device: Device.t;
}

let create ~data ~batch_size ~shuffle ~device =
  { data; batch_size; shuffle; device }

let shuffle_data data =
  let n = Array.length data in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    let temp = data.(i) in
    data.(i) <- data.(j);
    data.(j) <- temp;
  done

let batches t =
  let data = Array.copy t.data in
  if t.shuffle then shuffle_data data;
  Array.to_seq data
  |> Seq.map (fun (x, y) ->
      (Tensor.to_device x ~device:t.device,
       Tensor.to_device y ~device:t.device))
  |> Seq.chunks t.batch_size
  |> Seq.map Array.of_seq
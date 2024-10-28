open Type

let serialize_sample s chain_id iter =
  {
    theta_star = to_float1 s.theta_star |> Array.of_float1;
    sigma = s.sigma;
    iteration = iter;
    chain_id;
    timestamp = Unix.gettimeofday ();
  }

let save_samples filename samples chain_id =
  let oc = open_out filename in
  List.iteri (fun i s ->
    let stored = serialize_sample s chain_id i in
    Marshal.to_channel oc stored []
  ) samples;
  close_out oc

let load_samples filename =
  let ic = open_in filename in
  let samples = ref [] in
  try
    while true do
      let stored: sample_storage = Marshal.from_channel ic in
      samples := stored :: !samples
    done;
    !samples
  with End_of_file ->
    close_in ic;
    !samples
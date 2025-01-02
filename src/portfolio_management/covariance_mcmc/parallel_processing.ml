open Torch
open Types

type task =
  | SampleGeneration of (unit -> Tensor.t)
  | CovarianceComputation of (Tensor.t -> Tensor.t)
  | ChainUpdate of (Tensor.t -> Tensor.t)

type worker_pool = {
  tasks: task Queue.t;
  results: Tensor.t Queue.t;
  n_workers: int;
}

let create_worker_pool n_workers =
  {
    tasks = Queue.create ();
    results = Queue.create ();
    n_workers;
  }

let worker_function pool =
  let rec process_tasks () =
    match Queue.take_opt pool.tasks with
    | None -> ()
    | Some task ->
        let result = match task with
          | SampleGeneration f -> f ()
          | CovarianceComputation f -> f (Tensor.zeros [1])
          | ChainUpdate f -> f (Tensor.zeros [1])
        in
        Queue.add result pool.results;
        process_tasks ()
  in
  process_tasks ()

let parallel_chain_execution chain initial n_chains n_samples device_config =
  let pool = create_worker_pool (min n_chains 8) in
  
  let workers = List.init pool.n_workers (fun _ ->
    Domain.spawn (fun () -> worker_function pool)
  ) in
  
  for i = 0 to n_chains - 1 do
    Queue.add (SampleGeneration (fun () ->
      let x0 = Tensor.add initial 
        (Tensor.mul_scalar (Tensor.randn_like initial) 0.1) in
      let samples = ref x0 in
      for _ = 1 to n_samples do
        samples := chain.kernel !samples
      done;
      !samples
    )) pool.tasks
  done;
  
  let rec collect acc = function
    | 0 -> acc
    | n -> 
        let result = Queue.take pool.results in
        collect (result :: acc) (n - 1)
  in
  
  (* Wait for workers *)
  List.iter Domain.join workers;
  
  collect [] n_chains

let parallel_covariance_estimation chain initial n_chains n_samples epsilon delta =
  let device_config = Gpu_compute.{
    use_gpu = true;
    device_id = 0;
    precision = `Float;
  } in
  
  let chain_samples = parallel_chain_execution 
    chain initial n_chains n_samples device_config in
  
  let all_samples = Tensor.cat (Array.of_list chain_samples) ~dim:0 in
  
  {
    mean = Tensor.mean all_samples ~dim:[0] ~keepdim:false;
    covariance = Tensor.mm (Tensor.transpose all_samples 0 1) all_samples
                |> Tensor.div_scalar (float_of_int (n_chains * n_samples));
  }
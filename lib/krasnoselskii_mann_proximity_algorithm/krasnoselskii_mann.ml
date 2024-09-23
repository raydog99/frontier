open Torch
open Lwt.Infix
module L = Logs

type t = {
  alpha: float;
  max_iterations: int;
  tolerance: float;
  rel_tolerance: float;
  stagnation_window: int;
  parallel: bool;
  logging_level: Logs.level;
  adaptive_step: bool;
  preconditioner: Tensor.t option;
  early_stopping: bool;
  devices: int list;
  checkpoint_interval: int option;
  checkpoint_path: string option;
  plugins: (module Plugin.S) list;
}

let create ?(alpha=0.5) ?(max_iterations=1000) ?(tolerance=1e-6)
            ?(rel_tolerance=1e-8) ?(stagnation_window=10)
            ?(parallel=false) ?(logging_level=Logs.Info) ?(adaptive_step=false)
            ?(preconditioner=None) ?(early_stopping=false)
            ?(devices=[0]) ?(checkpoint_interval=None)
            ?(checkpoint_path=None) ?(plugins=[]) () =
  if alpha <= 0. || alpha >= 1. then
    invalid_arg "alpha must be in the range (0, 1)"
  else if max_iterations <= 0 then
    invalid_arg "max_iterations must be positive"
  else if tolerance <= 0. || rel_tolerance <= 0. then
    invalid_arg "tolerances must be positive"
  else if stagnation_window <= 0 then
    invalid_arg "stagnation_window must be positive"
  else if List.length devices = 0 then
    invalid_arg "at least one device must be specified"
  else
    { alpha; max_iterations; tolerance; rel_tolerance; stagnation_window; 
      parallel; logging_level; adaptive_step; preconditioner; early_stopping; devices;
      checkpoint_interval; checkpoint_path; plugins }

type proximity_fn = Tensor.t -> Tensor.t
type constraint_fn = Tensor.t -> Tensor.t
type validation_fn = Tensor.t -> float

exception ConvergenceError of string
exception CheckpointError of string
exception PluginError of string

type iteration_result = {
  solution: Tensor.t;
  iterations: int;
  converged: bool;
  final_residual: float;
  convergence_history: (int * float) list;
  stopping_criterion: string;
  validation_history: (int * float) list option;
  performance_metrics: (string * float) list;
  plugin_results: (string * Yojson.Safe.t) list;
}

let default_proximity_operator f x =
  let grad = Tensor.grad f x in
  Tensor.((x - grad) / (Scalar.one + grad))

let fixed_point_residual x T_x =
  Tensor.(x - T_x)

let log_progress src level i residual_norm validation_error =
  let message = match validation_error with
    | Some err -> Printf.sprintf "Iteration %d: residual = %e, validation error = %e" i residual_norm err
    | None -> Printf.sprintf "Iteration %d: residual = %e" i residual_norm
  in
  L.msg level (fun m -> m ~src "%s" message)

let check_convergence t i x x_prev residual_norm history validation_history =
  let rel_change = Tensor.(norm (x - x_prev) / (norm x +. 1e-10)) |> Tensor.to_float0_exn in
  let stagnated = 
    if List.length history >= t.stagnation_window then
      let window = List.take t.stagnation_window history in
      let improvement = (List.hd window |> snd) -. residual_norm in
      improvement < t.tolerance
    else
      false
  in
  let early_stop =
    match validation_history with
    | Some hist when List.length hist >= t.stagnation_window ->
        let window = List.take t.stagnation_window hist in
        let best_val_error = List.fold_left (fun acc (_, err) -> min acc err) Float.max_float window in
        (List.hd window |> snd) > best_val_error
    | _ -> false
  in
  if residual_norm < t.tolerance then
    Some "Absolute tolerance reached"
  else if rel_change < t.rel_tolerance then
    Some "Relative tolerance reached"
  else if stagnated then
    Some "Stagnation detected"
  else if early_stop then
    Some "Early stopping criterion met"
  else if i >= t.max_iterations then
    Some "Maximum iterations reached"
  else
    None

let adapt_step_size t x x_prev T_x T_x_prev =
  if not t.adaptive_step then t.alpha
  else
    let num = Tensor.(dot (x - x_prev) (T_x - T_x_prev)) in
    let den = Tensor.(norm (T_x - T_x_prev) ** 2) in
    let new_alpha = Tensor.to_float0_exn Tensor.(num / den) in
    Float.max 0.1 (Float.min new_alpha 0.9)

let apply_preconditioner t x =
  match t.preconditioner with
  | Some P -> Tensor.(matmul P x)
  | None -> x

let distribute_computation t f x =
  let num_devices = List.length t.devices in
  if num_devices = 1 then
    f x
  else
    let chunks = Tensor.chunk x num_devices 0 in
    let results = Lwt_list.map_p (fun (i, chunk) ->
      Lwt_preemptive.detach (fun () ->
        let device = List.nth t.devices i in
        let chunk_on_device = Tensor.to_device chunk device in
        f chunk_on_device
      ) ()
    ) (List.mapi (fun i c -> (i, c)) chunks)
    in
    let%lwt computed_chunks = results in
    Lwt.return (Tensor.cat computed_chunks 0)

let save_checkpoint i x history validation_history path =
  try
    let filename = Printf.sprintf "%s/checkpoint_%d.pt" path i in
    Tensor.save [("x", x); ("history", Tensor.of_float1 (Array.of_list history))] filename;
    (match validation_history with
    | Some vh -> Tensor.save [("validation_history", Tensor.of_float1 (Array.of_list vh))] (filename ^ ".val")
    | None -> ());
    L.info (fun m -> m "Checkpoint saved: %s" filename)
  with e ->
    raise (CheckpointError (Printf.sprintf "Failed to save checkpoint: %s" (Printexc.to_string e)))

let load_checkpoint path =
  try
    let tensors = Tensor.load path in
    let x = List.assoc "x" tensors in
    let history = Tensor.to_float1 (List.assoc "history" tensors) |> Array.to_list in
    let validation_history =
      try
        Some (Tensor.to_float1 (List.assoc "validation_history" (Tensor.load (path ^ ".val"))) |> Array.to_list)
      with _ -> None
    in
    Some (x, history, validation_history)
  with e ->
    L.warn (fun m -> m "Failed to load checkpoint: %s" (Printexc.to_string e));
    None

let iterate_sequential t src x T constraint_fn validation_fn =
  let rec loop i x history validation_history start_time =
    let iteration_start = Unix.gettimeofday () in
    let T_x = T x in
    let constrained_T_x = constraint_fn T_x in
    let preconditioned_T_x = apply_preconditioner t constrained_T_x in
    let res = fixed_point_residual x preconditioned_T_x in
    let residual_norm = Tensor.norm res |> Tensor.to_float0_exn in
    let validation_error = if t.early_stopping then Some (validation_fn x) else None in
    let new_history = (i, residual_norm) :: history in
    let new_validation_history = 
      match validation_error with
      | Some err -> Some ((i, err) :: (Option.value validation_history ~default:[]))
      | None -> validation_history
    in
    log_progress src t.logging_level i residual_norm validation_error;
    
    (match t.checkpoint_interval, t.checkpoint_path with
    | Some interval, Some path when i mod interval = 0 -> save_checkpoint i x new_history new_validation_history path
    | _ -> ());

    let iteration_time = Unix.gettimeofday () -. iteration_start in
    
    let plugin_results = List.map (fun (module P: Plugin.S) ->
      try
        (P.name, P.on_iteration i x residual_norm validation_error)
      with e ->
        raise (PluginError (Printf.sprintf "Plugin %s failed: %s" P.name (Printexc.to_string e)))
    ) t.plugins in
    
    match check_convergence t i x preconditioned_T_x residual_norm new_history new_validation_history with
    | Some reason ->
        let total_time = Unix.gettimeofday () -. start_time in
        let avg_iteration_time = total_time /. float_of_int i in
        { solution = x; iterations = i; converged = (reason <> "Maximum iterations reached"); 
          final_residual = residual_norm; convergence_history = List.rev new_history; 
          stopping_criterion = reason; validation_history = Option.map List.rev new_validation_history;
          performance_metrics = [
            ("total_time", total_time);
            ("avg_iteration_time", avg_iteration_time);
            ("last_iteration_time", iteration_time)
          ];
          plugin_results = plugin_results }
    | None ->
        let step_size = if i > 0 then adapt_step_size t x (List.hd history |> fst) preconditioned_T_x (List.hd history |> snd) else t.alpha in
        let x_next = Tensor.((1. - step_size) * x + step_size * preconditioned_T_x) in
        loop (i + 1) x_next new_history new_validation_history start_time
  in
  loop 0 x [] None (Unix.gettimeofday ())

let iterate_parallel t src x T constraint_fn validation_fn =
  let rec loop i x history validation_history start_time =
    let%lwt iteration_start = Lwt.return (Unix.gettimeofday ()) in
    let%lwt T_x = distribute_computation t T x in
    let constrained_T_x = constraint_fn T_x in
    let preconditioned_T_x = apply_preconditioner t constrained_T_x in
    let res = fixed_point_residual x preconditioned_T_x in
    let residual_norm = Tensor.norm res |> Tensor.to_float0_exn in
    let%lwt validation_error = 
      if t.early_stopping then
        Lwt_preemptive.detach (fun () -> Some (validation_fn x)) ()
      else
        Lwt.return None
    in
    let new_history = (i, residual_norm) :: history in
    let new_validation_history = 
      match validation_error with
      | Some err -> Some ((i, err) :: (Option.value validation_history ~default:[]))
      | None -> validation_history
    in
    log_progress src t.logging_level i residual_norm validation_error;
    
    (match t.checkpoint_interval, t.checkpoint_path with
    | Some interval, Some path when i mod interval = 0 -> save_checkpoint i x new_history new_validation_history path
    | _ -> ());

    let%lwt iteration_time = Lwt.return (Unix.gettimeofday () -. iteration_start) in
    
    let%lwt plugin_results = Lwt_list.map_p (fun (module P: Plugin.S) ->
      Lwt.catch
        (fun () -> Lwt_preemptive.detach (fun () -> (P.name, P.on_iteration i x residual_norm validation_error)) ())
        (fun e -> Lwt.fail (PluginError (Printf.sprintf "Plugin %s failed: %s" P.name (Printexc.to_string e))))
    ) t.plugins in
    
    match check_convergence t i x preconditioned_T_x residual_norm new_history new_validation_history with
    | Some reason ->
        let total_time = Unix.gettimeofday () -. start_time in
        let avg_iteration_time = total_time /. float_of_int i in
        Lwt.return { solution = x; iterations = i; converged = (reason <> "Maximum iterations reached"); 
                     final_residual = residual_norm; convergence_history = List.rev new_history; 
                     stopping_criterion = reason; validation_history = Option.map List.rev new_validation_history;
                     performance_metrics = [
                       ("total_time", total_time);
                       ("avg_iteration_time", avg_iteration_time);
                       ("last_iteration_time", iteration_time)
                     ];
                     plugin_results = plugin_results }
    | None ->
        let step_size = if i > 0 then adapt_step_size t x (List.hd history |> fst) preconditioned_T_x (List.hd history |> snd) else t.alpha in
        let x_next = Tensor.((1. - step_size) * x + step_size * preconditioned_T_x) in
        Lwt.pause () >>= fun () -> loop (i + 1) x_next new_history new_validation_history start_time
  in
  loop 0 x [] None (Unix.gettimeofday ())

let solve t ?(prox_op=default_proximity_operator) ?(constraint_fn=fun x -> x) ?(validation_fn=fun _ -> 0.) f x_init =
  let src = Logs.Src.create "KrasnoselskiiMann" in
  Logs.Src.set_level src (Some t.logging_level);
  let T x = prox_op f x in
  let result = 
    match t.checkpoint_path with
    | Some path ->
        (match load_checkpoint path with
        | Some (x, history, validation_history) ->
            L.info (fun m -> m ~src "Resuming from checkpoint");
            if t.parallel then
              Lwt_main.run (iterate_parallel t src x T constraint_fn validation_fn)
            else
              iterate_sequential t src x T constraint_fn validation_fn
        | None ->
            if t.parallel then
              Lwt_main.run (iterate_parallel t src x_init T constraint_fn validation_fn)
            else
              iterate_sequential t src x_init T constraint_fn validation_fn)
    | None ->
        if t.parallel then
          Lwt_main.run (iterate_parallel t src x_init T constraint_fn validation_fn)
        else
          iterate_sequential t src x_init T constraint_fn validation_fn
  in
  if not result.converged then
    raise (ConvergenceError result.stopping_criterion)
  else
    result

let project_onto_ball center radius x =
  let diff = Tensor.(x - center) in
  let norm = Tensor.norm diff in
  if Tensor.to_float0_exn norm <= radius then x
  else Tensor.(center + (radius / norm) * diff)

let project_onto_simplex x =
  let sorted_x = Tensor.sort x ~descending:true in
  let cumsum = Tensor.cumsum sorted_x ~dim:0 in
  let arange = Tensor.arange ~start:1 ~end_:(Tensor.shape x).[0] ~step:1 ~options:(Tensor.kind x) in
  let temp = Tensor.((sorted_x - ((cumsum - Scalar.one) / arange)) > Scalar.zero) in
  let k = Tensor.sum temp |> Tensor.to_int0_exn in
  let tau = Tensor.((cumsum.[k-1] - Scalar.one) / (Scalar.float k)) in
  Tensor.(max (x - tau) (Scalar.zero))

let solve_lasso ?(lambda=1.0) A b x_init =
  let f x =
    let residual = Tensor.(matmul A x - b) in
    Tensor.((dot residual residual + lambda * (norm x ~p:1))) / (Scalar.float 2)
  in
  let prox_op _ x =
    let sign_x = Tensor.sign x in
    let abs_x = Tensor.abs x in
    Tensor.(sign_x * (max (abs_x - Scalar.float lambda) (Scalar.zero)))
  in
  solve { alpha=0.5; max_iterations=1000; tolerance=1e-6; rel_tolerance=1e-8; 
          stagnation_window=10; parallel=false; logging_level=Logs.Info; adaptive_step=true;
          preconditioner=None; early_stopping=false; devices=[0]; 
          checkpoint_interval=None; checkpoint_path=None; plugins=[] } ~prox_op f x_init

let solve_ridge ?(lambda=1.0) A b x_init =
  let f x =
    let residual = Tensor.(matmul A x - b) in
    Tensor.((dot residual residual + lambda * (dot x x))) / (Scalar.float 2)
  in
  let prox_op _ x =
    Tensor.(x / (Scalar.one + Scalar.float (2.0 *. lambda)))
  in
  solve { alpha=0.5; max_iterations=1000; tolerance=1e-6; rel_tolerance=1e-8; 
          stagnation_window=10; parallel=false; logging_level=Logs.Info; adaptive_step=true;
          preconditioner=None; early_stopping=false; devices=[0]; 
          checkpoint_interval=None; checkpoint_path=None; plugins=[] } ~prox_op f x_init

let solve_elastic_net ?(alpha=0.5) ?(lambda=1.0) A b x_init =
  let f x =
    let residual = Tensor.(matmul A x - b) in
    let l1_term = Tensor.(lambda * alpha * (norm x ~p:1)) in
    let l2_term = Tensor.(lambda * (1. - alpha) * (dot x x) / Scalar.float 2) in
    Tensor.((dot residual residual / Scalar.float 2) + l1_term + l2_term)
  in
  let prox_op _ x =
    let sign_x = Tensor.sign x in
    let abs_x = Tensor.abs x in
    let soft_threshold = Tensor.(max (abs_x - Scalar.float (lambda * alpha)) (Scalar.zero)) in
    Tensor.(sign_x * soft_threshold / (Scalar.one + Scalar.float (lambda * (1. - alpha))))
  in
  solve { alpha=0.5; max_iterations=1000; tolerance=1e-6; rel_tolerance=1e-8; 
          stagnation_window=10; parallel=false; logging_level=Logs.Info; adaptive_step=true;
          preconditioner=None; early_stopping=false; devices=[0]; 
          checkpoint_interval=None; checkpoint_path=None; plugins=[] } ~prox_op f x_init
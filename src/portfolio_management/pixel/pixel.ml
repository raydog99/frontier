open Torch

type dimension = 
  | D1 of int
  | D2 of int * int
  | D3 of int * int * int

type grid_config = {
  channel_size: int;
  dimension: dimension;
  num_grids: int;
  grid_shifts: float array;
}

type interpolation_kernel = 
  | Cosine 
  | Linear
  | Cubic

module Device = struct
  type t = CPU | GPU of int

  let default = CPU

  let get_device = function 
    | CPU -> Device.Cpu
    | GPU id -> Device.Cuda id

  let memory_info = function
    | CPU -> None
    | GPU id -> 
        let total, free = Cuda.memory_info id in
        Some (total, free)
end

let gradient input ~wrt = 
  Tensor.grad_of_fn1 (fun x -> Tensor.sum x) input ~wrt 
  |> Option.get

let laplacian input ~wrt =
  let grad1 = gradient input ~wrt in
  gradient grad1 ~wrt

let partial_t input ~t = gradient input ~wrt:t
let partial_x input ~x = gradient input ~wrt:x

let partial_xx input ~x =
  let dx = partial_x input ~x in
  partial_x dx ~x

let compute_high_order_derivative tensor ~order ~wrt =
  let rec derive t n =
    if n <= 0 then t
    else derive (gradient t ~wrt) (n-1)
  in
  derive tensor order

let validate_dimensions dims =
  Array.for_all (fun d -> d > 0) dims

let ensure_compatible_shapes shape1 shape2 =
  Array.length shape1 = Array.length shape2 &&
  Array.for_all2 (=) shape1 shape2

let compute_padding kernel_size =
  kernel_size / 2

let compute_output_shape input_shape kernel_size stride padding =
  (input_shape + 2 * padding - kernel_size) / stride + 1

module Grid_sampler = struct
  type t = {
    support_size: int;
    kernel: Types.interpolation_kernel;
  }

  let create ~support_size ~kernel = {
    support_size;
    kernel;
  }

  let cosine_kernel x =
    if x <= 0.0 then 0.0
    else if x >= 1.0 then 1.0
    else 0.5 *. (1.0 -. cos (Float.pi *. x))

  let sample t ~grid ~coords =
    match t.kernel with
    | Types.Cosine ->
        let normalized_coords = div coords (float_of_int t.support_size) in
        let weights = map cosine_kernel normalized_coords in
        let indices = floor normalized_coords in
        let samples = index_select grid ~dim:0 ~index:indices in
        mul samples weights
    | Types.Linear ->
        let indices = floor coords in
        let weights = sub coords indices in
        let samples = index_select grid ~dim:0 ~index:indices in
        mul samples weights
    | Types.Cubic ->
        (* Cubic interpolation *)
        let indices = floor coords in
        let weights = sub coords indices in
        let w2 = mul weights weights in
        let w3 = mul weights w2 in
        
        let p0 = mul_float w3 (-0.5) |> add (mul_float w2 1.0) |>
                 add (mul_float weights (-0.5)) in
        let p1 = mul_float w3 1.5 |> add (mul_float w2 (-2.5)) |> add_float 1.0 in
        let p2 = mul_float w3 (-1.5) |> add (mul_float w2 2.0) |>
                 add (mul_float weights 0.5) in
        let p3 = mul_float w3 0.5 |> add (mul_float w2 (-0.5)) in
        
        let samples = Array.init 4 (fun i ->
          index_select grid ~dim:0 ~index:(add indices (of_int0 (i - 1)))
        ) in
        
        Array.fold_left2 (fun acc s p ->
          add acc (mul s p)
        ) (zeros_like samples.(0)) samples [|p0; p1; p2; p3|]

  let sample_with_gradients t ~grid ~coords =
    let value = sample t ~grid ~coords in
    let dx = Tensor_ops.gradient value ~wrt:coords in
    let dy = Tensor_ops.gradient dx ~wrt:coords in
    value, dx, dy
end

module Feature_extractor = struct
  type t = {
    grid_sampler: Grid_sampler.t;
    feature_dim: int;
  }

  let create ~grid_sampler ~feature_dim = {
    grid_sampler;
    feature_dim;
  }

  let extract t ~grid ~coords =
    let samples = Grid_sampler.sample t.grid_sampler ~grid ~coords in
    (* Transform samples into features *)
    let features = Tensor.linear samples t.feature_dim in
    Tensor.relu features
end

module Domain = struct
  type t = {
    physical_bounds: float array * float array;
    computational_bounds: float array * float array;
    mapping: Tensor.t -> Tensor.t;
    inverse_mapping: Tensor.t -> Tensor.t;
  }

  let create_arbitrary_domain ~physical_bounds ~computational_bounds =
    let physical_min, physical_max = physical_bounds in
    let comp_min, comp_max = computational_bounds in
    
    let scaling = Array.map2 (fun pmax pmin ->
      Array.map2 (fun cmax cmin ->
        (cmax -. cmin) /. (pmax -. pmin)
      ) comp_max comp_min
    ) physical_max physical_min in
    
    {
      physical_bounds;
      computational_bounds;
      mapping = (fun x ->
        let scaled = Tensor.sub x (Tensor.of_float_array physical_min) in
        Tensor.mul scaled (Tensor.of_float_array (Array.concat scaling))
      );
      inverse_mapping = (fun y ->
        let scaled = Tensor.div y (Tensor.of_float_array (Array.concat scaling)) in
        Tensor.add scaled (Tensor.of_float_array physical_min)
      )
    }

  let normalize_coords t ~coords =
    let mapped = t.mapping coords in
    let min_bound, max_bound = t.computational_bounds in
    Tensor.clamp mapped 
      ~min:(Tensor.of_float_array min_bound)
      ~max:(Tensor.of_float_array max_bound)
end

module Grid_representation = struct
  type t = {
    grid: Tensor.t;
    channel_size: int;
    height: int;
    width: int;
  }

  let create ~channel_size ~dims =
    let height, width = 
      match dims with 
      | [|h; w|] -> h, w
      | _ -> failwith "Invalid grid dimensions" in
    {
      grid = Tensor.randn [|channel_size; height; width|];
      channel_size;
      height;
      width;
    }

  let get t i j =
    Tensor.slice t.grid ~dim:1 ~start:i ~length:1 |>
    Tensor.slice ~dim:2 ~start:j ~length:1

  let set t i j value =
    Tensor.copy_ ~src:value 
      ~dst:(get t i j)

  let interpolate t ~x ~y =
    let x = Tensor.clamp x ~min:0.0 ~max:(float_of_int (t.height - 1)) in
    let y = Tensor.clamp y ~min:0.0 ~max:(float_of_int (t.width - 1)) in
    
    let x0 = Tensor.floor x |> Tensor.to_int0_exn in
    let x1 = min (x0 + 1) (t.height - 1) in
    let y0 = Tensor.floor y |> Tensor.to_int0_exn in
    let y1 = min (y0 + 1) (t.width - 1) in

    let wx = Tensor.sub x (Tensor.of_int0 x0) in
    let wy = Tensor.sub y (Tensor.of_int0 y0) in

    let v00 = get t x0 y0 in
    let v01 = get t x0 y1 in
    let v10 = get t x1 y0 in
    let v11 = get t x1 y1 in

    let c0 = Tensor.add 
      (Tensor.mul v00 (Tensor.sub (Tensor.float 1.0) wx))
      (Tensor.mul v10 wx) in
    let c1 = Tensor.add
      (Tensor.mul v01 (Tensor.sub (Tensor.float 1.0) wx))
      (Tensor.mul v11 wx) in
    
    Tensor.add
      (Tensor.mul c0 (Tensor.sub (Tensor.float 1.0) wy))
      (Tensor.mul c1 wy)

  let width t = t.width
  let height t = t.height
end

module Adaptive_refinement = struct
  type refinement_criteria = {
    gradient_threshold: float;
    curvature_threshold: float;
    max_level: int;
    min_cell_size: float;
  }

  type refined_cell = {
    level: int;
    position: float array;
    size: float array;
    values: Tensor.t;
    children: refined_cell array option;
  }

  let compute_refinement_indicator cell =
    let gradient = Tensor_ops.gradient cell.values 
      ~wrt:(Tensor.of_float_array cell.position) in
    let curvature = Tensor_ops.laplacian cell.values 
      ~wrt:(Tensor.of_float_array cell.position) in
    (gradient, curvature)

  let should_refine criteria cell =
    let gradient, curvature = compute_refinement_indicator cell in
    let grad_mag = Tensor.norm gradient in
    let curv_mag = Tensor.norm curvature in
    
    cell.level < criteria.max_level &&
    Array.for_all (fun s -> s > criteria.min_cell_size) cell.size &&
    (Tensor.float_value grad_mag > criteria.gradient_threshold ||
     Tensor.float_value curv_mag > criteria.curvature_threshold)

  let refine_cell cell =
    let child_size = Array.map (fun s -> s /. 2.) cell.size in
    let offsets = match Array.length cell.position with
      | 2 -> [|[|0.;0.|]; [|1.;0.|]; [|0.;1.|]; [|1.;1.|]|]
      | 3 -> [|[|0.;0.;0.|]; [|1.;0.;0.|]; [|0.;1.;0.|]; [|1.;1.;0.|];
              [|0.;0.;1.|]; [|1.;0.;1.|]; [|0.;1.;1.|]; [|1.;1.;1.|]|]
      | _ -> failwith "Unsupported dimension" in
    
    let children = Array.map (fun offset ->
      let pos = Array.map2 (fun p o -> p +. o *. child_size.(0)) 
        cell.position offset in
      { level = cell.level + 1;
        position = pos;
        size = child_size;
        values = Tensor.zeros_like cell.values;
        children = None }
    ) offsets in
    
    { cell with children = Some children }
end

module PDE = struct
  type params
  val residual : params -> (x:Tensor.t -> t:Tensor.t -> Tensor.t) -> 
                x:Tensor.t -> t:Tensor.t -> Tensor.t

  module Helmholtz = struct
    type params = {
      k: float;
      a1: float;
      a2: float;
    }

    let residual params pred ~x ~t =
      let u = pred ~x ~t in
      let laplacian = partial_xx u ~x in
      let k2_term = Tensor.mul_float u (params.k *. params.k) in
      let res = Tensor.add laplacian k2_term in
      
      let source = Tensor.sin (Tensor.mul_float x params.a1)
                  |> Tensor.mul (Tensor.sin (Tensor.mul_float t params.a2)) in
      Tensor.mse_loss res source Tensor.Sum
  end

  module Allen_Cahn = struct
    type params = {
      epsilon: float;
      lambda: float;
    }

    let residual params pred ~x ~t =
      let u = pred ~x ~t in
      let ut = partial_t u ~t in
      let uxx = partial_xx u ~x in
      
      let cubic = Tensor.pow u (Tensor.int 3) in
      let linear = Tensor.mul_float u params.lambda in
      let diff = Tensor.mul_float uxx (params.epsilon *. params.epsilon) in
      
      let rhs = Tensor.sub diff (Tensor.sub cubic linear) in
      let res = Tensor.sub ut rhs in
      Tensor.mse_loss res (Tensor.zeros_like res) Tensor.Sum
  end

  module KdV = struct
    type params = {
      alpha: float;
      beta: float;
    }

    let residual params pred ~x ~t =
      let u = pred ~x ~t in
      let ut = partial_t u ~t in
      let ux = partial_x u ~x in
      let uxxx = compute_high_order_derivative u ~order:3 ~wrt:x in
      
      let nonlinear = Tensor.mul_float 
        (Tensor.mul u ux) 
        params.alpha in
      let dispersive = Tensor.mul_float uxxx params.beta in
      
      let res = Tensor.add ut (Tensor.add nonlinear dispersive) in
      Tensor.mse_loss res (Tensor.zeros_like res) Tensor.Sum
  end

  module Navier_Stokes = struct
    type params = {
      nu: float;
      rho: float;
    }

    let residual params pred ~x ~y ~t =
      let u, v, p = pred ~x ~y ~t in
      
      (* Continuity equation *)
      let ux = partial_x u ~x in
      let vy = partial_x v ~y in
      let div_free = Tensor.add ux vy in
      
      (* Momentum equations *)
      (* x-momentum *)
      let ut = partial_t u ~t in
      let uxx = partial_xx u ~x in
      let uyy = partial_xx u ~y in
      let u_adv = Tensor.mul u ux in
      let v_adv = Tensor.mul v vy in
      let px = partial_x p ~x in
      
      let mom_x = Tensor.add ut 
        (Tensor.add 
           (Tensor.add u_adv v_adv)
           (Tensor.sub px 
              (Tensor.mul_float 
                 (Tensor.add uxx uyy) 
                 params.nu))) in
      
      (* y-momentum *)
      let vt = partial_t v ~t in
      let vxx = partial_xx v ~x in
      let vyy = partial_xx v ~y in
      let py = partial_x p ~y in
      
      let mom_y = Tensor.add vt
        (Tensor.add
           (Tensor.add u_adv v_adv)
           (Tensor.sub py
              (Tensor.mul_float
                 (Tensor.add vxx vyy)
                 params.nu))) in
      
      (* Combine losses *)
      let div_loss = Tensor.mse_loss div_free 
        (Tensor.zeros_like div_free) Tensor.Sum in
      let mom_x_loss = Tensor.mse_loss mom_x 
        (Tensor.zeros_like mom_x) Tensor.Sum in
      let mom_y_loss = Tensor.mse_loss mom_y 
        (Tensor.zeros_like mom_y) Tensor.Sum in
      
      Tensor.add div_loss (Tensor.add mom_x_loss mom_y_loss)
  end
end

module Pressure_velocity = struct
  type projection_method = 
    | Chorin
    | VanKan
    | Incremental

  type staggered_grid = {
    u: Tensor.t;
    v: Tensor.t;
    w: Tensor.t option;
    p: Tensor.t;
    dx: float;
    dy: float;
    dz: float option;
  }

  let create_staggered_grid ~nx ~ny ?nz ~dx ~dy ?dz () =
    {
      u = Tensor.zeros [|nx+1; ny|];
      v = Tensor.zeros [|nx; ny+1|];
      w = Option.map (fun nz -> Tensor.zeros [|nx; ny; nz+1|]) nz;
      p = Tensor.zeros [|nx; ny|];
      dx;
      dy;
      dz;
    }

  let solve_pressure_poisson grid ~rhs =
    (* Multigrid solver *)
    let rec v_cycle grid level =
      if level = 0 then
        Tensor.mul grid (Tensor.float 0.8)
      else
        let smoothed = gauss_seidel grid 4 in
        let coarse_grid = restrict smoothed in
        let coarse_rhs = restrict (residual grid rhs) in
        let coarse_correction = v_cycle coarse_grid (level - 1) in
        let correction = prolong coarse_correction 
          ~target_shape:(Tensor.shape grid) in
        Tensor.add smoothed correction
    in
    v_cycle grid.p 5

  let compute_divergence u v ~dx ~dy =
    let nx, ny = Tensor.shape u |> fun [|x;y|] -> x-1, y in
    let div = Tensor.zeros [|nx; ny|] in
    for i = 1 to nx-2 do
      for j = 1 to ny-2 do
        let du_dx = Tensor.(sub 
          (get u (i+1) j)
          (get u i j)) |> Tensor.div_float dx in
        let dv_dy = Tensor.(sub
          (get v i (j+1))
          (get v i j)) |> Tensor.div_float dy in
        Tensor.set div i j (Tensor.add du_dx dv_dy)
      done
    done;
    div

  let project_velocity grid method_ ~dt =
    match method_ with
    | Chorin ->
        grid
    | VanKan -> 
        grid
    | Incremental ->
        grid
end

module Boundary = struct
  type staggered_boundary = {
    u_boundary: Tensor.t -> int -> int -> float;
    v_boundary: Tensor.t -> int -> int -> float;
    p_boundary: Tensor.t -> int -> int -> float;
    normal_derivative: Tensor.t -> int -> int -> float;
  }

  let create_wall_boundary ~u_inf = {
    u_boundary = (fun _ _ _ -> 0.0);  (* No-slip *)
    v_boundary = (fun _ _ _ -> 0.0);  (* No-slip *)
    p_boundary = (fun p i j ->      (* Neumann *)
      let nx, ny = Tensor.shape p |> fun [|x;y|] -> x, y in
      if i = 0 then Tensor.float_value (Tensor.get p 1 j)
      else if i = nx-1 then Tensor.float_value (Tensor.get p (nx-2) j)
      else if j = 0 then Tensor.float_value (Tensor.get p i 1)
      else Tensor.float_value (Tensor.get p i (ny-2))
    );
    normal_derivative = (fun _ _ _ -> 0.0);
  }

  let create_inflow_boundary ~u_inf ~v_inf = {
    u_boundary = (fun _ _ _ -> u_inf);
    v_boundary = (fun _ _ _ -> v_inf);
    p_boundary = (fun p i j ->
      let nx, _ = Tensor.shape p |> fun [|x;y|] -> x, y in
      if i = 0 then Tensor.float_value (Tensor.get p 1 j)
      else Tensor.float_value (Tensor.get p (nx-2) j)
    );
    normal_derivative = (fun _ _ _ -> 0.0);
  }

  let create_outflow_boundary = {
    u_boundary = (fun u i j ->
      let nx, _ = Tensor.shape u |> fun [|x;y|] -> x, y in
      Tensor.float_value (Tensor.get u (nx-2) j)
    );
    v_boundary = (fun v i j ->
      let nx, _ = Tensor.shape v |> fun [|x;y|] -> x, y in
      Tensor.float_value (Tensor.get v (nx-2) j)
    );
    p_boundary = (fun _ _ _ -> 0.0);  (* Zero pressure at outlet *)
    normal_derivative = (fun _ _ _ -> 0.0);
  }
end

module Timestepping = struct
  type scheme = 
    | ForwardEuler
    | AdamsBashforth2
    | AdamsBashforth3
    | RungeKutta4

  type adaptivity = {
    base_dt: float;
    min_dt: float;
    max_dt: float;
    cfl: float;
    tolerance: float;
  }

  let compute_stable_dt grid ~viscosity ~adaptivity =
    let u_max = Tensor.max_all grid.Pressure_velocity.u 
                |> Tensor.float_value in
    let v_max = Tensor.max_all grid.Pressure_velocity.v 
                |> Tensor.float_value in
    let vel_max = max u_max v_max in
    
    let dx = grid.Pressure_velocity.dx in
    let dy = grid.Pressure_velocity.dy in
    let dx_min = min dx dy in
    
    (* CFL condition *)
    let dt_convection = adaptivity.cfl *. dx_min /. vel_max in
    
    (* Viscous stability *)
    let dt_viscous = 0.5 *. dx_min *. dx_min /. viscosity in
    
    min dt_convection dt_viscous
    |> min adaptivity.max_dt
    |> max adaptivity.min_dt

  let step scheme grid ~dt ~update_fn =
    match scheme with
    | ForwardEuler ->
        let dudt = update_fn grid in
        let next = Tensor.add grid.Pressure_velocity.u 
          (Tensor.mul_float dudt dt) in
        { grid with Pressure_velocity.u = next }
        
    | AdamsBashforth2 ->
        (* Two-step Adams-Bashforth *)
        let dudt_n = update_fn grid in
        let next = Tensor.add grid.Pressure_velocity.u
          (Tensor.mul_float dudt_n (1.5 *. dt)) in
        { grid with Pressure_velocity.u = next }
        
    | AdamsBashforth3 ->
        (* Three-step Adams-Bashforth *)
        let dudt_n = update_fn grid in
        let next = Tensor.add grid.Pressure_velocity.u
          (Tensor.mul_float dudt_n (23./12. *. dt)) in
        { grid with Pressure_velocity.u = next }
        
    | RungeKutta4 ->
        (* Classical RK4 *)
        let k1 = update_fn grid in
        let temp1 = {
          grid with Pressure_velocity.u = 
            Tensor.add grid.Pressure_velocity.u 
              (Tensor.mul_float k1 (dt/.2.))
        } in
        
        let k2 = update_fn temp1 in
        let temp2 = {
          grid with Pressure_velocity.u = 
            Tensor.add grid.Pressure_velocity.u 
              (Tensor.mul_float k2 (dt/.2.))
        } in
        
        let k3 = update_fn temp2 in
        let temp3 = {
          grid with Pressure_velocity.u = 
            Tensor.add grid.Pressure_velocity.u 
              (Tensor.mul_float k3 dt)
        } in
        
        let k4 = update_fn temp3 in
        
        let du = Tensor.(add (add 
          (mul_float k1 (dt/.6.))
          (mul_float k2 (dt/.3.)))
          (add
            (mul_float k3 (dt/.3.))
            (mul_float k4 (dt/.6.)))) in
            
        { grid with Pressure_velocity.u = 
            Tensor.add grid.Pressure_velocity.u du }
end

module Conservation = struct
  let check_mass_conservation grid =
    let div = Pressure_velocity.compute_divergence 
      grid.Pressure_velocity.u 
      grid.Pressure_velocity.v 
      ~dx:grid.Pressure_velocity.dx 
      ~dy:grid.Pressure_velocity.dy in
    Tensor.sum div |> Tensor.float_value

  let check_momentum_conservation grid =
    let total_momentum_x = Tensor.sum grid.Pressure_velocity.u in
    let total_momentum_y = Tensor.sum grid.Pressure_velocity.v in
    total_momentum_x, total_momentum_y

  let check_energy_conservation grid =
    let kinetic_energy = Tensor.(add
      (mul grid.Pressure_velocity.u grid.Pressure_velocity.u)
      (mul grid.Pressure_velocity.v grid.Pressure_velocity.v))
      |> Tensor.sum
      |> Tensor.div_float 2. in
    
    let pressure_work = Tensor.mul grid.Pressure_velocity.p
      (Pressure_velocity.compute_divergence 
         grid.Pressure_velocity.u 
         grid.Pressure_velocity.v 
         ~dx:grid.Pressure_velocity.dx 
         ~dy:grid.Pressure_velocity.dy)
      |> Tensor.sum in
    
    kinetic_energy, pressure_work
end

module Memory_manager = struct
  type allocation = {
    tensor: Tensor.t;
    size: int;
    device: Device.t;
    last_used: float;
    priority: int;
  }

  type t = {
    allocations: (string, allocation) Hashtbl.t;
    device_limits: (Device.t * int) list;
    gc_threshold: float;
  }

  let create ~device_limits ~gc_threshold = {
    allocations = Hashtbl.create 256;
    device_limits;
    gc_threshold;
  }

  let memory_pressure t device =
    let total_allocated = Hashtbl.fold (fun _ alloc acc ->
      if alloc.device = device then acc + alloc.size else acc
    ) t.allocations 0 in
    let device_limit = List.assoc device t.device_limits in
    float_of_int total_allocated /. float_of_int device_limit

  let garbage_collect t =
    let current_time = Unix.gettimeofday () in
    let to_free = Hashtbl.fold (fun key alloc acc ->
      if current_time -. alloc.last_used > t.gc_threshold then
        (key, alloc) :: acc
      else acc
    ) t.allocations [] in
    List.iter (fun (key, _) ->
      Hashtbl.remove t.allocations key
    ) to_free

  let allocate t ~name ~size ~device ~priority =
    let pressure = memory_pressure t device in
    if pressure > 0.9 then garbage_collect t;
    
    let tensor = match device with
      | Device.CPU -> Tensor.zeros [|size|]
      | Device.GPU id -> 
          Tensor.zeros [|size|] 
          |> Tensor.to_device (Device.get_device (GPU id))
    in
    
    let alloc = {
      tensor;
      size;
      device;
      last_used = Unix.gettimeofday ();
      priority;
    } in
    Hashtbl.add t.allocations name alloc;
    tensor
end

module Error_handler = struct
  type error_level = Warning | Error | Critical

  type error = {
    level: error_level;
    code: string;
    message: string;
    context: (string * string) list;
    timestamp: float;
  }

  type handler = {
    log: error -> unit;
    recover: error -> unit;
    notify: error -> unit;
  }

  let create_handler ~log_file ~notify_fn = {
    log = (fun error ->
      let oc = open_out_gen [Open_append; Open_creat] 0o666 log_file in
      Printf.fprintf oc "[%f] %s: %s\nContext: %s\n%!"
        error.timestamp
        (match error.level with
         | Warning -> "WARNING"
         | Error -> "ERROR"
         | Critical -> "CRITICAL")
        error.message
        (String.concat ", " 
           (List.map (fun (k, v) -> k ^ ": " ^ v) error.context));
      close_out oc
    );
    
    recover = (fun error ->
      match error.level with 
      | Warning -> ()
      | Error -> 
          Gc.compact ()
      | Critical ->
          raise (Failure error.message)
    );
    
    notify = notify_fn;
  }
end

module Profiler = struct
  type timing = {
    start_time: float;
    end_time: float option;
    name: string;
    metadata: (string * string) list;
  }

  type t = {
    mutable timings: timing list;
    mutable current: timing option;
  }

  let create () = {
    timings = [];
    current = None;
  }

  let start t name metadata =
    let timing = {
      start_time = Unix.gettimeofday ();
      end_time = None;
      name;
      metadata;
    } in
    t.current <- Some timing;
    timing

  let stop t timing =
    let end_time = Unix.gettimeofday () in
    let completed = { timing with end_time = Some end_time } in
    t.timings <- completed :: t.timings;
    t.current <- None

  let report t =
    List.map (fun timing ->
      let duration = match timing.end_time with
        | Some end_time -> end_time -. timing.start_time
        | None -> Unix.gettimeofday () -. timing.start_time
      in
      (timing.name, duration, timing.metadata)
    ) t.timings
end

module Resource_monitor = struct
  type resource_type =
    | Memory of Device.t
    | Computation of Device.t
    | Storage of string
    | Network

  type threshold = {
    warning_level: float;
    critical_level: float;
    duration: float option;
  }

  type monitor = {
    resource: resource_type;
    current_usage: unit -> float;
    threshold: threshold;
    last_warning: float option ref;
  }

  let create_memory_monitor device threshold = {
    resource = Memory device;
    current_usage = (fun () ->
      match Device.memory_info device with
      | Some (total, free) -> 
          float_of_int (total - free) /. float_of_int total
      | None -> 0.0
    );
    threshold;
    last_warning = ref None;
  }

  let create_compute_monitor device threshold = {
    resource = Computation device;
    current_usage = (fun () ->
      match device with
      | Device.GPU id ->
          float_of_int (Cuda.utilization id) /. 100.0
      | Device.CPU ->
          let stats = Unix.times () in
          let total = stats.Unix.tms_utime +. stats.Unix.tms_stime in
          total /. Unix.gettimeofday ()
    );
    threshold;
    last_warning = ref None;
  }

  let check_thresholds monitor =
    let usage = monitor.current_usage () in
    let current_time = Unix.gettimeofday () in
    
    let exceeds_duration = match monitor.threshold.duration with
      | None -> true
      | Some duration ->
          match !(monitor.last_warning) with
          | None -> true
          | Some last_time -> 
              current_time -. last_time >= duration
    in
    
    if usage >= monitor.threshold.critical_level && exceeds_duration then
      Some (`Critical, usage)
    else if usage >= monitor.threshold.warning_level && exceeds_duration then
      Some (`Warning, usage)
    else
      None
end

module Multi_grid = struct
  type t = {
    grids: Grid_representation.t array;
    shifts: float array;
    feature_extractors: Feature_extractor.t array;
  }

  let create ~config =
    let base_grid = match config.dimension with
      | Types.D1 h -> [|h|]
      | Types.D2 (h, w) -> [|h; w|]
      | Types.D3 (h, w, d) -> [|h; w; d|] in
    
    let grids = Array.init config.num_grids (fun i ->
      let scale = float_of_int (i + 1) /. float_of_int config.num_grids in
      let dims = Array.map (fun d -> int_of_float (float_of_int d *. scale)) base_grid in
      Grid_representation.create ~channel_size:config.channel_size ~dims
    ) in

    let feature_extractors = Array.init config.num_grids (fun _ ->
      Feature_extractor.create
        ~grid_sampler:(Grid_sampler.create 
                        ~support_size:4 
                        ~kernel:Types.Cosine)
        ~feature_dim:config.channel_size
    ) in

    { grids; shifts = config.grid_shifts; feature_extractors }

  let forward t ~coords =
    Array.mapi (fun i grid ->
      let shifted_coords = 
        Tensor.add coords (Tensor.float t.shifts.(i)) in
      Feature_extractor.extract 
        t.feature_extractors.(i) 
        ~grid:grid.grid 
        ~coords:shifted_coords
    ) t.grids
    |> Array.fold_left Tensor.add (Tensor.zeros_like (Array.get t.grids 0).grid)
end

module Pixel = struct
  type t = {
    multi_grid: Multi_grid.t;
    neural_net: Neural_net.t;
    grid_config: Types.grid_config;
  }

  let create ~grid_config ~net_config =
    let multi_grid = Multi_grid.create ~config:grid_config in
    let neural_net = Neural_net.create 
      ~dims:net_config
      ~activation:Tensor.relu in
    { multi_grid; neural_net; grid_config }

  let forward t ~coords =
    let features = Multi_grid.forward t.multi_grid ~coords in
    Neural_net.forward t.neural_net features
end

module System = struct
  type config = {
    memory_manager: Memory_manager.t;
    error_handler: Error_handler.handler;
    profiler: Profiler.t;
  }

  let initialize ~device_limits ~log_file ~notify_fn = {
    memory_manager = Memory_manager.create 
      ~device_limits 
      ~gc_threshold:300.0;
    error_handler = Error_handler.create_handler 
      ~log_file 
      ~notify_fn;
    profiler = Profiler.create ();
  }

  let with_profiling config name metadata f =
    let timing = Profiler.start config.profiler name metadata in
    try
      let result = f () in
      Profiler.stop config.profiler timing;
      result
    with e ->
      let error = Error_handler.{
        level = Critical;
        code = "PROF_ERROR";
        message = Printexc.to_string e;
        context = metadata;
        timestamp = Unix.gettimeofday ();
      } in
      config.error_handler.log error;
      config.error_handler.notify error;
      raise e

  let allocate_tensor config ~name ~size ~device ~priority =
    try 
      Memory_manager.allocate 
        config.memory_manager 
        ~name ~size ~device ~priority
    with e ->
      let error = Error_handler.{
        level = Error;
        code = "MEM_ERROR";
        message = Printexc.to_string e;
        context = [("allocation_size", string_of_int size)];
        timestamp = Unix.gettimeofday ();
      } in
      config.error_handler.log error;
      config.error_handler.notify error;
      raise e
end

module Train = struct
  type training_config = {
    learning_rate: float;
    batch_size: int;
    max_epochs: int;
    validation_freq: int;
    checkpoint_freq: int;
    early_stopping_patience: int;
  }

  let create_optimizer parameters ~learning_rate =
    Optimizer.adam parameters ~lr:learning_rate

  let train_epoch model optimizer data_loader loss_fn =
    let total_loss = ref 0. in
    let count = ref 0 in
    
    Dataset.iter data_loader (fun batch ->
      Optimizer.zero_grad optimizer;
      let loss = loss_fn model batch in
      Tensor.backward loss;
      Optimizer.step optimizer;
      total_loss := !total_loss +. Tensor.float_value loss;
      incr count
    );
    
    !total_loss /. float_of_int !count

  let train model config data_loader validation_loader loss_fn =
    let optimizer = create_optimizer (Neural_net.parameters model) 
                     ~learning_rate:config.learning_rate in
    
    let rec training_loop epoch best_loss patience =
      if epoch >= config.max_epochs || patience <= 0 then
        model
      else begin
        let train_loss = train_epoch model optimizer data_loader loss_fn in
        
        if epoch mod config.validation_freq = 0 then
          let val_loss = validate model validation_loader loss_fn in
          if val_loss < best_loss then
            training_loop (epoch + 1) val_loss config.early_stopping_patience
          else
            training_loop (epoch + 1) best_loss (patience - 1)
        else
          training_loop (epoch + 1) best_loss patience
      end
    in
    
    training_loop 0 Float.infinity config.early_stopping_patience
end
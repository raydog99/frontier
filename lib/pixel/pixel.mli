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

module Device : sig
  type t = CPU | GPU of int
  val default : t
  val get_device : t -> Device.t
  val memory_info : t -> (int * int) option
end

val gradient : Tensor.t -> wrt:Tensor.t -> Tensor.t
val laplacian : Tensor.t -> wrt:Tensor.t -> Tensor.t
val partial_t : Tensor.t -> t:Tensor.t -> Tensor.t
val partial_x : Tensor.t -> x:Tensor.t -> Tensor.t
val partial_xx : Tensor.t -> x:Tensor.t -> Tensor.t
val compute_high_order_derivative : Tensor.t -> order:int -> wrt:Tensor.t -> Tensor.t

module Grid_sampler : sig
  type t = {
    support_size: int;
    kernel: Types.interpolation_kernel;
  }

  val create : support_size:int -> kernel:Types.interpolation_kernel -> t
  val sample : t -> grid:Tensor.t -> coords:Tensor.t -> Tensor.t
  val sample_with_gradients : t -> grid:Tensor.t -> coords:Tensor.t -> 
    Tensor.t * Tensor.t * Tensor.t (* value, dx, dy *)
end

module Feature_extractor : sig
  type t = {
    grid_sampler: Grid_sampler.t;
    feature_dim: int;
  }

  val create : grid_sampler:Grid_sampler.t -> feature_dim:int -> t
  val extract : t -> grid:Tensor.t -> coords:Tensor.t -> Tensor.t
end

module Domain : sig
  type t = {
    physical_bounds: float array * float array;  (* min, max *)
    computational_bounds: float array * float array;
    mapping: Tensor.t -> Tensor.t;
    inverse_mapping: Tensor.t -> Tensor.t;
  }

  val create_arbitrary_domain : physical_bounds:(float array * float array) ->
                              computational_bounds:(float array * float array) -> t
  val normalize_coords : t -> coords:Tensor.t -> Tensor.t
end

module Grid_representation : sig
  type t = {
    grid: Tensor.t;
    channel_size: int;
    height: int;
    width: int;
  }

  val create : channel_size:int -> dims:int array -> t
  val get : t -> int -> int -> Tensor.t
  val set : t -> int -> int -> Tensor.t -> unit
  val interpolate : t -> x:Tensor.t -> y:Tensor.t -> Tensor.t
  val width : t -> int
  val height : t -> int
end

module Adaptive_refinement : sig
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

  val compute_refinement_indicator : refined_cell -> Tensor.t * Tensor.t
  val should_refine : refinement_criteria -> refined_cell -> bool
  val refine_cell : refined_cell -> refined_cell
end

module Multi_grid : sig
  type t = {
    grids: Grid_representation.t array;
    shifts: float array;
    feature_extractors: Feature_extractor.t array;
  }

  val create : config:Types.grid_config -> t
  val forward : t -> coords:Tensor.t -> Tensor.t
end

module Neural_net : sig
  type t = {
    layers: Tensor.t array;
    biases: Tensor.t array;
    activation: Tensor.t -> Tensor.t;
  }

  val create : dims:int array -> activation:(Tensor.t -> Tensor.t) -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module PDE : sig
	type params
    val residual : params -> (x:Tensor.t -> t:Tensor.t -> Tensor.t) -> 
                  x:Tensor.t -> t:Tensor.t -> Tensor.t

  module Helmholtz : sig
    type params = {
      k: float;
      a1: float;
      a2: float;
    }
  end

  module Allen_Cahn : sig
    type params = {
      epsilon: float;
      lambda: float;
    }
  end

  module Navier_Stokes : sig
    type params = {
      nu: float;
      rho: float;
    }
    val residual : params -> 
      (x:Tensor.t -> y:Tensor.t -> t:Tensor.t -> Tensor.t * Tensor.t * Tensor.t) -> 
      x:Tensor.t -> y:Tensor.t -> t:Tensor.t -> Tensor.t
  end

  module KdV : sig
    type params = {
      alpha: float;
      beta: float;
    }
  end
end

module Pressure_velocity : sig
  type projection_method = Chorin | VanKan | Incremental
  type staggered_grid = {
    u: Tensor.t;
    v: Tensor.t;
    w: Tensor.t option;
    p: Tensor.t;
    dx: float;
    dy: float;
    dz: float option;
  }

  val create_staggered_grid : nx:int -> ny:int -> ?nz:int -> 
                             dx:float -> dy:float -> ?dz:float -> unit -> staggered_grid
  val project_velocity : staggered_grid -> projection_method -> dt:float -> staggered_grid
  val solve_pressure_poisson : staggered_grid -> rhs:Tensor.t -> Tensor.t
  val compute_divergence : Tensor.t -> Tensor.t -> dx:float -> dy:float -> Tensor.t
end

module Boundary : sig
  type staggered_boundary = {
    u_boundary: Tensor.t -> int -> int -> float;
    v_boundary: Tensor.t -> int -> int -> float;
    p_boundary: Tensor.t -> int -> int -> float;
    normal_derivative: Tensor.t -> int -> int -> float;
  }

  val create_wall_boundary : u_inf:float -> staggered_boundary
  val create_inflow_boundary : u_inf:float -> v_inf:float -> staggered_boundary
  val create_outflow_boundary : staggered_boundary
end

module Timestepping : sig
  type scheme = ForwardEuler | AdamsBashforth2 | AdamsBashforth3 | RungeKutta4
  
  type adaptivity = {
    base_dt: float;
    min_dt: float;
    max_dt: float;
    cfl: float;
    tolerance: float;
  }

  val compute_stable_dt : Pressure_velocity.staggered_grid -> 
                         viscosity:float -> adaptivity:adaptivity -> float
  val step : scheme -> Pressure_velocity.staggered_grid -> 
            dt:float -> update_fn:(Pressure_velocity.staggered_grid -> Tensor.t) ->
            Pressure_velocity.staggered_grid
end

module Memory_manager : sig
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

  val create : device_limits:(Device.t * int) list -> gc_threshold:float -> t
  val allocate : t -> name:string -> size:int -> device:Device.t -> priority:int -> Tensor.t
  val garbage_collect : t -> unit
end

module Error_handler : sig
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

  val create_handler : log_file:string -> notify_fn:(error -> unit) -> handler
end

module Profiler : sig
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

  val create : unit -> t
  val start : t -> string -> (string * string) list -> timing
  val stop : t -> timing -> unit
  val report : t -> (string * float * (string * string) list) list
end

module Resource_monitor : sig
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

  type monitor
  val create_memory_monitor : Device.t -> threshold -> monitor
  val create_compute_monitor : Device.t -> threshold -> monitor
  val check_thresholds : monitor -> ([`Warning | `Critical] * float) option
end

module Pixel : sig
  type t = {
    multi_grid: Multi_grid.t;
    neural_net: Neural_net.t;
    grid_config: Types.grid_config;
  }

  val create : grid_config:Types.grid_config -> net_config:int array -> t
  val forward : t -> coords:Tensor.t -> Tensor.t
end


module System : sig
  type config = {
    memory_manager: Memory_manager.t;
    error_handler: Error_handler.handler;
    profiler: Profiler.t;
  }

  val initialize : device_limits:(Device.t * int) list -> 
                  log_file:string -> 
                  notify_fn:(Error_handler.error -> unit) -> config
  val with_profiling : config -> string -> (string * string) list -> 
                      (unit -> 'a) -> 'a
  val allocate_tensor : config -> name:string -> size:int -> 
                       device:Device.t -> priority:int -> Tensor.t
end

module Train : sig
  type training_config = {
    learning_rate: float;
    batch_size: int;
    max_epochs: int;
    validation_freq: int;
    checkpoint_freq: int;
    early_stopping_patience: int;
  }

  val create_optimizer : Tensor.t list -> learning_rate:float -> Optimizer.t
  val train_epoch : Neural_net.t -> Optimizer.t -> 
                   Dataset.t -> (Neural_net.t -> 'a -> Tensor.t) -> float
  val train : Neural_net.t -> training_config -> Dataset.t -> Dataset.t -> 
              (Neural_net.t -> 'a -> Tensor.t) -> Neural_net.t
end

module Conservation : sig
  val check_mass_conservation : Pressure_velocity.staggered_grid -> float
  val check_momentum_conservation : 
    Pressure_velocity.staggered_grid -> Tensor.t * Tensor.t
  val check_energy_conservation : 
    Pressure_velocity.staggered_grid -> Tensor.t * Tensor.t
end
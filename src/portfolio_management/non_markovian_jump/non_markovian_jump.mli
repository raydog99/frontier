open Torch

(** Configuration for the non-Markovian jump process *)
module Config : sig
  type t = private {
    initial_value : float;
    wave_numbers : Tensor.t;
    pdf_size : int;
    km_order : int;
    device : Device.t;
    adaptive_stepping : bool;
    tolerance : float;
    max_iterations : int;
    learning_rate : float;
    num_epochs : int;
    batch_size : int;
    use_gpu : bool;
    advanced_integrator : [ `EulerMaruyama | `Milstein | `StrongTaylor15 ];
    report_frequency : int;
    log_level : Logs.level;
    num_threads : int;
    distributed : bool;
    error_handling : [ `Raise | `Warn | `Ignore ];
    seed : int option;
  }

  val create : 
    ?initial_value:float ->
    ?wave_numbers:Tensor.t ->
    ?pdf_size:int ->
    ?km_order:int ->
    ?device:Device.t ->
    ?adaptive_stepping:bool ->
    ?tolerance:float ->
    ?max_iterations:int ->
    ?learning_rate:float ->
    ?num_epochs:int ->
    ?batch_size:int ->
    ?use_gpu:bool ->
    ?advanced_integrator:[ `EulerMaruyama | `Milstein | `StrongTaylor15 ] ->
    ?report_frequency:int ->
    ?log_level:Logs.level ->
    ?num_threads:int ->
    ?distributed:bool ->
    ?error_handling:[ `Raise | `Warn | `Ignore ] ->
    ?seed:int ->
    unit -> t
  (** Create a new configuration with optional parameters. *)
end

(** Auxiliary field for the non-Markovian jump process *)
module AuxiliaryField : sig
  type t = private {
    mutable z : Tensor.t;
    wave_numbers : Tensor.t;
  }
end

(** Main non-Markovian jump process type *)
module Process : sig
  type t = private {
    mutable time : float;
    mutable value : Tensor.t;
    intensity : Tensor.t -> Tensor.t -> AuxiliaryField.t -> Tensor.t;
    auxiliary_field : AuxiliaryField.t;
    mutable pdf : Tensor.t;
    mutable km_coefficients : Tensor.t array;
    mutable memory_kernel : Tensor.t;
    config : Config.t;
    mutable rng : Torch.Generator.t;
  }

  val create : (Tensor.t -> Tensor.t -> AuxiliaryField.t -> Tensor.t) -> Config.t -> t
  (** Create a new non-Markovian jump process. *)

  val step : t -> float -> (unit, error) result
  (** Perform a single step of the process. Returns Ok () on success, or Error on failure. *)

  val simulate : t -> float -> float -> ((float * float) list, error) result
  (** Simulate the process for a given duration and time step. *)

  val parallel_simulate : t -> float -> float -> ((float * float) list, error) result
  (** Simulate the process in parallel for a given duration and time step. *)

  val kramers_moyal_expansion : t -> unit
  (** Perform Kramers-Moyal expansion on the process. *)

  val field_master_equation : t -> float -> unit
  (** Solve the field master equation for a given time step. *)

  val system_size_expansion : t -> float -> unit
  (** Perform system size expansion on the process. *)

  val generalized_langevin_equation : t -> float -> float -> float -> unit
  (** Solve the generalized Langevin equation for the process. *)

  val stability_analysis : t -> unit
  (** Perform stability analysis on the process. *)

  val compute_autocorrelation : t -> int -> Tensor.t
  (** Compute the autocorrelation function of the process up to a given lag. *)

  val compute_power_spectrum : t -> Tensor.t
  (** Compute the power spectrum of the process. *)

  val compute_fractal_dimension : t -> float
  (** Compute the fractal dimension of the process trajectory. *)

  val estimate_parameters : t -> (float * float) list -> Tensor.t
  (** Estimate parameters of the process given observed data. *)

  val cross_validate : t -> (float * float) list -> int -> (Tensor.t * float) list
  (** Perform cross-validation on the process. *)

  val confidence_intervals : t -> (float * float) list -> int -> (float * float) list
  (** Compute confidence intervals for the process parameters. *)

  val information_criteria : t -> (float * float) list -> float * float
  (** Compute information criteria (AIC and BIC) for the process. *)

  val residual_analysis : t -> (float * float) list -> float * float * float * float list
  (** Perform residual analysis on the process. *)

  val mfdfa : t -> float list -> int list -> Tensor.t * Tensor.t
  (** Perform Multifractal Detrended Fluctuation Analysis on the process. *)

  val auto_hyperparameter_tuning : t -> (float * float) list -> t
  (** Automatically tune hyperparameters of the process. *)

  val generate_report : t -> (float * float) list -> unit
  (** Generate a comprehensive report about the process. *)

  val interactive_tuning : t -> (float * float) list -> Tensor.t
  (** Interactively tune the process parameters. *)
end

(** Distributed computing utilities *)
module Distributed : sig
  type node = {
    id : int;
    address : string;
    port : int;
  }

  val register_node : int -> string -> int -> unit
  (** Register a new node for distributed computing. *)

  val distribute_computation : ('a -> 'b) -> 'a list -> 'b list Lwt.t
  (** Distribute a computation across registered nodes. *)
end

(** Error handling *)
type error =
  | InvalidParameter of string
  | NumericalInstability of string
  | ComputationError of string
  | OptimizationError of string
  | GPUError of string
  | DistributedComputingError of string

val handle_error : Config.t -> error -> unit
(** Handle an error according to the configuration's error handling policy. *)

(** Utility functions *)
val safe_div : Tensor.t -> Tensor.t -> Tensor.t
(** Safely divide two tensors, avoiding division by zero. *)

val clip_tensor : Tensor.t -> float -> float -> Tensor.t
(** Clip a tensor to be within a specified range. *)

(** Parallel processing utilities *)
val parallel_map : ('a -> 'b) -> 'a list -> 'b list
(** Map a function over a list in parallel. *)

(** Model selection *)
val bayesian_model_selection : Process.t list -> (float * float) list -> (Process.t * float) list
(** Perform Bayesian model selection on a list of processes. *)
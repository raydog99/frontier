val serialize_sample : 
  Type.posterior_sample -> int -> int -> Type.sample_storage
(** [serialize_sample s chain_id iter] converts sample to storable format *)

val save_samples : 
  string -> Type.posterior_sample list -> int -> unit
(** [save_samples filename samples chain_id] saves samples to file *)

val load_samples : string -> Type.sample_storage list
(** [load_samples filename] loads samples from file *)
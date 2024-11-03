val train_vae : vae:VAE.t -> data:TimeSeries.t -> config:model_config -> unit
val train_flow : flow:FlowNetwork.t -> vae:VAE.t option -> config:model_config -> data:TimeSeries.t -> unit
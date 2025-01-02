open Krasnoselskii_mann

val send_metrics_to_prometheus : iteration_result -> string -> unit
val log_to_elasticsearch : iteration_result -> string -> unit
val notify_slack : iteration_result -> string -> unit
open Krasnoselskii_mann

let send_metrics_to_prometheus result endpoint =
  Printf.printf "Sending metrics to Prometheus at %s\n" endpoint;
  List.iter (fun (metric, value) ->
    Printf.printf "  %s: %f\n" metric value
  ) result.performance_metrics

let log_to_elasticsearch result endpoint =
  Printf.printf "Logging results to Elasticsearch at %s\n" endpoint;
  Printf.printf "Final residual: %f\n" result.final_residual;
  Printf.printf "Iterations: %d\n" result.iterations

let notify_slack result webhook_url =
  Printf.printf "Sending notification to Slack webhook: %s\n" webhook_url;
  Printf.printf "Algorithm %s after %d iterations\n" 
    (if result.converged then "converged" else "did not converge")
    result.iterations
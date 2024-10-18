type t = {
  n_assets: int;
  n_timepoints: int;
  target_return: float;
  covariance_method: [ `Tracy_Widom | `Linear_Shrinkage | `Naive ];
  optimization_method: [ `Markowitz | `NCO ];
}

let default = {
  n_assets = 50;
  n_timepoints = 1000;
  target_return = 0.1;
  covariance_method = `Tracy_Widom;
  optimization_method = `NCO;
}

let create ?n_assets ?n_timepoints ?target_return ?covariance_method ?optimization_method () =
  {
    n_assets = Option.value n_assets ~default:default.n_assets;
    n_timepoints = Option.value n_timepoints ~default:default.n_timepoints;
    target_return = Option.value target_return ~default:default.target_return;
    covariance_method = Option.value covariance_method ~default:default.covariance_method;
    optimization_method = Option.value optimization_method ~default:default.optimization_method;
  }
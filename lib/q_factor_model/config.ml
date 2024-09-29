type t = {
  start_year: int;
  end_year: int;
  confidence_level: float;
  n_bootstrap: int;
  newey_west_lags: int;
  train_ratio: float;
  rolling_window_size: int;
  rolling_window_step: int;
  risk_free_rate: float;
}

let default = {
  start_year = 1967;
  end_year = 2016;
  confidence_level = 0.95;
  n_bootstrap = 1000;
  newey_west_lags = 12;
  train_ratio = 0.8;
  rolling_window_size = 60;
  rolling_window_step = 12;
  risk_free_rate = 0.02;
}

let create ?start_year ?end_year ?confidence_level ?n_bootstrap ?newey_west_lags
           ?train_ratio ?rolling_window_size ?rolling_window_step ?risk_free_rate () =
  {
    start_year = Option.value start_year ~default:default.start_year;
    end_year = Option.value end_year ~default:default.end_year;
    confidence_level = Option.value confidence_level ~default:default.confidence_level;
    n_bootstrap = Option.value n_bootstrap ~default:default.n_bootstrap;
    newey_west_lags = Option.value newey_west_lags ~default:default.newey_west_lags;
    train_ratio = Option.value train_ratio ~default:default.train_ratio;
    rolling_window_size = Option.value rolling_window_size ~default:default.rolling_window_size;
    rolling_window_step = Option.value rolling_window_step ~default:default.rolling_window_step;
    risk_free_rate = Option.value risk_free_rate ~default:default.risk_free_rate;
  }
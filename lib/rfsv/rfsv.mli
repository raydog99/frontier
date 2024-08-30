open Base
open Torch

type params = {
  h : float;    (** Hurst parameter *)
  nu : float;   (** Volatility of volatility *)
  alpha : float; (** Mean reversion rate *)
  m : float;    (** Long-term mean *)
}

exception Invalid_parameter of string

exception Insufficient_data of string

(** Generate a sample path from the RFSV model
    @param params The parameters of the RFSV model
    @param n The number of points to generate
    @return A tensor representing the generated sample path
*)
val generate_rfsv : params -> int -> Tensor.t

(** Estimate the Hurst parameter from data
    @param data The input data tensor
    @return The estimated Hurst parameter
*)
val estimate_h : Tensor.t -> float

(** Forecast log-volatility using the RFSV model
    @param data The input data tensor
    @param h The Hurst parameter
    @param horizon The forecast horizon
    @return The forecasted log-volatility
*)
val forecast_log_volatility : Tensor.t -> float -> float -> float

(** Forecast variance using the RFSV model
    @param data The input data tensor
    @param h The Hurst parameter
    @param nu The volatility of volatility
    @param horizon The forecast horizon
    @return The forecasted variance
*)
val forecast_variance : Tensor.t -> float -> float -> float -> float

(** Forecast using the AR(p) model
    @param data The input data tensor
    @param p The order of the AR model
    @param horizon The forecast horizon
    @return The forecasted value
*)
val forecast_ar : Tensor.t -> int -> int -> float

(** Forecast using the HAR model
    @param data The input data tensor
    @param horizon The forecast horizon
    @return The forecasted value
*)
val forecast_har : Tensor.t -> int -> float

(** Prepare data for forecasting
    @param data The input data tensor
    @param window_size The size of the sliding window
    @param horizon The forecast horizon
    @return A tuple of input and target tensors
*)
val prepare_forecast_data : Tensor.t -> int -> int -> Tensor.t * Tensor.t

(** Calculate the mean squared error between predictions and targets
    @param predictions The predicted values
    @param targets The actual values
    @return The mean squared error
*)
val mse : Tensor.t -> Tensor.t -> Tensor.t

(** Calculate the P-ratio (MSE / variance of targets)
    @param predictions The predicted values
    @param targets The actual values
    @return The P-ratio
*)
val calculate_p_ratio : Tensor.t -> Tensor.t -> Tensor.t

(** Evaluate a forecast function using rolling window approach
    @param forecast_fn The forecasting function to evaluate
    @param data The input data tensor
    @param window_size The size of the sliding window
    @param horizon The forecast horizon
    @return The P-ratio of the forecast
*)
val evaluate_forecast : (Tensor.t -> int -> float) -> Tensor.t -> int -> int -> Tensor.t

(** Compare different forecasting methods
    @param data The input data tensor
    @param window_size The size of the sliding window
    @param horizons A list of forecast horizons to evaluate
*)
val compare_forecasts : Tensor.t -> int -> int list -> unit

(** Load data from a CSV file
    @param filename The path to the CSV file
    @return A tensor containing the loaded data
*)
val load_csv_data : string -> Tensor.t

(** Preprocess the input data (take log and compute returns)
    @param data The input data tensor
    @return A tensor of log returns
*)
val preprocess_data : Tensor.t -> Tensor.t

(** Estimate the volatility of volatility parameter
    @param data The input data tensor
    @return The estimated nu parameter
*)
val estimate_nu : Tensor.t -> float

(** Estimate the mean reversion rate parameter
    @param data The input data tensor
    @return The estimated alpha parameter
*)
val estimate_alpha : Tensor.t -> float

(** Estimate all RFSV model parameters from data
    @param data The input data tensor
    @return The estimated parameters
*)
val estimate_params : Tensor.t -> params

(** Convert a tensor to a list of floats
    @param t The input tensor
    @return A list of floats
*)
val tensor_to_list : Tensor.t -> float list

(** Convert a list of floats to a tensor
    @param l The input list of floats
    @return A tensor
*)
val list_to_tensor : float list -> Tensor.t

(** Calculate rolling volatility
    @param data The input data tensor
    @param window_size The size of the rolling window
    @return A tensor of rolling volatility estimates
*)
val rolling_volatility : Tensor.t -> int -> Tensor.t

(** Calculate Value at Risk (VaR)
    @param data The input data tensor
    @param confidence_level The confidence level (e.g., 0.95 for 95% VaR)
    @param horizon The time horizon for VaR calculation
    @return The calculated VaR
*)
val calculate_var : Tensor.t -> float -> int -> float

(** Calculate Expected Shortfall (ES)
    @param data The input data tensor
    @param confidence_level The confidence level (e.g., 0.95 for 95% ES)
    @param horizon The time horizon for ES calculation
    @return The calculated ES
*)
val calculate_es : Tensor.t -> float -> int -> float

(** Optimize RFSV model parameters using gradient descent
    @param data The input data tensor
    @return The optimized parameters
*)
val optimize_params : Tensor.t -> params

(** Perform batch forecasting
    @param forecast_fn The forecasting function to use
    @param data The input data tensor
    @param window_size The size of the sliding window
    @param horizon The forecast horizon
    @param batch_size The batch size for processing
    @return A tensor of forecasted values
*)
val batch_forecast : (Tensor.t -> int -> float) -> Tensor.t -> int -> int -> int -> Tensor.t
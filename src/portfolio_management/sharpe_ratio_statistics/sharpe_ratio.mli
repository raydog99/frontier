(** The type representing a Sharpe ratio calculation context *)
type t = private {
  returns: float array;
  risk_free_rate: float;
  max_lag: int;
}

(** Raised when there is insufficient data for calculations *)
exception InsufficientData of string

(** Raised when an invalid parameter is provided *)
exception InvalidParameter of string

(** Create a new Sharpe ratio calculation context 
    @param returns Array of return data
    @param risk_free_rate The risk-free rate
    @param max_lag Maximum lag for autocorrelation calculations
    @return A new Sharpe ratio calculation context
    @raise InsufficientData if returns array has less than 2 elements
    @raise InvalidParameter if max_lag is less than 1
*)
val create : float array -> float -> int -> t

(** Calculate the IID Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @return The IID Sharpe ratio
*)
val calculate_iid_sharpe_ratio : t -> float

(** Calculate the standard error of the IID Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @return The standard error of the IID Sharpe ratio
*)
val calculate_iid_sharpe_ratio_standard_error : t -> float

(** Calculate the GMM Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @return The GMM Sharpe ratio
*)
val calculate_gmm_sharpe_ratio : t -> float

(** Calculate the standard error of the GMM Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @return The standard error of the GMM Sharpe ratio
*)
val calculate_gmm_sharpe_ratio_standard_error : t -> float

(** Calculate the time-aggregated Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @param q The time aggregation factor
    @return The time-aggregated Sharpe ratio
    @raise InvalidParameter if q is less than 1
*)
val time_aggregate_sharpe_ratio : t -> int -> float

(** Calculate the standard error of the time-aggregated Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @param q The time aggregation factor
    @return The standard error of the time-aggregated Sharpe ratio
    @raise InvalidParameter if q is less than 1
*)
val time_aggregate_sharpe_ratio_standard_error : t -> int -> float

(** Perform the Ljung-Box test for autocorrelation 
    @param t The Sharpe ratio calculation context
    @return The Ljung-Box test statistic
*)
val ljung_box_test : t -> float

(** Perform the Jarque-Bera test for normality 
    @param t The Sharpe ratio calculation context
    @return The Jarque-Bera test statistic
*)
val jarque_bera_test : t -> float

(** Calculate the confidence interval for the Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @param confidence_level The desired confidence level (e.g., 0.95 for 95% CI)
    @return A tuple containing the lower and upper bounds of the confidence interval
    @raise InvalidParameter if confidence_level is not between 0 and 1
*)
val confidence_interval : t -> float -> float * float

(** Compare two Sharpe ratios and compute the p-value of their difference 
    @param t1 The first Sharpe ratio calculation context
    @param t2 The second Sharpe ratio calculation context
    @return The p-value of the difference between the two Sharpe ratios
*)
val compare_sharpe_ratios : t -> t -> float

(** Calculate rolling Sharpe ratios 
    @param t The Sharpe ratio calculation context
    @param window_size The size of the rolling window
    @return An array of rolling Sharpe ratios
    @raise InvalidParameter if window_size is less than 2 or greater than the number of returns
*)
val rolling_sharpe_ratio : t -> int -> float array

(** Perform bootstrap analysis of the Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @param num_samples The number of bootstrap samples to generate
    @return An array of bootstrapped Sharpe ratios
    @raise InvalidParameter if num_samples is less than 1
*)
val bootstrap_sharpe_ratio : t -> int -> float array

(** Adjust returns for autocorrelation 
    @param t The Sharpe ratio calculation context
    @return A new Sharpe ratio calculation context with adjusted returns
*)
val adjust_for_autocorrelation : t -> t

(** Detect outliers in the returns data 
    @param t The Sharpe ratio calculation context
    @param threshold The Z-score threshold for outlier detection
    @return An array of indices of detected outliers
*)
val detect_outliers : t -> float -> int array

(** Winsorize the returns data 
    @param t The Sharpe ratio calculation context
    @param percentile The percentile at which to winsorize (e.g., 0.05 for 5% winsorization)
    @return A new Sharpe ratio calculation context with winsorized returns
    @raise InvalidParameter if percentile is not between 0 and 0.5
*)
val winsorize : t -> float -> t

(** Generate a summary report of Sharpe ratio analysis 
    @param t The Sharpe ratio calculation context
    @return A string containing a summary report of the Sharpe ratio analysis
*)
val generate_summary_report : t -> string

(** Perform parallel bootstrap analysis of the Sharpe ratio 
    @param t The Sharpe ratio calculation context
    @param num_samples The total number of bootstrap samples to generate
    @param num_threads The number of threads to use for parallel processing
    @return An array of bootstrapped Sharpe ratios
    @raise InvalidParameter if num_samples or num_threads is less than 1
*)
val parallel_bootstrap_sharpe_ratio : t -> int -> int -> float array

(** Calculate the maximum drawdown of the returns
    @param t The Sharpe ratio calculation context
    @return The maximum drawdown as a percentage
*)
val maximum_drawdown : t -> float

(** Calculate the Sortino ratio
    @param t The Sharpe ratio calculation context
    @return The Sortino ratio
*)
val sortino_ratio : t -> float

(** Calculate the Calmar ratio
    @param t The Sharpe ratio calculation context
    @return The Calmar ratio
*)
val calmar_ratio : t -> float

(** Calculate the Omega ratio
    @param t The Sharpe ratio calculation context
    @param threshold The threshold return (default is risk-free rate)
    @return The Omega ratio
*)
val omega_ratio : t -> ?threshold:float -> unit -> float
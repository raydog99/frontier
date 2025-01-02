type t = {
  n_assets: int;
  n_timepoints: int;
  target_return: float;
  covariance_method: [ `Tracy_Widom | `Linear_Shrinkage | `Naive ];
  optimization_method: [ `Markowitz | `NCO ];
}

val default : t
val create :
  ?n_assets:int ->
  ?n_timepoints:int ->
  ?target_return:float ->
  ?covariance_method:[ `Tracy_Widom | `Linear_Shrinkage | `Naive ] ->
  ?optimization_method:[ `Markowitz | `NCO ] ->
  unit -> t
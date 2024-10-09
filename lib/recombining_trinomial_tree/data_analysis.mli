val moving_average : float array -> int -> float array
val exponential_moving_average : float array -> float -> float array
val bollinger_bands : float array -> int -> float -> float array * float array * float array
val relative_strength_index : float array -> int -> float array
val macd : float array -> int -> int -> int -> float array * float array
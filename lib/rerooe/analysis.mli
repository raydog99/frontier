val calculate_var : (float * float * float) list list -> float -> float
val calculate_expected_shortfall : (float * float * float) list list -> float -> float
val calculate_max_drawdown : (float * float * float) list list -> float
val calculate_sharpe_ratio : (float * float * float) list list -> float
val generate_report : (float * float * float) list list -> string -> unit
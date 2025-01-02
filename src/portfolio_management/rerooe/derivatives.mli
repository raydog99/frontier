type option_type = Call | Put

type t =
  | EuropeanOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
    }
  | AmericanOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
    }
  | AsianOption of {
      underlying: string;
      option_type: option_type;
      strike: float;
      expiry: float;
      averaging_period: float;
    }

val price : t -> float -> float -> float -> float
val delta : t -> float -> float -> float -> float
val gamma : t -> float -> float -> float -> float
open Torch

type country = US | Japan | UK | Canada | France | Germany
[@@deriving show, eq]

type industry = Technology | Healthcare | Finance | ConsumerGoods | Energy
[@@deriving show, eq]

type stock_data = {
  date: int;
  country: country;
  industry: industry;
  size: float;
  bm: float;
  op: float;
  inv: float;
  returns: float;
  exchange: string;
}
[@@deriving show, eq]

type sort_method = TwoByThree | TwoByTwo | TwoByTwoByTwoByTwo
[@@deriving show, eq]

val construct_factors : stock_data list -> sort_method -> float * float * float * float
val factors_to_tensor : float list -> Tensor.t
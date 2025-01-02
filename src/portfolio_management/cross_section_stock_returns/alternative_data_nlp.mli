open Torch

val analyze_sentiment : string -> float
val process_news_headlines : string list -> float list
val extract_sec_filing_features : string -> float list
val process_satellite_imagery : string -> Tensor.t
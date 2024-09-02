open Torch
open Types

val split_data : DataType.t -> float -> DataSplit.t
val mse_loss : LossType.t
val mae_loss : LossType.t
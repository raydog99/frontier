open Base

  type t = {
    mutable train_losses: float list;
    mutable val_losses: float list;
    mutable best_val_loss: float;
  }

  let create () =
    { train_losses = []; val_losses = []; best_val_loss = Float.infinity }

  let log_train_loss t loss =
    t.train_losses <- loss :: t.train_losses

  let log_val_loss t loss
type t =
  | Geometric_Brownian_Motion of float * float
  | Ornstein_Uhlenbeck of float * float * float
  | Heston of float * float * float * float * float
  | Merton_Jump_Diffusion of float * float * float * float

let simulate process dt num_steps initial_value =
  match process with
  | Geometric_Brownian_Motion (mu, sigma) ->
      let path = Array.make (num_steps + 1) initial_value in
      for i = 1 to num_steps do
        let dw = Random.float 1. in
        path.(i) <- path.(i-1) *. exp((mu -. 0.5 *. sigma ** 2.) *. dt +. sigma *. sqrt dt *. dw)
      done;
      path
  | Ornstein_Uhlenbeck (alpha, mu, sigma) ->
      let path = Array.make (num_steps + 1) initial_value in
      for i = 1 to num_steps do
        let dw = Random.float 1. in
        path.(i) <- path.(i-1) +. alpha *. (mu -. path.(i-1)) *. dt +. sigma *. sqrt dt *. dw
      done;
      path
  | Heston (kappa, theta, sigma, rho, v0) ->
      let s_path = Array.make (num_steps + 1) initial_value in
      let v_path = Array.make (num_steps + 1) v0 in
      for i = 1 to num_steps do
        let dw1 = Random.float 1. in
        let dw2 = rho *. dw1 +. sqrt(1. -. rho ** 2.) *. Random.float 1. in
        v_path.(i) <- max 0. (v_path.(i-1) +. kappa *. (theta -. v_path.(i-1)) *. dt +. sigma *. sqrt(v_path.(i-1) *. dt) *. dw2);
        s_path.(i) <- s_path.(i-1) *. exp((kappa *. theta -. 0.5 *. v_path.(i-1)) *. dt +. sqrt(v_path.(i-1) *. dt) *. dw1)
      done;
      s_path
  | Merton_Jump_Diffusion (mu, sigma, lambda, jump_size) ->
      let path = Array.make (num_steps + 1) initial_value in
      for i = 1 to num_steps do
        let dw = Random.float 1. in
        let jump = if Random.float 1. < lambda *. dt then jump_size else 0. in
        path.(i) <- path.(i-1) *. exp((mu -. 0.5 *. sigma ** 2.) *. dt +. sigma *. sqrt dt *. dw +. jump)
      done;
      path
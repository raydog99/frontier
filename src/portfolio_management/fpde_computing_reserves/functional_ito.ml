open Types

type functional = {
  value: non_anticipative_functional;
  horizontal_derivative: non_anticipative_functional;
  vertical_derivative: non_anticipative_functional;
  second_vertical_derivative: non_anticipative_functional;
}

let create_functional
    (value: non_anticipative_functional)
    (horizontal_derivative: non_anticipative_functional)
    (vertical_derivative: non_anticipative_functional)
    (second_vertical_derivative: non_anticipative_functional) : functional =
  { value; horizontal_derivative; vertical_derivative; second_vertical_derivative }

let horizontal_derivative (f: functional) : non_anticipative_functional =
  f.horizontal_derivative

let vertical_derivative (f: functional) : non_anticipative_functional =
  f.vertical_derivative

let second_vertical_derivative (f: functional) : non_anticipative_functional =
  f.second_vertical_derivative

let functional_ito_formula
    (f: functional)
    (t: time)
    (omega: path)
    (drift: non_anticipative_functional)
    (diffusion: non_anticipative_functional) : float =
  let dt = 1e-6 in  (* Small time step for approximation *)
  let dw = Random.float (sqrt dt) in  (* Approximate Brownian motion increment *)
  
  f.value t omega +.
  f.horizontal_derivative t omega *. dt +.
  f.vertical_derivative t omega *. (drift t omega *. dt +. diffusion t omega *. dw) +.
  0.5 *. f.second_vertical_derivative t omega *. (diffusion t omega ** 2.0) *. dt
open Types
open Lwt
open Domainslib

let apply_boundary_condition (bc: boundary_condition) (t: time) (x: float) (v: float) (dv: float) =
  match bc with
  | Dirichlet f -> f t x
  | Neumann f -> v +. dv *. f t x
  | Mixed f -> 
      let (a, b, c) = f t x in
      (c -. a *. v) /. b

let crank_nicolson_step
    (v : float array array)
    (dt : float)
    (dx : float array)
    (r : float)
    (sigma : float array array)
    (source : float array array)
    (lower_bc : boundary_condition array)
    (upper_bc : boundary_condition array) : float array array =
  let n = Array.length v in
  let m = Array.length v.(0) in
  
  let result = Array.make_matrix n m 0.0 in
  
  for i = 0 to n - 1 do
    let a = Array.make (m-2) 0.0 in
    let b = Array.make (m-2) 0.0 in
    let c = Array.make (m-2) 0.0 in
    let d = Array.make (m-2) 0.0 in
    
    for j = 1 to m - 2 do
      let alpha = 0.25 *. dt *. (sigma.(i).(i) *. sigma.(i).(i) /. (dx.(i) *. dx.(i)) +. r /. dx.(i)) in
      let beta = -0.5 *. dt *. (sigma.(i).(i) *. sigma.(i).(i) /. (dx.(i) *. dx.(i)) +. r) in
      let gamma = 0.25 *. dt *. (sigma.(i).(i) *. sigma.(i).(i) /. (dx.(i) *. dx.(i)) -. r /. dx.(i)) in
      
      a.(j-1) <- -alpha;
      b.(j-1) <- 1.0 -. beta;
      c.(j-1) <- -gamma;
      d.(j-1) <- alpha *. v.(i).(j-1) +. (1.0 +. beta) *. v.(i).(j) +. gamma *. v.(i).(j+1) +. 0.5 *. dt *. (source.(i).(j) +. source.(i).(j-1))
    done;
    
    let t = dt *. float_of_int i in
    result.(i).(0) <- apply_boundary_condition lower_bc.(i) t (float_of_int i *. dx.(i)) v.(i).(0) ((v.(i).(1) -. v.(i).(0)) /. dx.(i));
    result.(i).(m-1) <- apply_boundary_condition upper_bc.(i) t (float_of_int (m-1) *. dx.(i)) v.(i).(m-1) ((v.(i).(m-1) -. v.(i).(m-2)) /. dx.(i));
    
    for j = 1 to m - 3 do
      let w = a.(j) /. b.(j-1) in
      b.(j) <- b.(j) -. w *. c.(j-1);
      d.(j) <- d.(j) -. w *. d.(j-1);
    done;
    
    result.(i).(m-2) <- d.(m-3) /. b.(m-3);
    for j = m - 4 downto 0 do
      result.(i).(j+1) <- (d.(j) -. c.(j) *. result.(i).(j+2)) /. b.(j);
    done;
  done;
  
  result

let solve_pde_adi 
    (coeffs : Path_dependent_pde.pde_coefficients)
    (terminal_condition : non_anticipative_functional)
    (t0 : time)
    (t1 : time)
    (omega : path)
    (num_time_steps : int)
    (num_space_steps : int array)
    (lower_bc : boundary_condition array)
    (upper_bc : boundary_condition array) : Functional_ito.functional =
  let dt = (t1 -. t0) /. float_of_int num_time_steps in
  let dx = Array.map (fun steps -> (Array.fold_left max neg_infinity omega -. Array.fold_left min infinity omega) /. float_of_int steps) num_space_steps in
  
  let grid = Array.init (Array.length omega) (fun _ -> Array.make_matrix (num_time_steps + 1) (Array.get num_space_steps 0 + 1) 0.0) in
  
  for i = 0 to Array.length omega - 1 do
    for j = 0 to num_space_steps.(i) do
      let x = Array.get omega i +. float_of_int j *. dx.(i) in
      let omega_t = Array.copy omega in
      omega_t.(i) <- x;
      grid.(i).(num_time_steps).(j) <- terminal_condition t1 omega_t
    done
  done;
  
  for t = num_time_steps - 1 downto 0 do
    let current_time = t0 +. float_of_int t *. dt in
    let v = Array.map (fun g -> g.(t+1)) grid in
    let source = Array.init (Array.length omega) (fun i ->
      Array.init (num_space_steps.(i) + 1) (fun j ->
        let x = Array.get omega i +. float_of_int j *. dx.(i) in
        let omega_t = Array.copy omega in
        omega_t.(i) <- x;
        coeffs.source current_time omega_t
      )
    ) in
    let r = coeffs.rate current_time in
    let (_, sigma) = coeffs.diffusion current_time omega in
    let new_v = crank_nicolson_step v dt dx r sigma source lower_bc upper_bc in
    for i = 0 to Array.length omega - 1 do
      grid.(i).(t) <- new_v.(i)
    done
  done;
  
  let value t omega =
    let i = int_of_float ((t -. t0) /. dt) in
    let indices = Array.mapi (fun j x -> int_of_float ((x -. omega.(j)) /. dx.(j))) omega in
    Array.fold_left (fun acc g -> acc +. g.(i).(indices.(Array.length indices - 1))) 0.0 grid
  in
  let horizontal_derivative t omega =
    let i = int_of_float ((t -. t0) /. dt) in
    let indices = Array.mapi (fun j x -> int_of_float ((x -. omega.(j)) /. dx.(j))) omega in
    if i < num_time_steps then
      Array.fold_left (fun acc g -> 
        acc +. (g.(i+1).(indices.(Array.length indices - 1)) -. g.(i).(indices.(Array.length indices - 1))) /. dt
      ) 0.0 grid
    else
      0.0
  in
  let vertical_derivative t omega =
    let i = int_of_float ((t -. t0) /. dt) in
    let indices = Array.mapi (fun j x -> int_of_float ((x -. omega.(j)) /. dx.(j))) omega in
    Array.mapi (fun j idx ->
      if idx > 0 && idx < num_space_steps.(j) then
        (grid.(j).(i).(idx+1) -. grid.(j).(i).(idx-1)) /. (2.0 *. dx.(j))
      else
        0.0
    ) indices
  in
  let second_vertical_derivative t omega =
    let i = int_of_float ((t -. t0) /. dt) in
    let indices = Array.mapi (fun j x -> int_of_float ((x -. omega.(j)) /. dx.(j))) omega in
    Array.mapi (fun j idx ->
      if idx > 0 && idx < num_space_steps.(j) then
        (grid.(j).(i).(idx+1) -. 2.0 *. grid.(j).(i).(idx) +. grid.(j).(i).(idx-1)) /. (dx.(j) *. dx.(j))
      else
        0.0
    ) indices
  in
  Functional_ito.create_functional value horizontal_derivative vertical_derivative second_vertical_derivative

let parallel_solve_pde 
    (coeffs : Path_dependent_pde.pde_coefficients)
    (terminal_condition : non_anticipative_functional)
    (t0 : time)
    (t1 : time)
    (omega : path)
    (num_time_steps : int)
    (num_space_steps : int array)
    (lower_bc : boundary_condition array)
    (upper_bc : boundary_condition array)
    (num_threads : int) : Functional_ito.functional =
  let pool = Task.setup_pool ~num_domains:num_threads () in
  
  let dt = (t1 -. t0) /. float_of_int num_time_steps in
  let dx = Array.map (fun steps -> (Array.fold_left max neg_infinity omega -. Array.fold_left min infinity omega) /. float_of_int steps) num_space_steps in
  
  let grid = Array.init (Array.length omega) (fun _ -> Array.make_matrix (num_time_steps + 1) (Array.get num_space_steps 0 + 1) 0.0) in
  
  Task.parallel_for pool ~start:0 ~finish:(Array.length omega - 1) ~body:(fun i ->
    for j = 0 to num_space_steps.(i) do
      let x = Array.get omega i +. float_of_int j *. dx.(i) in
      let omega_t = Array.copy omega in
      omega_t.(i) <- x;
      grid.(i).(num_time_steps).(j) <- terminal_condition t1 omega_t
    done
  );
  
  for t = num_time_steps - 1 downto 0 do
    let current_time = t0 +. float_of_int t *. dt in
    let v = Array.map (fun g -> g.(t+1)) grid in
    let source = Array.init (Array.length omega) (fun i ->
      Array.init (num_space_steps.(i) + 1) (fun j ->
        let x = Array.get omega i +. float_of_int j *. dx.(i) in
        let omega_t = Array.copy omega in
        omega_t.(i) <- x;
        coeffs.source current_time omega_t
      )
    ) in
    let r = coeffs.rate current_time in
    let (_, sigma) = coeffs.diffusion current_time omega in
    let new_v = crank_nicolson_step v dt dx r sigma source lower_bc upper_bc in
    Task.parallel_for pool ~start:0 ~finish:(Array.length omega - 1) ~body:(fun i ->
      grid.(i).(t) <- new_v.(i)
    );
  done;
  
  Task.teardown_pool pool;
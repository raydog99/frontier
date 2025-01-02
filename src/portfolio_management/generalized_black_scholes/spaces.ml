open Utils

module FunctionSpace = struct
  type t = {
    lower: float;
    upper: float;
    grid: Grid.t;
    measure: Measure.t;
  }

  let make lower upper grid measure =
    { lower; upper; grid; measure }

  let v_m_norm space f =
    let dx = Grid.delta space.grid in
    let l2_norm = ref 0.0 in
    let derivative_norm = ref 0.0 in
    
    for i = 0 to Array.length f - 2 do
      let x = (Grid.points space.grid).(i) in
      l2_norm := !l2_norm +. f.(i) *. f.(i) *. dx;
      let derivative = (f.(i+1) -. f.(i)) /. dx in
      derivative_norm := !derivative_norm +. 
        x *. x *. derivative *. derivative *. dx
    done;
    sqrt(!l2_norm +. !derivative_norm)

  let project_v0m space f =
    let n = Array.length f in
    let result = Array.copy f in
    result.(0) <- 0.0;
    result.(n-1) <- 0.0;
    result
end

module BilinearForm = struct
  type t = {
    sigma: float;
    r: float;
    space: FunctionSpace.t;
  }

  let make sigma r space = { sigma; r; space }

  let evaluate form u v grid =
    let dx = Grid.delta grid in
    let points = Grid.points grid in
    let n = Array.length points in
    
    let du_dx = Array.make (n-1) 0.0 in
    let dv_dx = Array.make (n-1) 0.0 in
    
    for i = 0 to n-2 do
      du_dx.(i) <- (u.(i+1) -. u.(i)) /. dx;
      dv_dx.(i) <- (v.(i+1) -. v.(i)) /. dx
    done;
    
    let term1 = ref 0.0 in
    let term2 = ref 0.0 in
    let term3 = ref 0.0 in
    
    for i = 0 to n-2 do
      let x = points.(i) in
      term1 := !term1 +. form.sigma *. form.sigma *. x *. x /. 2.0 
               *. du_dx.(i) *. dv_dx.(i) *. dx;
      term2 := !term2 +. (form.sigma *. form.sigma -. form.r) *. x 
               *. du_dx.(i) *. v.(i) *. dx;
      term3 := !term3 +. form.r *. u.(i) *. v.(i) *. dx
    done;
    
    !term1 +. !term2 +. !term3

  let symmetric_part form u v grid =
    let b_uv = evaluate form u v grid in
    let b_vu = evaluate form v u grid in
    0.5 *. (b_uv +. b_vu)

  let nonsymmetric_part form u v grid =
    let dx = Grid.delta grid in
    let points = Grid.points grid in
    let result = ref 0.0 in
    
    for i = 0 to Array.length u - 2 do
      let x = points.(i) in
      let du = (u.(i+1) -. u.(i)) /. dx in
      result := !result +. (form.sigma *. form.sigma -. form.r) *. 
                x *. du *. v.(i) *. dx
    done;
    !result
end
open Torch

(* Core coordinate system *)
module Coordinates = struct
  type point = {
    x: float;   (* Independent variable *)
    u: float;   (* Dependent variable *)
    u1: float;  (* First derivative *)
  }

  let create x u u1 = {x; u; u1}
  let to_array {x; u; u1} = [|x; u; u1|]
  let of_array arr = {x = arr.(0); u = arr.(1); u1 = arr.(2)}
end

let inner_product v1 v2 g =
  let v1_arr = Tensor.of_float1 v1 in
  let v2_arr = Tensor.of_float1 v2 in
  Tensor.mm (Tensor.mm v1_arr g) (Tensor.transpose v2_arr ~dim0:0 ~dim1:1)

let normalize v =
  let norm = Tensor.norm v ~p:2.0 ~dim:[0] ~keepdim:true in
  Tensor.div v norm

let cross v1 v2 =
  let x1, y1, z1 = Tensor.get v1 0 0, Tensor.get v1 0 1, Tensor.get v1 0 2 in
  let x2, y2, z2 = Tensor.get v2 0 0, Tensor.get v2 0 1, Tensor.get v2 0 2 in
  Tensor.of_float2 [|[|y1 *. z2 -. z1 *. y2;
                      z1 *. x2 -. x1 *. z2;
                      x1 *. y2 -. y1 *. x2|]|]

let partial_derivative f point dim h =
  let pt_plus = Array.copy (Coordinates.to_array point) in
  let pt_minus = Array.copy (Coordinates.to_array point) in
  pt_plus.(dim) <- pt_plus.(dim) +. h;
  pt_minus.(dim) <- pt_minus.(dim) -. h;
  (f (Coordinates.of_array pt_plus) -. f (Coordinates.of_array pt_minus)) /. (2.0 *. h)

let det_3x3 m =
  let get i j = Tensor.get m i j in
  get 0 0 *. (get 1 1 *. get 2 2 -. get 1 2 *. get 2 1) +.
  get 0 1 *. (get 1 2 *. get 2 0 -. get 1 0 *. get 2 2) +.
  get 0 2 *. (get 1 0 *. get 2 1 -. get 1 1 *. get 2 0)

let inverse_3x3 m =
  let det = det_3x3 m in
  if abs_float det < 1e-10 then failwith "Matrix is singular";
  let adj = Tensor.zeros [|3; 3|] in
  let get i j = Tensor.get m i j in
  Tensor.set adj [|0; 0|] (get 1 1 *. get 2 2 -. get 1 2 *. get 2 1);
  Tensor.set adj [|0; 1|] (get 0 2 *. get 2 1 -. get 0 1 *. get 2 2);
  Tensor.set adj [|0; 2|] (get 0 1 *. get 1 2 -. get 0 2 *. get 1 1);
  Tensor.set adj [|1; 0|] (get 1 2 *. get 2 0 -. get 1 0 *. get 2 2);
  Tensor.set adj [|1; 1|] (get 0 0 *. get 2 2 -. get 0 2 *. get 2 0);
  Tensor.set adj [|1; 2|] (get 0 2 *. get 1 0 -. get 0 0 *. get 1 2);
  Tensor.set adj [|2; 0|] (get 1 0 *. get 2 1 -. get 1 1 *. get 2 0);
  Tensor.set adj [|2; 1|] (get 0 1 *. get 2 0 -. get 0 0 *. get 2 1);
  Tensor.set adj [|2; 2|] (get 0 0 *. get 1 1 -. get 0 1 *. get 1 0);
  Tensor.div_scalar adj det

(* Jet bundle structure *)
module JetBundle = struct
  type t = {
    domain: (float * float) * (float * float) * (float * float);
    contact_form: Coordinates.point -> Tensor.t;
  }

  let create domain = {
    domain;
    contact_form = fun point ->
      Tensor.of_float1 [|-point.Coordinates.u1; 1.0; 0.0|]
  }

  let prolongation f x =
    let h = 1e-6 in
    let u = f x in
    let u1 = (f (x +. h) -. f (x -. h)) /. (2.0 *. h) in
    Coordinates.create x u u1
end

(* Autonomous second-order ODE *)
module AutoODE = struct
  type t = {
    phi: float * float -> float;  (* u₂ = φ(u,u₁) *)
    domain: JetBundle.t;
  }

  let metric ode point =
    let {Coordinates.x=_; u; u1} = point in
    let phi_val = ode.phi (u, u1) in
    Tensor.of_float2 [|
      [|1.0 +. u1 *. u1;     -.2.0 *. u1;        -.2.0 *. phi_val /. u1|];
      [|-.2.0 *. u1;         1.0 +. phi_val *. phi_val /. (u1 *. u1);  0.0|];
      [|-.2.0 *. phi_val /. u1;  0.0;             1.0|]
    |]

  let vector_field ode point =
    let {Coordinates.x=_; u; u1} = point in
    Tensor.of_float1 [|1.0; u1; ode.phi (u, u1)|]
end

(* Connection theory *)
module Connection = struct
  type t = {
    christoffel: Coordinates.point -> Tensor.t;
    parallel_transport: Tensor.t -> Coordinates.point -> Coordinates.point -> Tensor.t;
  }

  let create_levi_civita metric =
    let christoffel point =
      let dim = 3 in
      let g = metric point in
      let g_inv = inverse_3x3 g in
      let h = 1e-6 in
      let result = Tensor.zeros [|dim; dim; dim|] in
      
      for i = 0 to dim-1 do
        for j = 0 to dim-1 do
          for k = 0 to dim-1 do
            let sum = ref 0.0 in
            for l = 0 to dim-1 do
              let g_il = Tensor.get g_inv [|i;l|] in
              let d_gjk = partial_derivative metric j k point h in
              let d_gkl = partial_derivative metric k l point h in
              let d_gjl = partial_derivative metric j l point h in
              sum := !sum +. 0.5 *. g_il *. (d_gjk +. d_gkl -. d_gjl)
            done;
            Tensor.set result [|i;j;k|] !sum
          done
        done
      done;
      result
    in

    let parallel_transport v start_pt end_pt =
      let steps = 100 in
      let h = 1.0 /. float_of_int steps in
      let current_v = ref v in
      let current_pt = ref start_pt in
      
      for _ = 1 to steps do
        let gamma = christoffel !current_pt in
        let tangent = normalize (
          Tensor.sub (Tensor.of_float1 (Coordinates.to_array end_pt))
                    (Tensor.of_float1 (Coordinates.to_array !current_pt))) in
        
        let new_v = Tensor.zeros [|3|] in
        for i = 0 to 2 do
          let sum = ref (Tensor.get !current_v [|i|]) in
          for j = 0 to 2 do
            for k = 0 to 2 do
              sum := !sum -. h *. Tensor.get gamma [|i;j;k|] *.
                           Tensor.get !current_v [|j|] *.
                           Tensor.get tangent [|k|]
            done
          done;
          Tensor.set new_v [|i|] !sum
        done;
        
        current_v := new_v;
        current_pt := Coordinates.of_array (
          Array.map2 (+.) (Coordinates.to_array !current_pt)
                         (Array.map (( *. ) h) (Array.of_list (List.init 3 (fun i -> 
                           Tensor.get tangent [|i|])))))
      done;
      !current_v
    in

    {christoffel; parallel_transport}
end

(* Energy foliation *)
module EnergyFoliation = struct
  type t = {
    ode: AutoODE.t;
    energy_fn: Coordinates.point -> float;
  }

  let distribution foliation point =
    let e1 = AutoODE.vector_field foliation.ode point in
    let e2 = Tensor.of_float1 [|0.0; 1.0; 
      foliation.ode.phi (point.Coordinates.u, point.u1) /. point.u1|] in
    (e1, e2)
end

(* Numerical solvers *)
module Numerical = struct
  type solution = {
    times: float array;
    points: Coordinates.point array;
  }

  let rk4_step ode h point =
    let f pt = (pt.Coordinates.u1, ode.AutoODE.phi (pt.u, pt.u1)) in
    
    let k1 = f point in
    let pt2 = {point with 
      Coordinates.u = point.u +. h *. fst k1 /. 2.0;
      u1 = point.u1 +. h *. snd k1 /. 2.0} in
    let k2 = f pt2 in
    let pt3 = {point with
      Coordinates.u = point.u +. h *. fst k2 /. 2.0;
      u1 = point.u1 +. h *. snd k2 /. 2.0} in
    let k3 = f pt3 in
    let pt4 = {point with
      Coordinates.u = point.u +. h *. fst k3;
      u1 = point.u1 +. h *. snd k3} in
    let k4 = f pt4 in
    
    {point with
      Coordinates.x = point.x +. h;
      u = point.u +. h *. (fst k1 +. 2.0 *. fst k2 +. 2.0 *. fst k3 +. fst k4) /. 6.0;
      u1 = point.u1 +. h *. (snd k1 +. 2.0 *. snd k2 +. 2.0 *. snd k3 +. snd k4) /. 6.0}

  let solve ode t0 t1 initial_point n_steps =
    let h = (t1 -. t0) /. float_of_int n_steps in
    let times = Array.init (n_steps + 1) (fun i -> t0 +. float_of_int i *. h) in
    let points = Array.make (n_steps + 1) initial_point in
    
    for i = 0 to n_steps - 1 do
      points.(i + 1) <- rk4_step ode h points.(i)
    done;
    
    {times; points}
end

(* Lagrangian mechanics *)
module Lagrangian = struct
  type t = {
    l: float * float -> float;        (* L(u,u₁) *)
    energy: float * float -> float;   (* Energy function h *)
  }

  let from_energy energy =
    let l (u, u1) =
      let h = 1e-6 in
      u1 *. (energy (u, u1) /. (u1 *. u1))
    in
    {l; energy}

  let damped_oscillator alpha lambda =
    assert (alpha *. alpha < 4.0 *. lambda);
    let omega = sqrt (4.0 *. lambda -. alpha *. alpha) /. 2.0 in
    
    let energy (u, u1) =
      let arg = (alpha *. u1 +. 2.0 *. lambda *. u) /. (2.0 *. omega *. u1) in
      exp (alpha *. atan arg /. omega) *.
        (alpha *. u *. u1 +. u1 *. u1 +. lambda *. u *. u) /. 2.0
    in
    
    from_energy energy

  let gravitational_field mass radius =
    let g = 6.67430e-11 in
    
    let phi r =
      if r <= radius then
        -.g *. mass *. (3.0 *. radius -. r) /. (2.0 *. radius ** 3.0)
      else
        -.g *. mass /. r
    in
    
    let l (u, u1) = 0.5 *. u1 *. u1 -. phi u in
    let energy (u, u1) = 0.5 *. u1 *. u1 +. phi u in
    {l; energy}
end


(* Tensor operations and differential forms *)
module TensorAlgebra = struct
  type differential_form = {
    degree: int;
    components: Tensor.t;
    indices: int list;
  }

  (* Create standard volume form *)
  let volume_form point =
    let dim = 3 in
    {
      degree = dim;
      components = Tensor.ones [|dim; dim; dim|];
      indices = [0; 1; 2]
    }

  (* Interior product with vector field *)
  let interior_product v form =
    match form.degree with
    | 1 -> 
        let result = ref 0.0 in
        for i = 0 to 2 do
          result := !result +. (Tensor.get v [|i|]) *. (Tensor.get form.components [|i|])
        done;
        {
          degree = 0;
          components = Tensor.of_float0 !result;
          indices = []
        }
    | 2 ->
        let result = Tensor.zeros [|3|] in
        for i = 0 to 2 do
          let sum = ref 0.0 in
          for j = 0 to 2 do
            sum := !sum +. (Tensor.get v [|j|]) *. (Tensor.get form.components [|i;j|])
          done;
          Tensor.set result [|i|] !sum
        done;
        {
          degree = 1;
          components = result;
          indices = [0]
        }

  (* Exterior derivative *)
  let d form point =
    let h = 1e-6 in
    match form.degree with
    | 0 ->
        let result = Tensor.zeros [|3|] in
        for i = 0 to 2 do
          let pt_plus = Coordinates.of_array (
            Array.mapi (fun j x -> if j = i then x +. h else x)
              (Coordinates.to_array point))
          in
          let pt_minus = Coordinates.of_array (
            Array.mapi (fun j x -> if j = i then x -. h else x)
              (Coordinates.to_array point))
          in
          let deriv = (Tensor.get (form.components) [||] -.
                      Tensor.get (form.components) [||]) /. (2.0 *. h) in
          Tensor.set result [|i|] deriv
        done;
        {
          degree = 1;
          components = result;
          indices = [0; 1; 2]
        }
    | 1 ->
        let result = Tensor.zeros [|3;3|] in
        for i = 0 to 2 do
          for j = 0 to 2 do
            let dji = partial_derivative form.components j i point h in
            let dij = partial_derivative form.components i j point h in
            Tensor.set result [|i;j|] (dji -. dij)
          done
        done;
        {
          degree = 2;
          components = result;
          indices = [0; 1; 2; 3; 4; 5]
        }

  (* Wedge product *)
  let wedge form1 form2 =
    match (form1.degree, form2.degree) with
    | (1, 1) ->
        let result = Tensor.zeros [|3;3|] in
        for i = 0 to 2 do
          for j = 0 to 2 do
            let val1 = Tensor.get form1.components [|i|] in
            let val2 = Tensor.get form2.components [|j|] in
            Tensor.set result [|i;j|] (val1 *. val2 -. val2 *. val1)
          done
        done;
        {
          degree = 2;
          components = result;
          indices = form1.indices @ form2.indices
        }
end

(* Jet bundle *)
module JetBundle = struct
  type t = {
    domain: (float * float) * (float * float) * (float * float);
    contact_form: Coordinates.point -> TensorAlgebra.differential_form;
  }

  (* Standard coordinates *)
  let standard_chart = {
    domain = ((-.max_float, max_float), 
             (-.max_float, max_float),
             (-.max_float, max_float));
    contact_form = fun point ->
      {
        TensorAlgebra.degree = 1;
        components = Tensor.of_float1 [|-point.Coordinates.u1; 1.0; 0.0|];
        indices = [0; 1; 2]
      }
  }

  (* Create jet bundle with given domain *)
  let create domain = {
    domain;
    contact_form = standard_chart.contact_form
  }

  (* Check if point satisfies contact condition *)
  let satisfies_contact bundle point vector =
    let theta = bundle.contact_form point in
    let inner = TensorAlgebra.interior_product 
      (Tensor.of_float1 (Coordinates.to_array vector))
      theta in
    abs_float (Tensor.get inner.components [||]) < 1e-10

  (* Compute first prolongation of function *)
  let prolongation f x =
    let h = 1e-6 in
    let u = f x in
    let u1 = (f (x +. h) -. f (x -. h)) /. (2.0 *. h) in
    Coordinates.create x u u1
end

(* Basic metric operations *)
module MetricOperations = struct
  (* Compute Christoffel symbols from metric *)
  let christoffel metric point =
    let dim = 3 in
    let g = metric point in
    let g_inv = inverse_3x3 g in
    let h = 1e-6 in
    
    let result = Tensor.zeros [|dim; dim; dim|] in
    for i = 0 to dim-1 do
      for j = 0 to dim-1 do
        for k = 0 to dim-1 do
          let sum = ref 0.0 in
          for l = 0 to dim-1 do
            let g_il = Tensor.get g_inv [|i;l|] in
            let d_gjk = partial_derivative g j k point h in
            let d_gkl = partial_derivative g k l point h in
            let d_gjl = partial_derivative g j l point h in
            sum := !sum +. 0.5 *. g_il *. (d_gjk +. d_gkl -. d_gjl)
          done;
          Tensor.set result [|i;j;k|] !sum
        done
      done
    done;
    result

  (* Compute Riemann curvature tensor *)
  let riemann_tensor gamma point =
    let dim = 3 in
    let result = Tensor.zeros [|dim; dim; dim; dim|] in
    let h = 1e-6 in
    
    for i = 0 to dim-1 do
      for j = 0 to dim-1 do
        for k = 0 to dim-1 do
          for l = 0 to dim-1 do
            let d_gamma_k = partial_derivative gamma i j l point h in
            let d_gamma_l = partial_derivative gamma i j k point l h in
            let sum = ref (d_gamma_k -. d_gamma_l) in
            
            for m = 0 to dim-1 do
              let term1 = Tensor.get gamma [|i;m;k|] *. 
                         Tensor.get gamma [|m;j;l|] in
              let term2 = Tensor.get gamma [|i;m;l|] *. 
                         Tensor.get gamma [|m;j;k|] in
              sum := !sum +. term1 -. term2
            done;
            
            Tensor.set result [|i;j;k;l|] !sum
          done
        done
      done
    done;
    result

  (* Compute sectional curvature *)
  let sectional_curvature riemann point v w =
    let r = Tensor.get riemann [|1;2;1;2|] in
    let g_vv = inner_product v v (metric point) in
    let g_ww = inner_product w w (metric point) in
    let g_vw = inner_product v w (metric point) in
    r /. (g_vv *. g_ww -. g_vw *. g_vw)
end

(* Curvature module *)
module Curvature = struct
  type t = {
    riemann: Coordinates.point -> int -> int -> int -> int -> float;
    ricci: Coordinates.point -> int -> int -> float;
    scalar: Coordinates.point -> float;
    sectional: Coordinates.point -> int -> int -> float;
  }

  (* Create curvature operators *)
  let create connection metric =
    let riemann point i j k l =
      let r = MetricOperations.riemann_tensor 
        (connection.Connection.christoffel point) point in
      Tensor.get r [|i;j;k;l|]
    in
    
    let ricci point i j =
      let sum = ref 0.0 in
      for k = 0 to 2 do
        sum := !sum +. riemann point k i k j
      done;
      !sum
    in
    
    let scalar point =
      let sum = ref 0.0 in
      for i = 0 to 2 do
        for j = 0 to 2 do
          sum := !sum +. ricci point i j
        done
      done;
      !sum
    in
    
    let sectional point i j =
      MetricOperations.sectional_curvature 
        (MetricOperations.riemann_tensor 
           (connection.Connection.christoffel point) point)
        point
        (Tensor.of_float1 [|float_of_int i|])
        (Tensor.of_float1 [|float_of_int j|])
    in
    
    {riemann; ricci; scalar; sectional}
end
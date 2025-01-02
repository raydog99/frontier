open Utils
open Spaces

module DirichletForm = struct
  type t = {
    bilinear: BilinearForm.t;
    measure: Measure.t;
  }

  let make bilinear measure = { bilinear; measure }

  let check_sector_condition form u v grid =
    let b_uv = BilinearForm.evaluate form.bilinear u v grid in
    let b_uu = BilinearForm.evaluate form.bilinear u u grid in
    let b_vv = BilinearForm.evaluate form.bilinear v v grid in
    abs_float b_uv <= sqrt(b_uu *. b_vv)

  let check_coercivity form u grid =
    let b_uu = BilinearForm.evaluate form.bilinear u u grid in
    let norm_u = ref 0.0 in
    Array.iter (fun x -> norm_u := !norm_u +. x *. x) u;
    b_uu >= form.bilinear.sigma *. form.bilinear.sigma *. !norm_u /. 2.0
end

module Generator = struct
  type t = {
    form: DirichletForm.t;
  }

  let make form = { form }

  let apply gen u grid =
    let n = Array.length u in
    let result = Array.make n 0.0 in
    let dx = Grid.delta grid in
    let points = Grid.points grid in
    
    for i = 1 to n-2 do
      let x = points.(i) in
      let d2u = (u.(i+1) -. 2.0 *. u.(i) +. u.(i-1)) /. (dx *. dx) in
      let du = (u.(i+1) -. u.(i-1)) /. (2.0 *. dx) in
      
      result.(i) <- 
        -. gen.form.bilinear.sigma *. gen.form.bilinear.sigma *. x *. x *. d2u /. 2.0
        -. gen.form.bilinear.r *. x *. du
        +. gen.form.bilinear.r *. u.(i)
    done;
    result

  let apply_adjoint gen u grid =
    let n = Array.length u in
    let result = Array.make n 0.0 in
    let dx = Grid.delta grid in
    let points = Grid.points grid in
    
    for i = 1 to n-2 do
      let x = points.(i) in
      let d2u = (u.(i+1) -. 2.0 *. u.(i) +. u.(i-1)) /. (dx *. dx) in
      let du = (u.(i+1) -. u.(i-1)) /. (2.0 *. dx) in
      
      result.(i) <- 
        -. gen.form.bilinear.sigma *. gen.form.bilinear.sigma *. x *. x *. d2u /. 2.0
        +. gen.form.bilinear.r *. x *. du
        +. gen.form.bilinear.r *. u.(i)
    done;
    result
end
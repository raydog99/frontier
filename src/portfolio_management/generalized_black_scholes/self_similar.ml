open Utils

type word = int list

let apply_word measure word x =
  List.fold_left (fun acc w ->
    if w = 1 then Measure.f1 acc
    else Measure.f2 measure acc) x word

module PrefractalGraph = struct
  type vertex = {
    position: float;
    level: int;
    word: int list;
  }

  type t = {
    vertices: vertex array;
    edges: (int * int) array;
    measure: Measure.t;
    level: int;
  }

  let make measure level =
    let rec generate_vertices current_level acc =
      if current_level = 0 then acc
      else
        let new_vertices = Array.fold_left (fun acc v ->
          let f1_v = {
            position = Measure.f1 v.position;
            level = current_level - 1;
            word = 1 :: v.word
          } in
          let f2_v = {
            position = Measure.f2 measure v.position;
            level = current_level - 1;
            word = 2 :: v.word
          } in
          Array.append acc [|f1_v; f2_v|]
        ) [||] acc in
        generate_vertices (current_level - 1) 
          (Array.append acc new_vertices)
    in
    
    let initial_vertices = [|{
      position = measure.lower;
      level;
      word = []
    }; {
      position = measure.upper;
      level;
      word = []
    }|] in
    
    let vertices = generate_vertices level initial_vertices in
    let edges = Array.init (Array.length vertices - 1) 
      (fun i -> (i, i+1)) in
    
    { vertices; edges; measure; level }
end

module HarmonicSpline = struct
  type basis_function = {
    support_left: float;
    support_right: float;
    coefficients: float array;
  }

  type t = {
    basis: basis_function array;
    grid: Grid.t;
    level: int;
  }

  let make grid level =
    let n = Grid.size grid in
    let points = Grid.points grid in
    let dx = Grid.delta grid in
    
    let basis = Array.init n (fun k ->
      if k = 0 then
        {
          support_left = points.(0);
          support_right = points.(1);
          coefficients = [|1.0; -1.0 /. dx|]
        }
      else if k = n-1 then
        {
          support_left = points.(n-2);
          support_right = points.(n-1);
          coefficients = [|-1.0 /. dx; 1.0|]
        }
      else
        {
          support_left = points.(k-1);
          support_right = points.(k+1);
          coefficients = [|1.0 /. dx; -2.0 /. dx; 1.0 /. dx|]
        }
    ) in
    
    { basis; grid; level }

  let evaluate space k x =
    let basis = space.basis.(k) in
    if x < basis.support_left || x > basis.support_right then
      0.0
    else
      let t = (x -. basis.support_left) /. 
        (basis.support_right -. basis.support_left) in
      Array.fold_left (fun acc c -> acc *. t +. c) 0.0 
        basis.coefficients
end
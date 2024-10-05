open Types

let golden_section_search f a b tol =
  let phi = (sqrt 5. +. 1.) /. 2. in
  let rec search a b c d fc fd =
    if abs_float (c -. d) < tol then (c +. d) /. 2.
    else
      let x = if fc < fd then
        b -. (b -. a) /. phi
      else
        a +. (b -. a) /. phi
      in
      let fx = f x in
      if fc < fd then
        search a x d c fx fd
      else
        search x b c d fc fx
  in
  let c = b -. (b -. a) /. phi in
  let d = a +. (b -. a) /. phi in
  search a b c d (f c) (f d)

let nelder_mead f initial_simplex tol max_iter =
  let dim = Array.length initial_simplex.(0) in
  let simplex = Array.copy initial_simplex in
  let values = Array.map f simplex in
  
  let centroid = Array.make dim 0. in
  let reflected = Array.make dim 0. in
  let expanded = Array.make dim 0. in
  let contracted = Array.make dim 0. in
  
  let rec iterate iter =
    if iter >= max_iter then (simplex.(0), values.(0))
    else
      let sorted_indices = Array.init (dim + 1) (fun i -> i)
                           |> Array.sort (fun i j -> compare values.(i) values.(j)) in
      
      (* Calculate centroid *)
      Array.fill centroid 0 dim 0.;
      for i = 0 to dim - 1 do
        for j = 0 to dim - 1 do
          centroid.(j) <- centroid.(j) +. simplex.(sorted_indices.(i)).(j)
        done
      done;
      Array.iteri (fun i x -> centroid.(i) <- x /. float dim) centroid;
      
      (* Reflection *)
      for i = 0 to dim - 1 do
        reflected.(i) <- 2. *. centroid.(i) -. simplex.(sorted_indices.(dim)).(i)
      done;
      let reflected_value = f reflected in
      
      if reflected_value < values.(sorted_indices.(dim - 1)) && reflected_value >= values.(sorted_indices.(0)) then begin
        Array.blit reflected 0 simplex.(sorted_indices.(dim)) 0 dim;
        values.(sorted_indices.(dim)) <- reflected_value;
        iterate (iter + 1)
      end else if reflected_value < values.(sorted_indices.(0)) then begin
        (* Expansion *)
        for i = 0 to dim - 1 do
          expanded.(i) <- 3. *. centroid.(i) -. 2. *. simplex.(sorted_indices.(dim)).(i)
        done;
        let expanded_value = f expanded in
        if expanded_value < reflected_value then begin
          Array.blit expanded 0 simplex.(sorted_indices.(dim)) 0 dim;
          values.(sorted_indices.(dim)) <- expanded_value
        end else begin
          Array.blit reflected 0 simplex.(sorted_indices.(dim)) 0 dim;
          values.(sorted_indices.(dim)) <- reflected_value
        end;
        iterate (iter + 1)
      end else begin
        (* Contraction *)
        for i = 0 to dim - 1 do
          contracted.(i) <- 0.5 *. (centroid.(i) +. simplex.(sorted_indices.(dim)).(i))
        done;
        let contracted_value = f contracted in
        if contracted_value < values.(sorted_indices.(dim)) then begin
          Array.blit contracted 0 simplex.(sorted_indices.(dim)) 0 dim;
          values.(sorted_indices.(dim)) <- contracted_value;
          iterate (iter + 1)
        end else begin
          (* Shrink *)
          for i = 1 to dim do
            for j = 0 to dim - 1 do
              simplex.(sorted_indices.(i)).(j) <- 0.5 *. (simplex.(sorted_indices.(0)).(j) +. simplex.(sorted_indices.(i)).(j))
            done;
            values.(sorted_indices.(i)) <- f simplex.(sorted_indices.(i))
          done;
          iterate (iter + 1)
        end
      end
  in
  iterate 0
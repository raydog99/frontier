type constraint_type =
  | MinWeight of float
  | MaxWeight of float
  | SectorExposure of string * float * float

type t = {
  objective: Portfolio_optimizer.optimization_method;
  constraints: constraint_type list;
}

let create objective constraints =
  { objective; constraints }

let optimize portfolio optimizer =
  let initial_weights = Portfolio.get_weights portfolio in
  let objective_fn = Portfolio_optimizer.optimize portfolio optimizer.objective in
  let constraint_fns = List.map (function
    | MinWeight min_w -> (fun w -> Array.for_all (fun x -> x >= min_w) w)
    | MaxWeight max_w -> (fun w -> Array.for_all (fun x -> x <= max_w) w)
    | SectorExposure (sector, min_exp, max_exp) ->
        (fun w ->
          let sector_exposure = Portfolio.get_sector_exposure portfolio sector in
          sector_exposure >= min_exp && sector_exposure <= max_exp)
  ) optimizer.constraints in
  Optimization.constrained_optimization objective_fn constraint_fns initial_weights
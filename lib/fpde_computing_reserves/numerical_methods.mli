open Types

val solve_pde_adi : 
  Path_dependent_pde.pde_coefficients ->
  non_anticipative_functional ->
  time ->
  time ->
  path ->
  int ->
  int array ->
  boundary_condition array ->
  boundary_condition array ->
  Functional_ito.functional

val parallel_solve_pde : 
  Path_dependent_pde.pde_coefficients ->
  non_anticipative_functional ->
  time ->
  time ->
  path ->
  int ->
  int array ->
  boundary_condition array ->
  boundary_condition array ->
  int ->
  Functional_ito.functional
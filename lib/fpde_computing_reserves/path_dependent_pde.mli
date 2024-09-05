open Types

type pde_coefficients = {
  drift: non_anticipative_functional;
  diffusion: non_anticipative_functional;
  rate: non_anticipative_functional;
  source: non_anticipative_functional;
}

val create_pde_coefficients :
  non_anticipative_functional ->
  non_anticipative_functional ->
  non_anticipative_functional ->
  non_anticipative_functional ->
  pde_coefficients

val path_dependent_pde_operator :
  pde_coefficients ->
  Functional_ito.functional ->
  time ->
  path ->
  float
open Types

type functional = {
  value: non_anticipative_functional;
  horizontal_derivative: non_anticipative_functional;
  vertical_derivative: non_anticipative_functional;
  second_vertical_derivative: non_anticipative_functional;
}

val create_functional :
  non_anticipative_functional ->
  non_anticipative_functional ->
  non_anticipative_functional ->
  non_anticipative_functional ->
  functional

val horizontal_derivative : functional -> non_anticipative_functional
val vertical_derivative : functional -> non_anticipative_functional
val second_vertical_derivative : functional -> non_anticipative_functional

val functional_ito_formula :
  functional ->
  time ->
  path ->
  non_anticipative_functional ->
  non_anticipative_functional ->
  float
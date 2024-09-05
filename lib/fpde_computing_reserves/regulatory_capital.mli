open Types
open Insurance_model

val compute_var :
  insurance_model ->
  path ->
  time ->
  time ->
  float ->
  int ->
  float

val compute_expected_shortfall :
  insurance_model ->
  path ->
  time ->
  time ->
  float ->
  int ->
  float

val compute_regulatory_capital :
  insurance_model ->
  path ->
  time ->
  time ->
  float ->
  int ->
  float

val compute_solvency_capital_requirement :
  insurance_model ->
  path ->
  time ->
  time ->
  float
open Types
open Insurance_model

type parameter = 
  | InterestRate
  | Volatility
  | Participation
  | Mortality

val perturb_model : insurance_model -> parameter -> float -> insurance_model

val compute_sensitivity : 
  insurance_model -> 
  parameter -> 
  float ->
  path ->
  time ->
  time -> 
  float

val stress_test :
  insurance_model ->
  (parameter * float) list ->
  path ->
  time ->
  time ->
  float
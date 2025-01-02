type t = {
  mutable tax_rate: float;
  mutable spending: float;
  mutable debt: float;
}

let create initial_tax_rate =
  { tax_rate = initial_tax_rate; spending = 0.; debt = 0. }

let collect_taxes government total_income =
  let tax_revenue = government.tax_rate *. total_income in
  government.spending <- tax_revenue;
  tax_revenue

let adjust_spending government gdp unemployment_rate =
  let target_spending = 0.3 *. gdp *. (1. +. unemployment_rate) in
  government.spending <- 0.9 *. government.spending +. 0.1 *. target_spending

let issue_debt government =
  government.debt <- government.debt +. (government.spending -. (government.tax_rate *. government.spending))
type t = {
  id: int;
  mutable reserves: float;
  mutable loans: float;
  mutable interest_rate: float;
  mutable equity: float;
}

let create id initial_reserves initial_equity =
  { id; reserves = initial_reserves; loans = 0.; interest_rate = 0.05; equity = initial_equity }

let set_interest_rate bank central_bank_rate =
  bank.interest_rate <- central_bank_rate +. 0.02

let calculate_loan_rate bank central_bank_rate firm_leverage =
  let base_rate = central_bank_rate +. 0.02 in
  let risk_premium = 0.01 *. firm_leverage in
  base_rate +. risk_premium

let process_loan bank amount firm_leverage =
  let loan_rate = calculate_loan_rate bank bank.interest_rate firm_leverage in
  if amount <= bank.reserves && Random.float 1. < (1. -. firm_leverage) then
    (bank.reserves <- bank.reserves -. amount;
     bank.loans <- bank.loans +. amount;
     Some loan_rate)
  else
    None

let receive_payment bank amount =
  bank.loans <- max 0. (bank.loans -. amount);
  bank.reserves <- bank.reserves +. amount

let update_equity bank profit =
  bank.equity <- bank.equity +. profit
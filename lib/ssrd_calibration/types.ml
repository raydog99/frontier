type params = {
  alpha1 : float;
  beta1 : float;
  sigma1 : float;
  alpha2 : float;
  beta2 : float;
  sigma2 : float;
  rho : float;
  r0 : float;
  lambda0 : float;
}

type market_data = {
  zcb_prices : (float * float) list;
  cds_spreads : (float * float) list;
}

type approximation_order = Zeroth | First | Second

type optimization_result = {
  params : params;
  error : float;
}
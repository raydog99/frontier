open Torch
open Types
open Dns_model
open Functional_regression

type dns_fr_model = {
  dns: model;
  kpca: kpca_result;
  beta: Tensor.t;
  n_components: int;
  sigma: float;
}

let create_dns_fr_model lambda state_dim obs_dim n_components sigma =
  let dns = create_dns_model lambda state_dim obs_dim in
  {
    dns;
    kpca = { eigenvalues = Tensor.zeros [1]; eigenvectors = Tensor.zeros [1; 1]; mean = Tensor.zeros [1] };
    beta = Tensor.zeros [n_components; obs_dim];
    n_components;
    sigma;
  }

let fit_dns_fr_model model reference_yields response_yields maturities =
  let dns_fitted = Mle.estimate_parameters model.dns response_yields maturities 0.01 1000 in
  let kpca, beta = functional_regression reference_yields response_yields model.n_components model.sigma in
  { model
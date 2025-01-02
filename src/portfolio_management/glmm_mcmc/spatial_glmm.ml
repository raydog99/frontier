open Torch
open Types

module Correlation = struct
  let matern_kernel distances nu rho =
    let kappa = Tensor.(sqrt (mul (float_tensor [8.0 *. nu]) rho)) in
    let h = Tensor.div distances kappa in
    
    (* Modified Bessel function approximation *)
    let k = Tensor.(
      where (eq h (float_tensor [0.0]))
        (ones_like h)
        (mul
          (exp (neg h))
          (add
            (float_tensor [1.0])
            (add h
              (div (mul h h) (float_tensor [3.0])))))
    ) in
    
    Tensor.(mul (exp (neg h)) k)

  let exponential_kernel distances rho =
    Tensor.(exp (div (neg distances) rho))

  let compute_correlation_matrix coords kernel_fn params =
    let n = Tensor.size coords 0 in
    let distances = Matrix_Ops.Spatial.blocked_distance_matrix coords in
    kernel_fn distances params
end

module MCMC = struct
  let spatial_mala_step model state epsilon =
    let data = model.data in
    let spatial_params = model.spatial_params in
    
    (* Compute spatial correlation *)
    let corr = Correlation.compute_correlation_matrix 
      spatial_params.coords
      (fun d p -> Correlation.matern_kernel d spatial_params.nu p)
      spatial_params.rho in
      
    (* Modified gradient computation incorporating spatial correlation *)
    let grad = Gradients.log_posterior data state.params state.prior_params in
    let spatial_grad = Tensor.(mm corr grad.u) in
    
    (* Generate proposal with spatial correlation *)
    let proposal = {state.params with
      u = Tensor.(add state.params.u
        (add
          (mul spatial_grad (float_tensor [epsilon /. 2.0]))
          (mul (mm (cholesky corr) (randn (size state.params.u)))
             (sqrt (float_tensor [epsilon])))))
    } in
    
    (* Accept/reject *)
    let forward_ll = Models.Logistic.log_likelihood data proposal in
    let backward_ll = Models.Logistic.log_likelihood data state.params in
    
    if Random.float 1.0 < Float.exp (forward_ll -. backward_ll) then
      {state with 
       params = proposal;
       log_prob = forward_ll;
       accepted = state.accepted + 1;
       total = state.total + 1}
    else
      {state with total = state.total + 1}

  let spatial_hmc_step model state epsilon l =
    let data = model.data in
    let spatial_params = model.spatial_params in
    
    (* Compute spatial correlation *)
    let corr = Correlation.compute_correlation_matrix 
      spatial_params.coords
      (fun d p -> Correlation.matern_kernel d spatial_params.nu p)
      spatial_params.rho in
      
    (* Modified HMC incorporating spatial correlation *)
    let modified_state = {state with
      momentum = {state.momentum with
        u = Tensor.(mm corr state.momentum.u)
      }
    } in
    
    HMC.step data modified_state epsilon l
end
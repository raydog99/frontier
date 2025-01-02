open Torch
open Types

let compute tau base_scm abstract_scm interventions divergence =
  (* Sample from both SCMs under each intervention *)
  let errors = List.map (fun i ->
    let base_samples = Scm.sample base_scm 1000 in
    let abstract_samples = Scm.sample abstract_scm 1000 in
    
    (* Apply tau to base samples *)
    let transformed = tau.tau base_samples in
    
    (* Compute divergence *)
    Tensor.item (Divergence.compute divergence transformed abstract_samples)
  ) interventions in
  
  (* Average error across interventions *)
  List.fold_left (+.) 0.0 errors /. float_of_int (List.length errors)
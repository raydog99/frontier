open Types

let optimize base_scm abstract_scm interventions config =
  let modified_config = {config with gamma = config.gamma /. 2.0} in
  
  (* Use the main COTA optimizer with halved gamma *)
  Cota.optimize base_scm abstract_scm interventions modified_config
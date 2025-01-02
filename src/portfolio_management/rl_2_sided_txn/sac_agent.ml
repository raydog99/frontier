open Torch

let get_alpha agent = agent.alpha

let save agent filename =
  let state = Hashtbl.create 7 in
  Hashtbl.add state "actor" (Layer.state_dict agent.actor);
  Hashtbl.add state "critic1" (Layer.state_dict agent.critic1);
  Hashtbl.add state "critic2" (Layer.state_dict agent.critic2);
  Hashtbl.add state "log_alpha" agent.log_alpha;
  Serialize.save ~filename state

let load agent filename =
  let state = Serialize.load ~filename in
  Layer.load_state_dict agent.actor (Hashtbl.find state "actor");
  Layer.load_state_dict agent.critic1 (Hashtbl.find state "critic1");
  Layer.load_state_dict agent.critic2 (Hashtbl.find state "critic2");
  agent.log_alpha <- Hashtbl.find state "log_alpha";
  agent.alpha <- Tensor.to_float0_exn (Tensor.exp agent.log_alpha)
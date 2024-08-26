open Torch
open Meta_controller
open Order_generator

type t = {
  meta_controller: Meta_controller.t;
  order_generator: Order_generator.t;
}

let create ~num_timesteps ~channels ~num_samples ~seq_length ~condition_dim ~encoder_type
           ~initial_price ~risk_aversion ~fundamental_weight ~chartist_weight ~noise_weight =
  {
    meta_controller = Meta_controller.create ~num_timesteps ~channels ~num_samples ~seq_length ~condition_dim ~encoder_type;
    order_generator = Order_generator.create ~initial_price ~risk_aversion ~fundamental_weight ~chartist_weight ~noise_weight;
  }

let train t ~data ~conditions ~learning_rate ~num_epochs =
  let preprocessed_data = Utils.preprocess_data data in
  Meta_controller.train t.meta_controller ~data:preprocessed_data ~conditions ~learning_rate ~num_epochs

let generate t control_target ~guidance_scale =
  let market_states = Meta_controller.generate_market_states t.meta_controller control_target ~guidance_scale in
  Order_generator.generate_orders t.order_generator market_states

let evaluate t ~real_data ~control_target ~guidance_scale =
  let generated_orders = generate t control_target ~guidance_scale in
  let generated_prices = List.map (fun order -> order.Order_generator.price) generated_orders |> Tensor.of_float1 in
  let control_error = Utils.evaluate_control_error control_target generated_prices in
  let fidelity_error = Utils.evaluate_fidelity real_data generated_prices in
  (control_error, fidelity_error)
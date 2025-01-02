open Utils
open Pricing

module BlackScholesOperator = struct
  type params = {
    r: float;
    sigma: float;
  }

  let make r sigma = 
    if sigma *. sigma >= 4.0 *. r then
      failwith "Coercivity condition not satisfied: sigma^2 must be < 4r";
    { r; sigma }

  let apply params grid u =
    let n = Array.length u in
    let result = Array.make n 0.0 in
    let dx = Grid.delta grid in
    let points = Grid.points grid in
    
    for i = 1 to n-2 do
      let x = points.(i) in
      let d2u = (u.(i+1) -. 2.0 *. u.(i) +. u.(i-1)) /. (dx *. dx) in
      let du = (u.(i+1) -. u.(i-1)) /. (2.0 *. dx) in
      
      result.(i) <- 
        -. params.sigma *. params.sigma *. x *. x *. d2u /. 2.0
        -. params.r *. x *. du
        +. params.r *. u.(i)
    done;
    result

  let check_coercivity params =
    params.sigma *. params.sigma < 4.0 *. params.r
end

module GeneralizedBlackScholesSolver = struct
  type params = {
    measure: Measure.t;
    bs_params: BlackScholesOperator.params;
    grid: Grid.t;
    dt: float;
    n_steps: int;
    strike: float;
    option_type: OptionType.t;
    scheme: NumericalScheme.scheme;
  }

  let make_payoff params grid =
    let points = Grid.points grid in
    Array.map (fun s -> OptionType.payoff params.option_type params.strike s) points

  (* Main solver *)
  let solve params =
    (* Validate stability conditions *)
    let stability = StabilityAnalysis.analyze_stability params in
    if not stability.is_stable then
      failwith "Numerical scheme is unstable with current parameters";

    let n = Grid.size params.grid in
    let current = Tensor.zeros [n] in
    let next = Tensor.zeros [n] in
    let solution = Tensor.zeros [params.n_steps; n] in
    
    (* Initial condition *)
    let initial_condition = make_payoff params params.grid in
    Tensor.copy_ ~src:(Tensor.of_float1 initial_condition) ~dst:current;
    for i = 0 to n-1 do
      Tensor.set solution [0; i] initial_condition.(i)
    done;

    (* System matrix setup *)
    let system_matrix = NumericalScheme.make_system_matrix 
      params.scheme params.bs_params params.grid params.dt in
    
    (* Time stepping with measure adjustment *)
    for t = 1 to params.n_steps-1 do
      (* Apply generalized operator *)
      let bs_applied = BlackScholesOperator.apply params.bs_params params.grid 
        (Array.init n (fun i -> Tensor.get current [i])) in
      
      (* Integrate against measure *)
      for i = 0 to n-1 do
        let measure_integral = Measure.integrate_psi params.measure t i in
        Tensor.set next [i] (bs_applied.(i) *. measure_integral)
      done;

      (* Solve system *)
      let next = Tensor.solve system_matrix current in
      
      (* Apply boundary conditions *)
      let t_val = float_of_int t *. params.dt in
      for i = 0 to n-1 do
        if Grid.is_boundary params.grid i then
          let x = (Grid.points params.grid).(i) in
          let bc = OptionType.boundary_condition params.option_type 
            params.strike params.bs_params.r t_val x in
          Tensor.set next [i] bc
      done;

      (* Store result and swap buffers *)
      for i = 0 to n-1 do
        Tensor.set solution [t; i] (Tensor.get next [i])
      done;
      Tensor.copy_ ~src:next ~dst:current
    done;

    (* Calculate Greeks *)
    let greeks = Greeks.calculate params.bs_params params.grid params.dt solution in
    solution, greeks

  let validate params solution =
    (* Calculate exact solution for comparison *)
    let points = Grid.points params.grid in
    let exact = Array.init (Array.length points) (fun i ->
      let s = points.(i) in
      let t = float_of_int params.n_steps *. params.dt in
      OptionType.payoff params.option_type params.strike s *.
      exp(-. params.bs_params.r *. t)) in
    
    (* Verify accuracy *)
    let error_stats = Validation.analyze_error solution exact params.grid in
    error_stats.l2_error < 1e-6 && error_stats.max_error < 1e-6
end
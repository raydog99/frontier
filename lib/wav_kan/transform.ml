open Torch

module DWT = struct
  type coefficients = {
    approximation: Tensor.t;
    details: Tensor.t list;
    level: int;
  }

  module FilterBank = struct
    type t = {
      decomp_low: Tensor.t;
      decomp_high: Tensor.t;
      recon_low: Tensor.t;
      recon_high: Tensor.t;
    }

    let create_daubechies4 () = {
      decomp_low = Tensor.of_float1 [|0.6830127; 1.1830127; 
                                     0.3169873; -0.1830127|];
      decomp_high = Tensor.of_float1 [|-0.1830127; -0.3169873; 
                                      1.1830127; -0.6830127|];
      recon_low = Tensor.of_float1 [|-0.1830127; 0.3169873; 
                                    1.1830127; 0.6830127|];
      recon_high = Tensor.of_float1 [|0.6830127; -1.1830127; 
                                     0.3169873; 0.1830127|];
    }

    let apply_filter signal filter padding =
      let pad_size = (Tensor.shape filter |> List.hd) / 2 in
      let padded = match padding with
        | `Zero -> Tensor.pad signal ~pad:[pad_size; pad_size] ~mode:Constant
        | `Reflect -> Tensor.pad signal ~pad:[pad_size; pad_size] ~mode:Reflect
        | `Periodic -> 
            let left = Tensor.slice signal ~dim:0 
                        ~start:((Tensor.shape signal |> List.hd) - pad_size) 
                        ~length:pad_size in
            let right = Tensor.slice signal ~dim:0 
                         ~start:0 ~length:pad_size in
            Tensor.cat [left; signal; right] ~dim:0
      in
      Tensor.conv1d padded ~weight:(Tensor.unsqueeze filter ~dim:0) 
                          ~stride:1 ~padding:Valid
  end

  let downsample tensor =
    Tensor.slice tensor ~dim:0 ~start:0 ~step:(Some 2) ~length:None

  let upsample tensor =
    let size = Tensor.shape tensor |> List.hd in
    let result = Tensor.zeros [size * 2] in
    Tensor.copy_ tensor ~src:(Tensor.slice result ~dim:0 
                                           ~start:0 
                                           ~step:(Some 2) 
                                           ~length:None);
    result

  let decompose signal level =
    let filter_bank = FilterBank.create_daubechies4 () in
    let rec decompose_level signal level acc =
      if level = 0 then 
        { approximation = signal;
          details = List.rev acc;
          level }
      else
        let low = FilterBank.apply_filter signal filter_bank.decomp_low `Reflect
                 |> downsample in
        let high = FilterBank.apply_filter signal filter_bank.decomp_high `Reflect
                  |> downsample in
        decompose_level low (level - 1) (high :: acc)
    in
    decompose_level signal level []

  let reconstruct coeffs =
    let filter_bank = FilterBank.create_daubechies4 () in
    let rec reconstruct_impl approx = function
      | [] -> approx
      | detail :: rest ->
          let up_approx = upsample approx in
          let up_detail = upsample detail in
          let rec_approx = FilterBank.apply_filter up_approx 
                            filter_bank.recon_low `Zero in
          let rec_detail = FilterBank.apply_filter up_detail 
                            filter_bank.recon_high `Zero in
          let combined = Tensor.(+) rec_approx rec_detail in
          reconstruct_impl combined rest
    in
    reconstruct_impl coeffs.approximation coeffs.details
end
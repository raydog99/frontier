open Cmdliner
open Data_loader
open Model_comparison

let data_file =
  let doc = "Input file containing stock data." in
  Arg.(required & pos 0 (some string) None & info [] ~docv:"DATA_FILE" ~doc)

let input_format =
  let doc = "Input file format (CSV, JSON, or SQLite)." in
  Arg.(value & opt (enum ["csv", CSV; "json", JSON; "sqlite", SQLite]) CSV & info ["f"; "format"] ~docv:"FORMAT" ~doc)

let batch_size =
  let doc = "Batch size for training." in
  Arg.(value & opt int 100 & info ["b"; "batch-size"] ~docv:"BATCH_SIZE" ~doc)

let num_bootstraps =
  let doc = "Number of bootstrap iterations for validation." in
  Arg.(value & opt int 1000 & info ["n"; "num-bootstraps"] ~docv:"NUM_BOOTSTRAPS" ~doc)

let k_folds =
  let doc = "Number of folds for k-fold cross-validation." in
  Arg.(value & opt int 5 & info ["k"; "k-folds"] ~docv:"K_FOLDS" ~doc)

let run data_file input_format batch_size num_bootstraps k_folds =
  let stock_data = load_data data_file input_format in
  run_analysis stock_data batch_size num_bootstraps k_folds

let cmd =
  let doc = "Run Five-Factor Asset Pricing Model analysis" in
  let man = [
    `S Manpage.s_description;
    `P "This command runs a comprehensive analysis of the Five-Factor Asset Pricing Model on the provided stock data.";
    `P "It supports multiple input formats (CSV, JSON, SQLite) and performs model comparison, robustness checks, statistical tests, and generates visualizations.";
    `P "The results are printed to stdout and visualization files are saved in the current directory.";
  ] in
  Term.(ret (const run $ data_file $ input_format $ batch_size $ num_bootstraps $ k_folds)),
  Term.info "five_factor_model" ~version:"1.0" ~doc ~man

let () = Term.(exit @@ eval cmd)
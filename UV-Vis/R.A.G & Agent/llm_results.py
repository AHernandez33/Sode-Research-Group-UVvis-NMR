# from __future__ import annotations

# import argparse
# from pathlib import Path

# import matplotlib.pyplot as plt
# import pandas as pd


# PROJECT_FOLDER = Path(__file__).resolve().parent

# DEFAULT_RESULTS_FILE = (
#     PROJECT_FOLDER
#     / "llm_comparison_results.csv"
# )

# DEFAULT_SUMMARY_FILE = (
#     PROJECT_FOLDER
#     / "llm_simple_comparison_summary.csv"
# )


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Create a simplified Qwen, Llama, and Gemma comparison "
#             "using MSE, lambda-max error, runtime, and success rate."
#         )
#     )

#     parser.add_argument(
#         "--results-file",
#         type=str,
#         default=str(
#             DEFAULT_RESULTS_FILE
#         ),
#         help="Path to llm_comparison_results.csv."
#     )

#     parser.add_argument(
#         "--summary-file",
#         type=str,
#         default=str(
#             DEFAULT_SUMMARY_FILE
#         ),
#         help="Path for the simplified summary CSV."
#     )

#     return parser


# def resolve_path(
#         value: str
# ) -> Path:
#     path = Path(
#         value
#     ).expanduser()

#     if not path.is_absolute():
#         path = (
#             PROJECT_FOLDER
#             / path
#         )

#     return path.resolve()


# def numeric_mean(
#         series: pd.Series
# ) -> float:
#     values = pd.to_numeric(
#         series,
#         errors="coerce"
#     )

#     return float(
#         values.mean()
#     )


# def save_bar_chart(
#         dataframe: pd.DataFrame,
#         value_column: str,
#         ylabel: str,
#         title: str,
#         output_file: Path
# ) -> None:
#     figure = plt.figure(
#         figsize=(
#             8,
#             6
#         )
#     )

#     plt.bar(
#         dataframe[
#             "llm"
#         ],
#         dataframe[
#             value_column
#         ]
#     )

#     plt.xlabel(
#         "LLM"
#     )

#     plt.ylabel(
#         ylabel
#     )

#     plt.title(
#         title
#     )

#     plt.grid(
#         axis="y",
#         alpha=0.3
#     )

#     plt.tight_layout()

#     figure.savefig(
#         output_file,
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.close(
#         figure
#     )


# def main() -> None:
#     args = build_parser().parse_args()

#     results_file = resolve_path(
#         args.results_file
#     )

#     summary_file = resolve_path(
#         args.summary_file
#     )

#     if not results_file.exists():
#         raise FileNotFoundError(
#             f"Comparison results file not found: {results_file}"
#         )

#     dataframe = pd.read_csv(
#         results_file
#     )

#     required_columns = {
#         "llm",
#         "prediction_status",
#         "spectrum_mse",
#         "lambda_max_absolute_error",
#         "runtime_seconds"
#     }

#     missing_columns = required_columns.difference(
#         dataframe.columns
#     )

#     if missing_columns:
#         raise ValueError(
#             "The results file is missing these columns: "
#             f"{sorted(missing_columns)}"
#         )

#     summary_rows = []

#     for llm_name, group in dataframe.groupby(
#         "llm",
#         sort=True
#     ):
#         successful = group[
#             group[
#                 "prediction_status"
#             ].astype(
#                 str
#             ).str.lower().eq(
#                 "success"
#             )
#         ].copy()

#         total_runs = int(
#             len(
#                 group
#             )
#         )

#         successful_runs = int(
#             len(
#                 successful
#             )
#         )

#         success_rate = float(
#             successful_runs
#             / max(
#                 total_runs,
#                 1
#             )
#             * 100.0
#         )

#         summary_rows.append(
#             {
#                 "llm": llm_name,
#                 "mean_mse": numeric_mean(
#                     successful[
#                         "spectrum_mse"
#                     ]
#                 ),
#                 "mean_lambda_max_error_nm": numeric_mean(
#                     successful[
#                         "lambda_max_absolute_error"
#                     ]
#                 ),
#                 "mean_runtime_seconds": numeric_mean(
#                     group[
#                         "runtime_seconds"
#                     ]
#                 ),
#                 "success_rate_percent": success_rate,
#                 "total_runs": total_runs,
#                 "successful_runs": successful_runs,
#                 "failed_runs": (
#                     total_runs
#                     - successful_runs
#                 )
#             }
#         )

#     summary_dataframe = pd.DataFrame(
#         summary_rows
#     )

#     summary_dataframe.sort_values(
#         by=[
#             "mean_mse",
#             "mean_lambda_max_error_nm"
#         ],
#         inplace=True,
#         na_position="last"
#     )

#     summary_file.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     summary_dataframe.to_csv(
#         summary_file,
#         index=False
#     )

#     output_folder = summary_file.parent

#     mse_plot_file = (
#         output_folder
#         / "llm_mean_mse.png"
#     )

#     lambda_error_plot_file = (
#         output_folder
#         / "llm_mean_lambda_max_error.png"
#     )

#     runtime_plot_file = (
#         output_folder
#         / "llm_mean_runtime.png"
#     )

#     success_rate_plot_file = (
#         output_folder
#         / "llm_success_rate.png"
#     )

#     save_bar_chart(
#         dataframe=summary_dataframe,
#         value_column="mean_mse",
#         ylabel="Mean spectrum MSE",
#         title="Mean UV-Vis Spectrum MSE by LLM",
#         output_file=mse_plot_file
#     )

#     save_bar_chart(
#         dataframe=summary_dataframe,
#         value_column="mean_lambda_max_error_nm",
#         ylabel="Mean absolute lambda-max error (nm)",
#         title="Mean Lambda-Max Error by LLM",
#         output_file=lambda_error_plot_file
#     )

#     save_bar_chart(
#         dataframe=summary_dataframe,
#         value_column="mean_runtime_seconds",
#         ylabel="Mean runtime (seconds)",
#         title="Mean Agent Runtime by LLM",
#         output_file=runtime_plot_file
#     )

#     save_bar_chart(
#         dataframe=summary_dataframe,
#         value_column="success_rate_percent",
#         ylabel="Success rate (%)",
#         title="Agent Success Rate by LLM",
#         output_file=success_rate_plot_file
#     )

#     print()
#     print(
#         "Simplified LLM Comparison"
#     )
#     print(
#         "-------------------------"
#     )
#     print(
#         summary_dataframe.to_string(
#             index=False
#         )
#     )

#     print()
#     print(
#         "Summary CSV:",
#         summary_file
#     )
#     print(
#         "MSE plot:",
#         mse_plot_file
#     )
#     print(
#         "Lambda-max error plot:",
#         lambda_error_plot_file
#     )
#     print(
#         "Runtime plot:",
#         runtime_plot_file
#     )
#     print(
#         "Success-rate plot:",
#         success_rate_plot_file
#     )

# if __name__ == "__main__":
#     main()
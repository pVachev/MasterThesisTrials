import pandas as pd


def export_model_results_to_excel(results, output_file: str) -> None:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        best_seeds_rows = []

        for res in results:
            sheet_suffix = res.spec.code
            summary_sheet = f"Summary_{sheet_suffix}"
            sweep_sheet = f"Sweep_{sheet_suffix}"
            detail_sheet = f"Detail_{sheet_suffix}"

            startrow = 0

            if res.regime_summary is not None:
                res.regime_summary.to_excel(writer, sheet_name=summary_sheet, index=False, startrow=startrow)
                startrow += len(res.regime_summary) + 3
            
            res.moment_table.to_excel(writer, sheet_name=summary_sheet, startrow=startrow)
            startrow += len(res.moment_table) + 3

            res.trans.to_excel(writer, sheet_name=summary_sheet, startrow=startrow)
            startrow += len(res.trans) + 3

            res.duration.to_excel(writer, sheet_name=summary_sheet, startrow=startrow)
            startrow += len(res.duration) + 3

            res.chatter.to_excel(writer, sheet_name=summary_sheet, index=False, startrow=startrow)
            startrow += len(res.chatter) + 3

            if res.corr_table is not None:
                res.corr_table.to_excel(writer, sheet_name=summary_sheet, index=False, startrow=startrow)

            res.sweep.to_excel(writer, sheet_name=sweep_sheet, index=False)
            res.pp.df_m.to_excel(writer, sheet_name=detail_sheet)

            best_seeds_rows.append({
                "model_code": res.spec.code,
                "model_label": res.spec.label,
                "best_seed": res.best_seed,
            })

        pd.DataFrame(best_seeds_rows).to_excel(writer, sheet_name="Best_Seeds", index=False)

    print(f"Saved results to {output_file}")
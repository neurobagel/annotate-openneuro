## Bulk annotations files

### `participants_tsv_*_summary.tsv`
Tables containing data summaries for each column across all datasets, 
and for each unique value in columns determined to be categorical (based on heuristics) across all datasets.

### `participants_tsv_*_summary_first_guess.tsv`
Summary tables with heuristic-based column and value annotations for identifier and demographic standardized variables.

### `participants_tsv_*_summary_first_guess_manual_pass.tsv`
- Annotated obvious session ID columns missed by heuristics
- Annotated obvious "Other" sex for sex columns (not including specified non-binary or undisclosed sex)
- Annotated age columns missed by heuristics (except those with unsupported units or formats)
- Marked "exclude" for columns with clear quality issues (e.g., swapped age & sex)

### `participants_tsv_*_summary_first_guess_manual_pass_with_assessments.tsv`
- Includes annotations for columns detected as assessment columns by an LLM, including the assessment term information and mapping confidence

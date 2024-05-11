Validation and scoring scripts for the challenge.

Metrics returned and used for ranking are:

- RMSE
- Pearson (two-sided, resampling method)

## Usage

### Validate

```text
python validate.py -p PATH/TO/PREDICTIONS_FILE.CSV -g PATH/TO/GOLDSTANDARD_FILE.CSV [-o RESULTS_FILE]
```
If `-o/--output` is not provided, then results will print to STDOUT, e.g.

```json
{"submission_status": "VALIDATED", "submission_errors": ""}
```

What it will check for:

- four columns named `Dataset`, `Mixture_1`, `Mixture_2`, and `Predicted_Experimental_Values` (extraneous columns will be ignored)
- `Dataset` values are strings
- `Mixture_1` and `Mixture_2` are integers
- `disease_probability` values are floats between 0.0 and 1.0, and cannot be null/None
- there is one prediction per mixture (so, no missing or duplicated combination of Dataset + Mixture_1 + Mixture_2)
- there are no extra predictions (so, no unknown combination of Dataset + Mixture_1 + Mixture_2)

### Score

```text
python score.py -p PATH/TO/PREDICTIONS_FILE.CSV -g PATH/TO/GOLDSTANDARD_FILE.CSV [-o RESULTS_FILE]
```

If `-o/--output` is not provided, then results will output to `results.json`.

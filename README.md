# Olfactory Mixtures Predictions Evaluation

Validation and scoring scripts for the
[Olfactory Mixtures Predictions DREAM Challenge] and
[DREAM Olfactory Mixtures Prediction Challenge 2025].
For collecting writeups, see `writeup-workflow`.

## Evaluation Overview

The challenge is split into two phases:

- **Leaderboard phase**: participants submit a prediction
file that will be evaluated against a validation dataset.

- **Final phase**: participants submit a prediction file
that will be evaluated against a test dataset. 

Metrics returned and used for ranking are:

#### 2024

- **Mean Root Mean Square Error (mRMSE)**: This metric
measures the average difference between predicted and
actual values across all data points. A lower mRMSE
indicates a model with higher accuracy, as it demonstrates
a smaller average error between the model's predictions
and the actual outcomes.

- **Pearson Correlation**: This metric assesses the
linear relationship between the predicted and actual 
values. A higher Pearson correlation value signifies a
strong positive relationship, indicating that the model's
predictions align closely with the actual data.

#### 2025

- **Cosine distance**: This metric measures the difference
in orientation between predicted and actual vectors,
independent of their magnitudes. A lower cosine distance
indicates greater similarity in direction, demonstrating
that the model effectively captures the overall pattern
or profile of the data.

- **Pearson correlation**: This metric evaluates the strength
of the linear relationship between predicted and actual values.
A higher Pearson correlation indicates a stronger positive
linear relationship, suggesting that the model's predictions
closely match the observed data.


By combining these two metrics for each year, the challenges provide
a comprehensive evaluation of each model's accuracy and
predictive power, ensuring that both the magnitude of the
prediction errors and the consistency of the predicted
trends are taken into account.
## Usage - Python

### Validate

```text
python validate.py \
  -p PATH/TO/PREDICTIONS_FILE.CSV \
  -g PATH/TO/GOLDSTANDARD_FILE.CSV [-o RESULTS_FILE]
```
If `-o/--output` is not provided, then results will print
to STDOUT, e.g.

```json
{"submission_status": "VALIDATED", "submission_errors": ""}
```

What it will check for:

- Four columns named `Dataset`, `Mixture_1`, `Mixture_2`, 
  and `Predicted_Experimental_Values` (extraneous columns 
  will be ignored)
- `Dataset` values are strings
- `Mixture_1` and `Mixture_2` are integers
- `disease_probability` values are floats between 0.0 
  and 1.0, and cannot be null/None
- There is exactly one prediction per mixture (so: no missing
  or duplicated combination of Dataset + Mixture_1 + Mixture_2)
- There are no extra predictions (so: no unknown combination
  of Dataset + Mixture_1 + Mixture_2)

### Score

```text
python score.py \
  -p PATH/TO/PREDICTIONS_FILE.CSV \
  -g PATH/TO/GOLDSTANDARD_FILE.CSV [-o RESULTS_FILE]
```

If `-o/--output` is not provided, then results will output
to `results.json`.

[Olfactory Mixtures Predictions DREAM Challenge]: https://www.synapse.org/#!Synapse:syn53470621/wiki/626022
[DREAM Olfactory Mixtures Prediction Challenge 2025]: https://www.synapse.org/Synapse:syn64743570

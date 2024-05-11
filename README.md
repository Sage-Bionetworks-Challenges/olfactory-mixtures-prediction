# Olfactory Mixtures Predictions Evaluation Workflow

The repository contains the evaluation workflow for the
[Olfactory Mixtures Predictions DREAM Challenge].

## Evaluation Overview

The challenge is split into two phases:

- **Leaderboard phase**: participants submit a Docker model to generate a CSV prediction file that will be evaluated against a validation dataset. ([Sample prediction file format](https://www.synapse.org/#!Synapse:syn57406750))

- **Final phase**: participants submit a Docker model to generate a CSV prediction file that will be evaluated against a test dataset. ([Sample prediction file format](https://www.synapse.org/#!Synapse:syn57405848))

Metrics returned and used for ranking are:

- **Mean Root Mean Square Error (mRMSE)**: This metric measures the average difference between predicted and actual values across all data points. A lower mRMSE indicates a model with higher accuracy, as it demonstrates a smaller average error between the model's predictions and the actual outcomes.

- **Pearson Correlation**: This metric assesses the linear relationship between the predicted and actual values. A higher Pearson correlation value signifies a strong positive relationship, indicating that the model's predictions align closely with the actual data.

By combining these two metrics, the challenge provides a comprehensive evaluation of each model's accuracy and predictive power, ensuring that both the magnitude of the prediction errors and the consistency of the predicted trends are taken into account.

Code for the above computations are available in the `evaluation` folder of the repo.

[Olfactory Mixtures Predictions DREAM Challenge]: https://www.synapse.org/#!Synapse:syn53470621/wiki/626022

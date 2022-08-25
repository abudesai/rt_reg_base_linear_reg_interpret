Linear Regression using Interpret ML and Scikit-Learn Elastic Net as the linear class in model interpretability for Regression.

- glass box
- linear regression
- model explainability
- interpret ml
- xai
- global explanations
- elasticnet
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker
- abalone
- auto prices
- computer activity
- heart disease
- white wine quality
- ailerons

This is an explainable version of Elastic Net Regressor- linear glass box model using the InterpretML package.

Using InterpretML, the relationship between the response and its explanatory variables are modeled with linear predictor functions.

Global explanations helps users understand how the model makes decisions, based on a global view of its features and each feature importance. These explanations can be viewed by means of various plots.
Since this is a linear model, global explanations will also match local explanations.

Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent.

HPT includes choosing the optimal values for regularization parameters.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, InterpretML for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.

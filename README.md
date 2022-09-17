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

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for regularization parameters.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, InterpretML for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.

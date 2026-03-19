# ST-498
_____
In the code folder on **MAIN**, there are a few notebooks: 
1) **SARIMA_fitting.py** This contains all the code (using classes, functions and the like) to automatically fit and store the best fitting SARIMA model. Currently, it selects the model with the best AIC, however we can change this as we see fit. While fitting each transformation, the model is verbose and will return various other statistics on the fit of our model. This will allow us to easily identify any problems that may have been glazed over when picked the best fit.
2) **Modeling.ipynb** This notebook contains Danny's prelimary attempts at fitting models - but due not incorporating lags, it does not serve much purpose. It serves just as some pre-built models if we choose to use them.
3) **DataCompiliation.ipynb** This contains most, if not all of the needed columns for our project. Additionally, it is set up in a way that we can easily move it to a .py file when we want to "push to production" and will provide a clean pipeline. **Currently it does not contain lags.** This is something that we will work on
4) **EDA.ipynb** This version will be removed when Andres finishes his work.

On **DH_Work** there are a few additional notebooks:
1) **ML_Pipeline.py** Similar to SARIMA_fitting.py, it is a file that can be called easily into other files to run and fit various ML models and provide key stats for each model.
2)**TimeSeriesAnalysis_Testing.ipynb** This shows example outputs from SARIMA_fitting.py- serves as a sample baseline for reference

**Next steps**
We need to break down how much to lag each variable, each one will have its own literature, this is to understand for modeling.
Create addtional columns with lagged data, will need to incorporate into SARIMAX (X is for exodgenous) & all ML models 


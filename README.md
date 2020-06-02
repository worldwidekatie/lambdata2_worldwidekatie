# lambdata_worldwidekatie

## Installation
Use !pip install -i https://test.pypi.org/simple/ lambdata-worldwidekatie==1.3

Then import my_lambdata.my_mod as md

Using md.enlarge(x), you can multiply integers(x) by 10.

Using md.t_test(df, 'class', 'class1', 'class2'), you can make at table to t-tests with 'class' being the name of the column with the two independent variables and 'class1' and 'class2' being how they are represented in that column.

Using md.chi2(df, variable), you can make a table of chi^2 results with variable being the indepedent variable.

Using md.pac_explain(pipeline), you can make a pandas dataframe of features selected along side their coefficients for a pipeline that includes a tfidvectorizer, selectpercentile, and passive-agressive classifier trained on a textual dataset.

Using md.importances(sample, importance_df), with the importance_df outputted by md.pac_explain(pipeline), you can make a pandas dataframe of features used to make a single prediction by your model alongside their coefficients. 






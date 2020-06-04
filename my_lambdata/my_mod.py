from scipy import stats
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.datasets import make_classification
import numpy as np

def t_test(dataframe, column, group_a, group_b):
  """
  This function takes in a pandas dataframe, the name of
  the column with your independent variables as a string
  and the names of your two independent variables also
  as strings. It returns a pandas dataframe of t-statistics and
  p-values for al of the dependent variables in the
  other columns.
  """
  groupaa = dataframe[dataframe[column]==group_a]
  groupa = groupaa.drop(columns=[column])
  groupbb = dataframe[dataframe[column]==group_b]
  groupb = groupbb.drop(columns=[column])
  themes = groupa.columns.tolist()
  output=[] 

  for theme in themes:
    output.append([theme, groupa[theme].mean(), groupb[theme].mean(), 
                   stats.ttest_ind(groupa[theme], groupb[theme], nan_policy='omit')])

  output2 = pd.DataFrame([[i[0], i[1], i[2], i[3][0], i[3][1]] for i in output],
                  columns=['Variable', 'Group A Mean', 'Group B Mean', 'T-Statistic', 'P-Value'])

  return output2.sort_values(by=['P-Value'])

def chi2(df, dependent_var):
  """
  This function takes in a pandas dataframe and the name
  of the column with the independent variable as a string.
  It returns a pandas dataframe of chi^2 and p-values for
  all of the other columns/dependent variables.
  """
  columns = df.columns.tolist()
  columns.remove(dependent_var)
  output=[]

  for column in columns:
    crosstab = pd.crosstab(df[dependent_var], df[column])
    crosstab = crosstab.values
    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
    output.append([column, chi2, p_value])
  
  df2 = pd.DataFrame([[i[0], i[1], i[2]] for i in output],
                  columns=['Independent Variable', 'Chi^2', 'P-Value'])

  return df2.sort_values(by=['P-Value'])


def enlarge(n):
  """
  This functions takes a number as an integer or float
  and multiplies it by 100.
  """
  return int(n)*100

def pac_explain(pipeline):
  """
  This function takes a pipeline fitted on a textual dataset
  with a tfidvectorizer, selectpercentile, and a 
  passive agressive classifier. It outputs a dataframe of
  selected features and their coefficients.
  """
  clf = pipeline.named_steps.passiveaggressiveclassifier
  weights = clf.coef_
  weights = list(weights[0])
  vect = pipeline.named_steps.tfidfvectorizer
  vocab = list(vect.vocabulary_)
  sp = pipeline.named_steps.selectpercentile
  select_p = list(sp.get_support())
  df1 = pd.DataFrame({'Word': vocab, 'Used': select_p})
  df1 = df1[df1['Used']==True]
  features = list(df1['Word'])
  importances = pd.DataFrame({"Feature": features, "Weight": weights})
  slim_importances = importances[importances["Weight"]!=0].sort_values(by="Weight", ascending='False')
  return slim_importances

def importances(sample, importance_df):
  """
  This function is for explaining predictions made by a
  pipeline that has been put through the pac_explain()
  function. It takes the output datframe and the
  sample string you want to predict and outputs a 
  pandas dataframe with all of the words from that string
  your model used to make a prediction and their 
  coefficients.
  """
  x = sample.lower().split()
  features = []
  weights = []
  array_weights = np.array(importance_df)
  for word in x:
    for i in array_weights:
      if word == i[0]:
        features.append(word)
        weights.append(i[1])
  
  feature_importances = pd.DataFrame({'Feature': features, 'Weight': weights})
  
  return feature_importances

if __name__ == "__main__":
    pass
    # only run the following code
    # when we run this script rom the command-line
    # otherwise don't invoke this code 
    # (for example when importing from another file)

    x = 5
    print(enlarge(x))


    z = input("Please choose a number to enlarge:")
    print(enlarge(z))

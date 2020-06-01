from scipy import stats
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.datasets import make_classification
import numpy as np

def t_test(dataframe, column, group_a, group_b):
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
    return int(n)*100

def pac_explain(pipeline):
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

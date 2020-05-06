from scipy import stats
import pandas as pd

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

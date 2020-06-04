import unittest
import pandas as pd
import numpy as np
from my_lambdata.my_mod import enlarge
from my_lambdata.my_mod import t_test
from my_lambdata.my_mod import chi2


iris = 'my_lambdata/tests/iris.csv'
ttest = 'my_lambdata/tests/t_test.csv'
chi2_test = 'my_lambdata/tests/chi2_test.csv'

class TestMyMod(unittest.TestCase):

    def test_enlarge(self):
        self.assertEqual(enlarge(5), 500)

    def test_t_test(self):
        df = pd.read_csv(iris, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
        df = t_test(df, 'class', 'Iris-setosa', 'Iris-virginica')
        df2 = pd.read_csv(ttest, index_col='Unnamed: 0')
        for i in df:
            self.assertIn(i, df2)

    def test_chi2(self):
        df = pd.read_csv(iris, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
        df = chi2(df, 'sepal_length')
        df2 = pd.read_csv(chi2_test, index_col='Unnamed: 0')
        for i in df:
            self.assertIn(i, df2)

if __name__ == '__main__':
    unittest.main()
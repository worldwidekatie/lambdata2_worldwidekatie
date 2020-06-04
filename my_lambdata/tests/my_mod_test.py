import unittest
import pandas as pd
from my_lambdata.my_mod import enlarge
from my_lambdata.my_mod import t_test

class TestMyMod(unittest.TestCase):

    def test_enlarge(self):
        self.assertEqual(enlarge(5), 500)
    
    df1 = pd.read_csv("iris.csv", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    df2 = pd.read_csv("t_test.csv")

    def test_t_test(self):
        self.assertEqual(t_test(df1, 'class', 'Iris-setosa', 'Iris-virginica'), df2)

if __name__ == '__main__':
    unittest.main()
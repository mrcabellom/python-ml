import pandas as pd
import numpy as np

class GivenFixture(object):
    """Fixture Tests"""

    def __init__(self):
        pass
    @staticmethod
    def random_perceptron_dataset(rows=2, cols=2, columns=None):
        dataframe = pd.DataFrame(np.random.rand(rows, cols), columns=columns)
        dataframe.replace(to_replace=columns[-1], value=0, inplace=True)
        return dataframe

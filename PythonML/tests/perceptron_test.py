from tests.base_test import BaseTestCase
from algoritms.perceptron import Perceptron

class TestPerceptron(BaseTestCase):

    def test_should_obtain_weights(self):

        dataset = self._given.random_perceptron_dataset(
            rows=10,
            cols=3,
            columns=['petallength', 'sepallength', 'class'])
        perceptron = Perceptron()
        perceptron.fit(dataset.iloc[:, 0:2].values, dataset.iloc[:, 2].values)

        self.assertEqual(len(perceptron.weights_), 3)

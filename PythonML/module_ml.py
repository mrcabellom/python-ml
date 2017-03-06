from azureml import Workspace
from local_settings import AZURE_ML_SETIINGS
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from algoritms.perceptron import Perceptron
from graph.graph_plot import plot_sequence, plot_contour, plot_scatters

def read_dataset(name):

    workspace = Workspace(
        workspace_id=AZURE_ML_SETIINGS['WORKSPACE_ID'],
        authorization_token=AZURE_ML_SETIINGS['TOKEN_ID'],
        endpoint=AZURE_ML_SETIINGS['END_POINT'])
    dataset = workspace.datasets[name]
    frame = dataset.to_dataframe()
    return frame

def get_first_rows(dataframe):

    dataframe_selected = dataframe[0:100]
    return dataframe_selected

def save_plot_dataframe(dataframe):

    x = dataframe.iloc[0:100, [0, 2]].values
    scatters = []
    scatters.extend(
        [{'x': x[:50, 0],
          'y': x[:50, 1],
          'color': 'red',
          'marker': 'o',
          'label': 'setosa'
         },
         {'x': x[50:100, 0],
          'y': x[50:100, 1],
          'color': 'blue',
          'marker': 'x',
          'label': 'setosa'
         }])
    plot_scatters(scatters, title='Iris', labelx='sepal length', labely='petal length')

def encode_label(dataframe):

    label_encoder = LabelEncoder()
    dataframe = dataframe.assign(encodedclass=label_encoder.fit_transform(dataframe['class']))
    return  dataframe

def select_columns(dataframe):

    selected_columns = dataframe.drop('class', 1)
    return selected_columns


def split_test_train(dataframe):

    train, test = train_test_split(dataframe, test_size=0.5, random_state=0)
    dataset = train.append(test, ignore_index=True)
    return dataset

def train_perceptron(dataframe):

    perceptron = Perceptron()
    x = dataframe.iloc[0:100, [0, 2]].values
    y = dataframe.iloc[0:100, 4].values
    perceptron.fit(x, y)
    plot_sequence(perceptron.errors_, title='Mismatch', labelx='Epochs', labely='Errors')
    return perceptron

def test_perceptron(dataframe, perceptron):

    steps_h = 0.02
    x = dataframe.iloc[50:100, [0, 2]].values
    y = dataframe.iloc[50:100, 4].values
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    x_tuple, y_tuple = np.meshgrid(
        np.arange(x_min, x_max, steps_h),
        np.arange(y_min, y_max, steps_h))

    z = perceptron.predict(np.c_[x_tuple.ravel(), y_tuple.ravel()])
    plot_contour(x_tuple, y_tuple, z, title='DecisionBoundary', x=x, y=y)

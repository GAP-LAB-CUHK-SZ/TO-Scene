import pandas as pd
import numpy as np
import argparse
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm



def k_means(args):
  iris = datasets.load_iris()


  x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
  y = pd.DataFrame(iris.target, columns=['Target'])

  plt.figure(figsize=(12,3))
  colors = np.array(['red', 'green', 'blue'])
  iris_targets_legend = np.array(iris.target_names)
  red_patch = mpatches.Patch(color='red', label='Setosa')
  green_patch = mpatches.Patch(color='green', label='Versicolor')
  blue_patch = mpatches.Patch(color='blue', label='Virginica')


  plt.subplot(1, 2, 1)
  plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']])
  plt.title('Sepal Length vs Sepal Width')
  plt.legend(handles=[red_patch, green_patch, blue_patch])

  plt.subplot(1,2,2)
  plt.scatter(x['Petal Length'], x['Petal Width'], c= colors[y['Target']])
  plt.title('Petal Length vs Petal Width')
  plt.legend(handles=[red_patch, green_patch, blue_patch])

  plt.savefig("data_dist.png")



  iris_k_mean_model = KMeans(n_clusters=3)
  iris_k_mean_model.fit(x)

  plt.figure(figsize=(12,3))

  colors = np.array(['red', 'green', 'blue'])

  predictedY = np.choose(iris_k_mean_model.labels_, [1, 0, 2]).astype(np.int64)

  plt.subplot(1, 2, 1)
  plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']])
  plt.title('Before classification')
  plt.legend(handles=[red_patch, green_patch, blue_patch])

  plt.subplot(1, 2, 2)
  plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY])
  plt.title("Model's classification")
  plt.legend(handles=[red_patch, green_patch, blue_patch])

  plt.savefig("result.png")



  sm.accuracy_score(predictedY, y['Target'])

  
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', type=int , default=3, help='number of clusters')
    args = parser.parse_args()
    print(args)
    k_means(args)
  


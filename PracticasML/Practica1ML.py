import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np 
import sys
import pickle

class Validacion_set:
  def __init__(self,x_e,y_e):
    self.x_e = x_e
    self.y_e = y_e
    
class Prueba_set:
  def __init__(self,x_p,y_p):
    self.x_p = x_p
    self.y_p = y_p
    
class Data_set:
  def __init__(self,Validacion_set,Prueba_set):
    self.Validacion_set = Validacion_set
    self.Prueba_set = Prueba_set
    
def generador(NombreArchivo):
  pd.options.display.max_colwidth = 200
  Pr1Dataframe = pd.read_csv(NombreArchivo,sep =',',engine= 'python')
  x=Pr1Dataframe.drop('thal',axis= 1 ).values
  y=Pr1Dataframe['thal'].values

  x_e, x_p, y_e, y_p = train_test_split(x,y,test_size=0.4,shuffle=False)
  
  print("Dataset",Pr1Dataframe)
  print("Corpus")
  print("\n",*x)
  print("-------")
  print("Etiquetas")
  print("\n",*y)
  print("-----")
  print("Conjunto de prueba")
  print("\n x_prueba \n ",*x_p)
  print( "\n y_prueba \n",*y_p)
  print("-----")
  print("Conjunto de entrenamiento")
  print("\n x_entrenamiento",*x_e)
  print("\n y_entrenamiento",*y_e)

  MiPrueba = Prueba_set(x_p ,y_p)
  MiEntrenamiento = Validacion_set(x_e,y_e)

  MiDataset = Data_set(MiEntrenamiento,MiPrueba)
  return MiDataset

if __name__=='__main__':
  MiDataset = generador("C:/Users/Angel/Downloads/heart.csv")
  np.savetxt("x_p.csv", MiDataset.Prueba_set.x_p ,delimiter=",", fmt="%d",header="x")
  np.savetxt("y_p.csv", MiDataset.Prueba_set.y_p ,delimiter=",", fmt="%d",header="y")
  np.savetxt("x_e.csv", MiDataset.Validacion_set.x_e ,delimiter=",", fmt="%d",header="x")
  np.savetxt("y_e.csv", MiDataset.Validacion_set.y_e ,delimiter=",", fmt="%d",header="y")
  
  
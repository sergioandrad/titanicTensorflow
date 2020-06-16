#Importing the libs we're going to use:
import pandas            as pd
import keras             as ks
import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import os
from numpy.random          import seed
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import tensorflow as tf
import random     as python_random

#Pasta de trabalho:
os.getcwd()
os.chdir("C:/Users/sergi/Desktop/kaggle/titanic")

#Loading the dataset:
train        = pd.read_csv('train.csv')
response     = train.iloc[:,1]
train['Sex'] = pd.get_dummies(train['Sex'])
train        = train.join(pd.get_dummies(train['Embarked']))
#train        = train.join(pd.get_dummies(pd.qcut(train['Fare'], q = [0,.6,.8,.9,1], labels = ['Ultra Poor', 'Poor','Rich', 'SuperRich'])))
train['Ticket'][train['Ticket'].str.split(' ', expand=True).iloc[:,0].str.isdigit()] = 'SIMPLES'
train['Ticket'][train['Ticket'].str.contains('A/')]    = 'A'
train['Ticket'][train['Ticket'].str.contains('A.')]    = 'A'
train['Ticket'][train['Ticket'].str.contains('A4')]    = 'A'
train['Ticket'][train['Ticket'].str.contains('C.A.')]  = 'CA'
train['Ticket'][train['Ticket'].str.contains('CA')]    = 'CA'
train['Ticket'][train['Ticket'].str.contains('SOTON')] = 'SOTON'
train['Ticket'][train['Ticket'].str.contains('STON')]  = 'SOTON'
train['SOTON']    = pd.get_dummies(train['Ticket'].str.split(' ', expand=True).iloc[:,0].str.replace('.','').str.replace('/',''))['SOTON']
train             = train.drop(columns = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
impute            = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(train.values) 
covariates        = impute.transform(train.values)
covariatesStd     = scale(covariates)
pca = PCA()
pca.fit(covariatesStd)
principalComponents = pd.DataFrame(pca.transform(covariatesStd))
train = pd.DataFrame(covariates)
train['ScorePCA']   = principalComponents.iloc[:,0]

#Executamos as mesmas transformações no conjunto de teste que realizamos no conjunto de treino:
responseTest   = pd.read_csv('gender_submission.csv')
test           = pd.read_csv('test.csv')
test['Sex']    = pd.get_dummies(test['Sex'])
test           = test.join(pd.get_dummies(test['Embarked']))
#test           = test.join(pd.get_dummies(pd.qcut(test['Fare'], q = [0,.6,.8,.9,1], labels = ['Ultra Poor', 'Poor','Rich', 'SuperRich'])))
test['Ticket'][test['Ticket'].str.split(' ', expand=True).iloc[:,0].str.isdigit()] = 'SIMPLES'
test['Ticket'][test['Ticket'].str.contains('A/')]    = 'A'
test['Ticket'][test['Ticket'].str.contains('A.')]    = 'A'
test['Ticket'][test['Ticket'].str.contains('A4')]    = 'A'
test['Ticket'][test['Ticket'].str.contains('C.A.')]  = 'CA'
test['Ticket'][test['Ticket'].str.contains('CA')]    = 'CA'
test['Ticket'][test['Ticket'].str.contains('SOTON')] = 'SOTON'
test['Ticket'][test['Ticket'].str.contains('STON')]  = 'SOTON'
test['SOTON']           = pd.get_dummies(test['Ticket'].str.split(' ', expand=True).iloc[:,0].str.replace('.','').str.replace('/',''))['SOTON']
test                    = test.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
covariatesTest          = impute.transform(test.values)
covariatesTestStd       = scale(covariatesTest)
test                    = pd.DataFrame(covariatesTest)
principalComponentsTest = pd.DataFrame(pca.transform(covariatesTestStd))
test['ScorePCA']        = principalComponentsTest.iloc[:,0]


#Criando, ajustando e medindo a performance do modelo através da sequential API:
def KerasModel(seed_num, l1, l2, dropout, lr, f_loss, data_x, data_y, adam_par1, adam_par2, epoch_n, optimMethod, layer1, layer2 = None, layer3 = None, momentumSGD = None):
   global model 
   global historico_perda
   seed(seed_num)
   model     = ks.Sequential()
   model.add(ks.layers.Dense(units = data_x.shape[1], activation = "relu", input_dim = data_x.shape[1], kernel_initializer = "uniform"))
   model.add(ks.layers.Dense(layer1, activation = "relu", kernel_initializer = "uniform", kernel_regularizer = ks.regularizers.l1_l2(l1, l2)))
   model.add(ks.layers.Dropout(dropout))
   if layer2 != None:
       model.add(ks.layers.Dense(layer2, activation = "relu", kernel_initializer = "uniform", kernel_regularizer= ks.regularizers.l1_l2(l1, l2)))
       model.add(ks.layers.Dropout(dropout))
   if layer3 != None:    
       model.add(ks.layers.Dense(layer3, activation = "relu", kernel_initializer = "uniform", kernel_regularizer= ks.regularizers.l1_l2(l1, l2)))
       model.add(ks.layers.Dropout(dropout))
   model.add(ks.layers.Dense(1, activation = 'sigmoid')) 
   if optimMethod == 'adam':
       optimMethod = ks.optimizers.Adam(learning_rate = lr, beta_1 = adam_par1, beta_2 = adam_par2, amsgrad = False)
       if optimMethod == 'sgd':
           optimMethod = ks.optimizers.Adam(learning_rate = lr, momentum = momentumSGD, nesterov = True)
           if optimMethod == 'Nadam':
               optimMethod = ks.optimizers.Nadam(learning_rate = lr, beta_1 = adam_par1, beta_2 = adam_par2, amsgrad = False)
   earlyStopping = ks.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
   model.compile(optimizer = optimMethod, loss = f_loss, metrics=['accuracy'])
   historico_perda = model.fit(x = data_x, y = data_y, epochs = epoch_n, validation_split = 0.3, callbacks = [earlyStopping])   

def plotarMedidas(historico):
    plt.subplot(121)
    plt.title('Perda')
    plt.plot(historico.history['loss'],     label='treino')
    plt.plot(historico.history['val_loss'], label='teste')
    plt.legend()
    plt.subplot(122)
    plt.title('Acurácia')
    plt.plot(historico.history['accuracy'],     label='treino')
    plt.plot(historico.history['val_accuracy'], label='teste')
    plt.show()

l1           = 0.0001
l2           = 0
layer1       = 30
layer2       = 30
layer3       = None
dropout      = .3
lr           = .0015
adam_par1    = .9
adam_par2    = .99
f_loss       = 'binary_crossentropy'
epoch_n      = 100
data_x       = train
data_y       = response
optimMethod  = 'adam'
momentumSGD1 = 0
seed_num     = 2019
KerasModel(seed_num, l1, l2, dropout, lr, f_loss, data_x, data_y, adam_par1, adam_par2, epoch_n, optimMethod, layer1, layer2, layer3)

#Performance do Modelo:
plotarMedidas(historico_perda)
model.evaluate(test, responseTest.iloc[:,1])

#Ajustando seeds para garantir reprodutibilidade dos resultados:
python_random.seed(2019)
np.random.seed(2019)
tf.random.set_seed(2019)

#Modelo com PCA:
pca.explained_variance_ratio_
data_x      = principalComponents.iloc[:,[0,1,2,3,4,5]]
KerasModel(seed_num, l1, l2, dropout, lr, f_loss, data_x, data_y, adam_par1, adam_par2, epoch_n, optimMethod, layer1, layer2, layer3)
plotarMedidas(historico_perda)
model.evaluate(principalComponentsTest.iloc[:,[0,1,2,3,4,5]], responseTest.iloc[:,1])

##############################################################################################################################################################################
## Outras coisas testadas:
# Testando grid-search nas penalizacoes das perdas:
l1Lista          = [1, .1, .01, .001, .0001, .00001]
l2Lista          = [1, .1, .01, .001, .0001, .00001]
outputGridsearch = []
for i in l1Lista:
    for j in l2Lista:
        KerasModel(i, j, dropout, lr, inputs_n , f_loss, data_x, data_y, adam_par1, adam_par2, epoch_n, optimMethod)
        outputGridsearch.append([f"L1: {i}, L2: {j}", model.evaluate(covariatesTest, responseTest.iloc[:,1])])

## Estudando os nomes
train['Name'].str.split(' ', expand = True).iloc[:,0]
train[train['Sex'] == 'female']['Name'].str.split('(', expand = True).iloc[:,1].str.replace(')','').str.split(' ', expand = True).iloc[:,0:3]
train[train['Sex'] == 'male']['Name'].str.split(' ', expand = True)[0].str.replace(',','')
#test        = test.join(pd.get_dummies(train['Name'].str.split(' ', expand = True).iloc[:,0].str.replace(',','')))
#train       = train.join(pd.get_dummies(train['Name'].str.split(' ', expand = True).iloc[:,0].str.replace(',','')))

## Graficos Fare:
train['Fare'].describe()
train['Fare'].plot(kind='hist')
kwargs = {'cumulative': True}
sns.distplot(train['Fare'], hist_kws=kwargs, kde_kws=kwargs)
sns.distplot(train['Fare'], rug=True,)
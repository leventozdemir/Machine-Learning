# Fish species predictor
<img src="https://s3.ap-southeast-1.amazonaws.com/images.asianage.com/images/aa-Cover-h6hghrab3kclj3g171n66g9u43-20171011195123.Medi.jpeg">
## 1- Importing the liberires and the data
      import pandas as pd
      import numpy as np 
      import matplotlib.pyplot as plt
      %matplotlib inline
      import seaborn as sns
      
      data_path = 'Fish.csv'
      data = pd.read_csv(data_path)
      
## 2- Exploring the data

      print(data.head())
<img width="498" alt="1" src="https://user-images.githubusercontent.com/51120437/126552461-3ce72d39-9de5-4476-9aeb-4bdb115a3a53.png">

      print(data.Species.unique())
['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

#### We have 7 features at all 
#### One of the features is Species which we want to use it the as output we have 6 different classes

**['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']**

      print(data.isnull().sum())
<img width="121" alt="2" src="https://user-images.githubusercontent.com/51120437/126552599-1ea1e389-e488-4349-9426-1abc096db069.png">

      print(data.describe().transpose())
<img width="624" alt="3" src="https://user-images.githubusercontent.com/51120437/126552669-8c0357bb-45e6-4899-9c5a-2704f9c52926.png">

### Weight have a big STD for this we will normalize it later 
## 3- Data Preprocessing
#### 3.1- Normalizing the data: first lets drop categorical feature

      Y = data['Species']
      data = data.drop(['Species'], axis=1)
      names= data.columns
      
      from sklearn.preprocessing import Normalizer
      
      norms= Normalizer().fit(data)
      data_norms=norms.transform(data)
      data_norms= np.asarray(data_norms)
      
      nomred_data = pd.DataFrame(data_norms)
      nomred_data.columns =names
      print(nomred_data)
      
<img width="514" alt="4" src="https://user-images.githubusercontent.com/51120437/126552889-4d31e1dc-ec0d-44ed-ac58-e66e4fdcd0e5.png">
#### 3.2- Categorical to numerical

      Y= Y.replace({"Bream": 0, "Roach": 1,
                    "Whitefish":2, "Parkki":3, 
                    "Perch":4, "Pike":5, "Smelt":6})
      print(Y.unique())       
  [0 1 2 3 4 5 6]

#### 3.3- Compain all the data togather for visualization
      data_prepared = pd.concat([nomred_data,Y], axis=1)
      print(data_prepared)
<img width="518" alt="5" src="https://user-images.githubusercontent.com/51120437/126553078-de65292f-fb74-44f7-8223-d169a2b64311.png">


#### Now the data is normalized and numerical so its ready to make some visualization
<img src="https://media.makeameme.org/created/wohooo-0392028a2b.jpg">


## 4- Data Visualization

      plt.figure(figsize=(5,7))
      sns.boxplot(x= data_prepared['Species'], y= data_prepared['Width'])
      sns.swarmplot(x= data_prepared['Species'], y= data_prepared['Width'])

<img width="342" alt="6" src="https://user-images.githubusercontent.com/51120437/126553342-c6dd596f-ff3f-41ab-b230-9778e8bbde50.png">

      plt.figure(figsize=(5,7))
      sns.boxplot(x= data_prepared['Species'], y= data_prepared['Height'])
      sns.swarmplot(x= data_prepared['Species'], y= data_prepared['Height'])
<img width="354" alt="7" src="https://user-images.githubusercontent.com/51120437/126553419-c5e8d2c5-4a8e-42f0-b095-3974ead68d7d.png">
#### By lookint to the plotting we can see that there is 3 Outliers Let's drop them!!!


<img width="561" alt="Outliers" src="https://user-images.githubusercontent.com/51120437/126553460-43735131-b424-422f-a072-45622ce7ffcc.png">
#### 4.2- Droping outliers

#### 4.2.1- Get the index of the outlier and drop
      data_4_cat= data_prepared.loc[data_prepared.Species==4]
      print(data_4_cat['Width'].idxmax())
      indx_to_drop = data_4_cat['Width'].idxmax()
      data_prepared =data_prepared.drop([indx_to_drop], axis=0)

      data_1_cat= data_prepared.loc[data_prepared.Species==1]
      print(data_1_cat['Width'].idxmax())
      indx_to_drop = data_1_cat['Width'].idxmax()
      data_prepared =data_prepared.drop([indx_to_drop], axis=0)

      data_1_cat= data_prepared.loc[data_prepared.Species==1]
      print(data_1_cat['Width'].idxmax())
      indx_to_drop = data_1_cat['Width'].idxmax()
      data_prepared =data_prepared.drop([indx_to_drop], axis=0)

      plt.figure(figsize=(10,10))
      sns.boxplot(x= data_prepared['Species'], y= data_prepared['Width'])
      sns.swarmplot(x= data_prepared['Species'], y= data_prepared['Width'])
<img width="358" alt="8" src="https://user-images.githubusercontent.com/51120437/126553614-9afc556a-a089-47b7-a32a-df2626258cb7.png">
#### We don't have Outlierssssss  🥳🥳🥳🥳🥳
<img src="https://i2.wp.com/novocom.top/image/bWVkalwaHkWEuZ2lwaHkuY29t/media/3o6ZsXoYhtUlGEyIRa/giphy.gif">

## 5- Preparing the model and the data for training


      from sklearn.model_selection import GridSearchCV
      from sklearn.svm import SVC
      from sklearn.model_selection import train_test_split

      SVC = SVC()

      parameters = {'C': [10,100,1000], 'gamma': [4,5,6,7],
                    'kernel': ['rbf', 'poly', 'sigmoid']}
      grid_search = GridSearchCV(SVC, parameters, n_jobs=20, cv=6, refit=True,verbose=10)

      Y = data_prepared['Species']
      X= data_prepared.drop(['Species'], axis=1)

      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=40, test_size=0.3)

      grid_search.fit(X_train, Y_train)

      print("Best parameters for SVC Clasiifier",grid_search.best_estimator_)
**Best parameters for SVC Clasiifier SVC(C=1000, gamma=5, kernel='poly')**

## 6- Evaluate the model

      from sklearn.metrics import accuracy_score
      preds = grid_search.predict(X_test)
      score= accuracy_score(Y_test, preds)
      print("_Accuracy = %",score*100)
**_Accuracy = % 97.87234042553192**


# The End....
<img src="https://giffiles.alphacoders.com/105/105882.gif">

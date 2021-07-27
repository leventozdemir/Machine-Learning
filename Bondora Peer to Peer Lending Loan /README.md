# Bondora Peer to Peer Lending Loan
![intro_wme](https://user-images.githubusercontent.com/51120437/127114935-7f66d452-9141-4853-aef6-a8115bdff022.jpg)
## 1- Calling the libraries and the data:

      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      %matplotlib inline
      import seaborn as sns
      path= "../input/bondora-peer-to-peer-lending-loan-data/LoanData_Bondora.csv"
      data= pd.read_csv(path, low_memory=False)
      
## 2- Data Exploration and Preprocessing:
      data.shape
      # Output: (179235, 112)
      for i in range(112):
          print(data.columns[i])
          
The features we have :
  - ReportAsOfEOD: EOD stands for End of the Day. It points to the top of a trading day in financial markets, the purpose in time when the trading ceases for the day. It's also referred to as end of business, close of business and close of play.

  - LoanId

  - LoanNumber

  - ListedOnUTC: Date when the loan application appeared on Primary Market

  - BiddingStartedOn: A bid is an offer made by an investor, trader, or dealer in an effort to buy an asset or to compete for a contract.

  - BidsPortfolioManager.
  - BidsApi.

  - BidsManual.

  - UserName.

  - NewCreditCustomer.

  - LoanApplicationStartedDate.

  - LoanDate.

  - ContractEndDate.

  - FirstPaymentDate.

  - MaturityDate_Original: The time between the issue and the maturity date for a particular bond.

  - MaturityDate_Last.

  - ApplicationSignedHour.

  - ApplicationSignedWeekday.

  - VerificationType.

  - LanguageCode.

  - Age.

  - DateOfBirth.

  - Gender.

  - Country.

  - AppliedAmount.

  - Amount.

  - Interest.

  - LoanDuration.

  - MonthlyPayment.

  - County.

  - City.

  - UseOfLoan.

  - Education.

  - MaritalStatus.

  - NrOfDependants.

  - etc...




### 2.1 Working with Null features:
      for i in data.columns:
          print(i,data[i].isnull().sum())
          
alot of features have more than 90% null Let's drop them:

      list_of_50_percent_null = [ ]
      for i in data.columns:
          if data[i].isnull().sum() >= (90*179235)/100:
              list_of_50_percent_null.append(i)
              
      print(list_of_50_percent_null)
      #OutPut:['DateOfBirth','County','City','EmploymentPosition','EL_V0','Rating_V0','EL_V1','Rating_V1','CreditScoreEsEquifaxRisk']
      data = data.drop(list_of_50_percent_null, axis=1)
 
## 3- Exploration for Object and Bool data types:
      cat_data= data.select_dtypes('object')
      data = data.drop(cat_data.columns, axis=1)

      bool_data= data.select_dtypes('bool')
      data = data.drop(bool_data.columns, axis=1)
      
Fill the object data that have less then 90% null:
      cat_data =cat_data.fillna("unknown")
      date_type= cat_data["BiddingStartedOn"].astype('datetime64[ns]')
      cat_data= cat_data.drop(['BiddingStartedOn'],axis=1)
      
      
      features_cat_data= list(cat_data.columns)
      features_cat_data_viz= ["Country","EmploymentDurationCurrentEmployer","Rating","WorseLateCategory",
                         "CreditScoreEsMicroL"]
      for i in features_cat_data_viz:
          cat_data[i].value_counts().plot(kind='pie', figsize=(6,6), autopct="%1.2f%%")
          plt.title(i)
          plt.show()
![1](https://user-images.githubusercontent.com/51120437/127117432-6673ae23-c527-4ba1-a650-c42a538f5aab.png)
![1](https://user-images.githubusercontent.com/51120437/127117513-300b307e-b434-42e5-92fd-373d18be5881.png)
![1](https://user-images.githubusercontent.com/51120437/127117544-0772eae2-73b2-4cb2-a99d-24ec5e0f0ff7.png)
![1](https://user-images.githubusercontent.com/51120437/127117569-8182a788-6ec9-4077-95a9-e7bb16f158c3.png)
![1](https://user-images.githubusercontent.com/51120437/127117588-9965c56b-84af-4f91-ac73-e9c414fcf72d.png)

      features_bool_data= list(bool_data.columns)
      features_bool_data_viz= ["NewCreditCustomer", "ActiveScheduleFirstPaymentReached","Restructured"]
      for i in features_bool_data_viz:
          bool_data[i].value_counts().plot(kind='bar', figsize=(5,5))
          plt.title(i)
          plt.show()
          

![1](https://user-images.githubusercontent.com/51120437/127117683-7f5b8d53-d1e4-4ef5-9061-f4f705719287.png)
![1](https://user-images.githubusercontent.com/51120437/127117708-b8e7fc32-c710-4f07-9f2f-fdaca6e3902d.png)
![1](https://user-images.githubusercontent.com/51120437/127117738-fcecfe60-ef6f-4e2e-89e3-f63d32e0316b.png)


## 4- Preprocessing for Object and Bool data types:

      cat_array= np.array(cat_data).reshape(-1)
      bool_array= np.array(bool_data).reshape(-1)

      from sklearn.preprocessing import LabelEncoder


      encoder_1= LabelEncoder()
      encoder_2= LabelEncoder()
      cat_enc= encoder_1.fit_transform(cat_array)
      bool_enc= encoder_2.fit_transform(bool_array)
      cat_enc= pd.DataFrame(cat_enc.reshape(179235,int(cat_enc.shape[0]/179235)))
      bool_enc= pd.DataFrame(bool_enc.reshape(179235,int(bool_enc.shape[0]/179235)))
      cat_enc.columns= features_cat_data
      bool_enc.columns= features_bool_data
      object_data = pd.concat([cat_enc,bool_enc], axis=1)
      
## 5- Exploration for Numerical data:

Fill the object data that have less then 90% null

<img width="584" alt="2" src="https://user-images.githubusercontent.com/51120437/127118766-683273db-b34f-43a7-ab1a-1166c1b96559.png">
      from sklearn.impute import SimpleImputer
      
      names_num = data.columns
      imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
      imp_median.fit(np.array(data))
      imp_data= imp_median.transform(data)
      data= pd.DataFrame(imp_data)
      data.columns= names_num

      print(data.describe().transpose())

<img width="582" alt="Screen Shot 2021-07-27 at 11 04 33" src="https://user-images.githubusercontent.com/51120437/127118958-44714247-6b45-4827-9b80-379b6b64241b.png">


## 6- Preprocessing for Numerical data:

We will use Z-Score for normalization


      def z_score_normalizer(X):
          m = X.shape[0]
          n = 1
          for i in range(n):
              X = (X - X.mean(axis=0))/X.std(axis=0)
          return X
          
      data = z_score_normalizer(data)

## 7- Target Selection:

      Y= object_data['Status']
      object_data = object_data.drop(['Status'],axis=1)
      Y=encoder_1.inverse_transform(Y)
      Y= pd.DataFrame(Y, columns=['Status'])
      print(Y.Status.unique())
      #OutPut: ['Late', 'Repaid', 'Current']
      
we have 3 type in status
- Late
- Repaid
- Current 

      Y= Y.loc[Y.Status!='Current']
      Y= Y.replace(['Late','Repaid'],[0,1])

<img width="118" alt="Screen Shot 2021-07-27 at 11 07 34" src="https://user-images.githubusercontent.com/51120437/127119409-6ed483de-a419-4377-a470-08e39abed212.png">




## 8- Compine all the data togather:

      all_data = pd.concat([object_data,data,Y], axis=1)
      all_data= all_data.dropna() #drop if still null
      
## 9- Machine Learning Model:

      from sklearn.model_selection import GridSearchCV
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.model_selection import train_test_split
      from sklearn import metrics


      Y= all_data["Status"]
      X= all_data.drop(["Status"],axis= 1)

      X_train, X2, Y_train, Y2= train_test_split(X, Y, test_size=0.2, random_state=24)
      X_val, X_test, Y_val, Y_test= train_test_split(X2, Y2, test_size=0.5, random_state=4)

      model = DecisionTreeClassifier(random_state=4)

      parameters = {"criterion" : ["gini", "entropy"], 
                    'max_depth': [20,21],
                    'min_samples_split': [50,51]}


      grid_search = GridSearchCV(model, parameters, n_jobs=50,verbose=100,cv=2, refit='best_params_')
      grid_search.fit(X_train, Y_train)

      print("Best parameters for DT Clasiifier",grid_search.best_params_)

Best parameters for DT Clasiifier {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 50}


      preds = grid_search.predict(X_val)
      fpr, tpr, thresholds = metrics.roc_curve(Y_val, preds)
      print("AUC Score :",metrics.auc(fpr, tpr))

AUC Score : 0.9999267721148214

          
      metrics.plot_roc_curve(grid_search, X_val, Y_val)
      
![1](https://user-images.githubusercontent.com/51120437/127119781-94bab9e0-f562-49b0-b8d3-2d4c12ea1b15.png)


      preds2 = grid_search.predict(X_test)
      roc_score2 = metrics.roc_auc_score(Y_test, preds2)
      print("Roc Score :\n",roc_score2)         

Roc Score : 0.9999268898961837

          
Roc Score on the Validation set = 99.99%
Roc Score on the Test set = 99.99%
 

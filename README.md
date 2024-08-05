## Titanic Dataset Description
 The dataset includes data about passengers on the Titanic and whether they survived the disaster.

• PassengerId: An integer unique identifier for each passenger.                                
• Survived: An integer indicating whether the passenger survived (1) or not (0).              
• Pclass: An integer representing the passenger's class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).                                                                                        
• Name: A string containing the passenger's full name.                                         
• Sex: A string indicating the passenger's gender (male or female).                            
• Age: A float representing the passenger's age in years.            
• SibSp: An integer indicating the number of siblings or spouses the passenger had aboard the Titanic.                                                                                       
• Parch: An integer representing the number of parents or children the passenger had aboard the Titanic.                                                                                
• Ticket: A string containing the ticket number.                                              
• Fare: A float representing the fare the passenger paid for the ticket.                      
• Cabin: A string indicating the cabin number. Many values are missing (NaN).                
• Embarked: A string indicating the port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).                                                                                 

#### Data Dictionary
• PassengerId: Unique ID for each passenger.                                                  
• Survived: Survival (0 = No; 1 = Yes).                                                       
• Pclass: Ticket class (1 = 1st; 2 = 2nd; 3 = 3rd).                                           
• Name: Name of the passenger.                                                       
• Sex: Sex of the passenger.                   
• Age: Age of the passenger in years.                                   
• SibSp: Number of siblings/spouses aboard the Titanic.                            
• Parch: Number of parents/children aboard the Titanic.                     
• Ticket: Ticket number.                                            
• Fare: Fare paid by the passenger.                                        
• Cabin: Cabin number.                                       
• Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).              

### Libraires
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

### Load and Check Data
```
train_df = pd.read_csv(r'C:\...\train.csv')
```
Examining the First 5 Columns of the Dataset
```
train_df.head()
```
Then checked the columns
```
train_df.columns
```
The data was described
```
train_df.describe()
```
## Visualizing the Titanic Dataset
We made a function for bar plot visualizing the data.
```
def bar_plot(variable):
    var = train_df[variable]
    varValue= var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
```
For categorical variables, we made bar plots.
```
categry1 = ["Survived","Sex","Pclass","Embarked","SiSp","Parch"]
for c in categry1:
    bar_plot(c)
```
##### Interpretation of the Survived Bar Plot
A majority of passengers did not survive the disaster. The number of passengers who did not survive is notably higher than the number of passengers who survived.
This plot indicates that survival was less common than non-survival among the Titanic passengers.
##### Interpretation of the Gender Bar Plot
There were more male passengers on the Titanic compared to female passengers. The number of male passengers is almost twice the number of female passengers.
This plot highlights the gender distribution among the passengers, indicating that a larger proportion of the passengers were male.
##### Interpretation of the Class Distribution (Pclass) Bar Plot
A large majority of Titanic passengers traveled in 3rd class, suggesting that the Titanic was a popular choice for lower-income passengers.
The relatively lower numbers of 1st and 2nd class passengers indicate that these classes were more expensive and thus chosen by fewer passengers.
##### Interpretation of the Embarkation Port Distribution (Embarked) Bar Plot
A large majority of Titanic passengers boarded from Southampton (S). This suggests that Southampton was the primary embarkation point for the Titanic.
Fewer passengers boarded from Cherbourg (C) and Queenstown (Q), indicating that these ports were smaller stops with less passenger traffic.
##### Interpretation of the Number of Siblings/Spouses Distribution (SibSp) Bar Plot
Most passengers boarded the Titanic alone or without siblings/spouses.
There is a notable group of passengers traveling with one sibling or spouse, but this number is still relatively small.
The number of passengers traveling with 2 or more siblings/spouses is very low, indicating that large family groups were rare.
##### Interpretation of the Number of Parents/Children Distribution (Parch) Bar Plot
Most passengers boarded the Titanic alone or without parents/children.
There is a notable group of passengers traveling with one parent or child, but this number is still relatively small.
The number of passengers traveling with 2 or more parents/children is very low, indicating that large family groups were rare.

#### We made a histogram plot function for visualizing the data.
```
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} disturbution with hist".format(variable))
    plt.show()
```
For numerical variables, we made histogram plots.
```
numericVar= ["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
```
#### Fare Distribution
Most passengers paid low fares, which suggests they were likely in the third class. The right-skewness of the fare distribution indicates that there were some wealthier passengers who could afford much higher fares, likely corresponding to first-class accommodations.
#### Age Distribution
The Titanic carried a wide range of passengers of all ages, but young adults (ages 20-30) were the most common age group. This distribution provides insights into the demographics of the passengers, showing that the ship had a significant number of young people, but also included many children and older adults.

### Analyzing Relationships Between Categorical and Numerical Variables
```
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived")
```
First-class passengers had the highest survival rate, while third-class passengers had the lowest. This indicates that passengers traveling in higher classes had a better chance of survival during the Titanic disaster.
```
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived")
```
Female passengers had a significantly higher survival rate compared to male passengers. This suggests that the "women and children first" policy was in effect during the rescue operations.
```
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived")
```
Passengers with 1 or 2 siblings/spouses had the highest survival rates, suggesting that having a small family group may have increased chances of survival. However, having more than 2 siblings/spouses aboard significantly decreased the survival rate, possibly due to the difficulty in managing and ensuring the safety of a larger group during the chaos.
```
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived")
```
Passengers with 1, 2, or 3 parents/children had higher survival rates, indicating that family units of this size had a better chance of staying together and being rescued. Conversely, passengers with 4 or more parents/children had a significantly lower survival rate, which might be due to the increased difficulty of ensuring the safety of larger family groups during the disaster.

### Outlier Detection and Removal
We created a function to detect and remove outliers:
```
def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile (25th percentile)
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile (75th percentile)
        Q3 = np.percentile(df[c], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
     
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers =  list(i for i, v in outlier_indices.items() if v>2)
    return multiple_outliers
    
    return outlier_indicesOutlier detection
```
We detected the outliers
```
train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])]
```
We merged the datasets
```
train_df_len=len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
```
## Find Missing Value
We found the missing values
```
train_df.columns[train_df.isnull().any()]
```
We visualized the columns with missing values
```
train_df.boxplot(column="Fare",by="Embarked")
plt.show()
```
We filled the missing values
```
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
```
We checked for missing values
```
train_df[train_df["Fare"].isnull()]
```
We filled the missing fare values
```
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
```
```
train_df[train_df["Fare"].isnull()]
```
### Visualization
#### Correlation Between Sibsp,Parch,Age,Fare,Survived¶
```
list = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list].corr(), annot = True, fmt = ".2f")
plt.show()
```
Fare feature seems to have correlation with survived feature (0.26).

#### SibSp/Survived
```
s = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
s.set_ylabels("Survived Probability")
plt.show()
```
Passengers with a high number of siblings/spouses (SibSp) had lower chances of survival. If the SibSp count is 0, 1, or 2, passengers had a higher likelihood of surviving. We can consider creating a new feature to describe these categories.

#### Parch/Survived
```
p = sns.catplot(x = "Parch", y = "Survived", kind = "bar", data = train_df)
p.set_ylabels("Survived Probability")
plt.show()
```
SibSp and Parch can be utilized for new feature extraction with a threshold of 3. Small families tend to have a higher chance of survival. There is a noticeable standard deviation in the survival rate of passengers with Parch = 3.

#### Age/Survived
```
a = sns.FacetGrid(train_df, col = "Survived")
a.map(sns.distplot, "Age", bins = 25)
plt.show()
```
Passengers aged 10 and below have a high survival rate. The oldest passenger, aged 80, survived. A significant number of 20-year-old passengers did not survive. Most passengers fall within the 15-35 age range. It is advisable to use the age feature during training and to utilize the age distribution to fill in missing age values.

#### Embarked/Sex/Fare/Survived
```
t = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")
t.map(sns.barplot, "Sex", "Fare")
t.add_legend()
plt.show()
```
Passsengers who pay higher fare have better survival. Fare can be used as categorical for training.

#### Fill Missing Age Values
```
train_df[train_df["Age"].isnull()]
```
```
sns.catplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()
```
```
sns.catplot(x = "Sex", y = "Age", hue = "Pclass",data = train_df, kind = "box")
plt.show()
```
First-class passengers tend to be older than second-class passengers, who in turn are generally older than third-class passengers.

```
sns.catplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.catplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()
```
#### Correlation with Sex
```
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()
```
Age is not correlated with sex but it is correlated with parch, sibsp and pclass.
```
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med
```
This code fills missing age values by predicting them based on similar passengers' characteristics. It first identifies the indices of missing ages. For each missing age, it calculates the median age of passengers with the same SibSp, Parch, and Pclass. If a median age is found, it uses this value; otherwise, it fills the missing age with the overall median age. This approach ensures a more accurate and complete dataset for analysis and modeling.
```
train_df[train_df["Age"].isnull()]
```

## Drop Passenger ID and Cabin 
```
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
train_df.columns
```
We drop these columns because these are useless.

## Modeling
```
test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)
```
We split our data 
```
train = train_df[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))
```
### Hyperparameter Tuning -- Grid Search -- Cross Validation
Decision Tree                                      
SVM                      
Random Forest                                     
KNN                       
Logistic Regression                         
```
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]
``` 
```
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
```
As you can see the highest score is KNN's 0.7226229508196721.

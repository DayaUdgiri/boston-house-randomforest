# necessary Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestRegressor

# loading the data
boston = load_boston()

# fetching features(X) and label(Y)
x = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# scaling the dataset
scaler = StandardScaler()
X_sc = scaler.fit_transform(x)

# splitting the dataset into Train and Test sets
X_sc_train, X_sc_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.33, random_state=42)

# fitting the model
rf_model=RandomForestRegressor()
rf_model.fit(X_sc_train,y_train)

print("Training score= ", rf_model.score(X_sc_train,y_train))
print("Testing SCore = ", rf_model.score(X_sc_test,y_test))

# saving the model to the local file system
filename='RF_model.pickle'
pickle.dump(rf_model,open(filename,'wb'))

# prediction using the saved model
loaded_model=pickle.load(open(filename,'rb'))

for i in range(0,500,50):
    prediction_output = loaded_model.predict(scaler.transform([x.iloc[i]]))
    print("\n *** For input", np.array(x.iloc[i],dtype=str),"\n prediction_output = ",\
          prediction_output, "\n Expected output= ", y[i])



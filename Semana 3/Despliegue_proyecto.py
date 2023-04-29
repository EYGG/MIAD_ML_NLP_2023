#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_price(year, mileage, state, make, model):

    reg = joblib.load(os.path.dirname(__file__) + '/xgb.pkl') 

    x_test = pd.DataFrame()
    x_test['year'] = year
    x_test['mileage'] = mileage
    x_test['state'] = state
    x_test['make'] = make
    x_test['model'] = model

    # Convertir en categorias
    x_test['state'] = x_test['state'].astype('category')
    x_test['make'] = x_test['make'].astype('category')
    x_test['model'] = x_test['model'].astype('category')

    # Crear variables dummies
    x_test = pd.get_dummies(x_test, columns=['state', 'make', 'model'], drop_first=True)

    # Predecir
    y_pred = reg.predict(x_test)

    return y_pred


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingrese los valores del vehículo')
        
    else:
            
            year = sys.argv[1]
            mileage = sys.argv[2]
            state = sys.argv[3]
            make = sys.argv[4]
            model = sys.argv[5]
    
            price = predict_price(year, mileage, state, make, model)
            
            print('Price: ', price)
        

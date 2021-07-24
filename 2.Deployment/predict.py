import pandas as pd
from make_prediction import *


def predict(testdata):
    if(testdata is None or len(testdata)>500):
        return None
    
    predictions = predict_proba(testdata)
    predictions = pd.DataFrame({
            'TransactionID':testdata['TransactionID'].astype(int),
            'Predictions':np.round(predictions, 5)
    })
    
    del testdata
    
    return predictions
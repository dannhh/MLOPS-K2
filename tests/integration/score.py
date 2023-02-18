import joblib
import json
import numpy as np
import pandas as pd
import os
from datetime import datetime

DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
MODEL_ALGORITHM = 'light gradient boosting'
MODEL_NAME = 'credit-score-model.pkl'

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), MODEL_NAME)
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

def run(data):
    '''
    data: is a input json that requests to API endpoint.
          data include: "data" list of float that request to API.
          for example:
            {'data': [[0,1,8,1,0,0,1,0,0,0,0,0,0,0,12,1,0,0,0.5,0.3,0.610327781,7,1,-1,0,-1,1,1,1,2,1,65,1,0.316227766,0.669556409,0.352136337,3.464101615,0.1,0.8,0.6,1,1,6,3,6,2,9,1,1,1,12,0,1,1,0,0,1],[4,2,5,1,0,0,0,0,1,0,0,0,0,0,5,1,0,0,0.9,0.5,0.771362431,4,1,-1,0,0,11,1,1,0,1,103,1,0.316227766,0.60632002,0.358329457,2.828427125,0.4,0.5,0.4,3,3,8,4,10,2,7,2,0,3,10,0,0,1,1,0,1]]}
    '''
    try:
        start_time = datetime.strftime(datetime.now(), DATE_FORMAT)
        test = json.loads(data)
        input_data = test['data']
        # np_data = np.array(input_data)
        pd_data = pd.DataFrame(input_data, columns=['AGE','GENDER','EDUCATIONAL','PAYROLL_FLAG','EMAIL_FLAG','PHONE_FLAG','ADDRESS','CUST_INCOME','MOB','DPD_ABOVE1_L3M','DPD_ABOVE1_L6M','DPD_ABOVE30_L3M','DPD_ABOVE30_L6M','DPD_ABOVE60_L3M','DPD_ABOVE60_L6M','DPD_ABOVE90_L3M','DPD_ABOVE90_L6M','CONS_DPD_ABOVE1_L3M','CONS_DPD_ABOVE1_L5M','CONS_DPD_ABOVE30_L3M','CONS_DPD_ABOVE30_L5M','CONS_DPD_ABOVE60_L3M','CONS_DPD_ABOVE60_L5M','CONS_DPD_ABOVE90_L3M','CONS_DPD_ABOVE90_L5M','DPD_INC_L3M','DPD_DEC_L3M','DPD_INC_L6M','DPD_DEC_L6M','CONS_DPD_INC_L3M','CONS_DPD_DEC_L3M','CONS_DPD_INC_L6M','CONS_DPD_DEC_L6M','UTIL_L3M','UTIL_L6M','UTIL_PREV_OVER_L3M','UTIL_L3M_OVER_L6M','UTIL_L3M_OVER_P3M','UTIL_L6M_OVER_P6M','UTIL_INC_L3M','UTIL_INC_L6M','CONS_UTIL_INC_L3M','CONS_UTIL_INC_L6M','UTIL_DEC_L3M','UTIL_DEC_L6M','CONS_UTIL_DEC_L3M','CONS_UTIL_DEC_L6M','UTIL_ABOVE50_L3M','UTIL_ABOVE75_L3M','UTIL_ABOVE90_L3M','CONS_UTIL_ABOVE50_L3M','CONS_UTIL_ABOVE75_L3M','CONS_UTIL_ABOVE90_L3M','UTIL_ABOVE50_L6M','UTIL_ABOVE75_L6M','UTIL_ABOVE90_L6M','CONS_UTIL_ABOVE50_L5M','CONS_UTIL_ABOVE75_L5M','CONS_UTIL_ABOVE90_L5M','BAL_PREV_OVER_L3M','BAL_L3M_OVER_L6M','BAL_L3M_OVER_P3M','BAL_L6M_OVER_P6M','BAL_INC_L3M','BAL_INC_L6M','CONS_BAL_INC_L3M','CONS_BAL_INC_L6M','BAL_DEC_L3M','BAL_DEC_L6M','CONS_BAL_DEC_L3M','CONS_BAL_DEC_L6M','CRLMT_OVER_CASA_L3M','CRLMT_OVER_CASA_L6M','BAL_OVER_CASA_L3M','BAL_OVER_CASA_L6M','BAL_OVER_CREDITCASA_L3M','BAL_OVER_CREDITCASA_L6M','CASHADV_OVER_BAL_L3M','CASHADV_OVER_BAL_L6M','PURCHASE_OVER_BAL_L3M','PURCHASE_OVER_BAL_L6M','PURCHASE_ABOVE0_L3M','PURCHASE_ABOVE0_L6M','CASH_ADV_ABOVE0_L3M','CASH_ADV_ABOVE0_L6M','LATE_PYMT_ABOVE0_L3M','LATE_PYMT_ABOVE0_L6M','INTEREST_ABOVE0_L3M','INTEREST_ABOVE0_L6M','CONS_PURCHASE_ABOVE0_L3M','CONS_PURCHASE_ABOVE0_L5M','CONS_CASH_ADV_ABOVE0_L3M','CONS_CASH_ADV_ABOVE0_L5M','CONS_LATE_PYMT_ABOVE0_L3M','CONS_LATE_PYMT_ABOVE0_L5M','CONS_INTEREST_ABOVE0_L3M','CONS_INTEREST_ABOVE0_L5M','REVOLVED_L3M','REVOLVED_L6M','CONS_REVOLVED_L3M','CONS_REVOLVED_L5M','PAID_L3M','PAID_L6M'])
        # score = model.predict(np_data).tolist()
        score = model.predict(pd_data).tolist()
        end_time = datetime.strftime(datetime.now(), DATE_FORMAT)
        # You can return any JSON-serializable object.
        result = {
                'model_algorithm': MODEL_ALGORITHM,
                'local_feature_importances': None,
                'prediction_score': score,
                'start_time': start_time,
                'end_time': end_time
            }
        return json.dumps(result)

    except Exception as e:
        error = str(e)
        return error
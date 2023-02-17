# import all libraries required
import json, sys, math
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from azureml.core import Workspace, Model, Dataset, Webservice
from azureml.core.authentication import AzureCliAuthentication

def read_config(config_file):
    # read config file
    with open(config_file, encoding='utf-8-sig') as f:
        return json.load(f)['aci']['configuration']

def read_model_config(config_file):
    # read model config file
    with open(config_file, encoding='utf-8-sig') as f:
        return json.load(f)['model']['configuration']

def inference_aci_webservice(aci_service):
    sample_input = json.dumps({'data': [[1,0,48,"Female",5,"N","Y","Y","north",3,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.025,0,0,0,0.263157895,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.015724472,0,0,0,0,0,1,0,0,0.417059453,0.391962386,0,7.98E-06,0,3.22E-05,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,3,6]]})
    print('----------------aci scoring uri: \n', aci_service.scoring_uri)
    print('----------------aci service name: \n', aci_service.name)
    print('----------------request sample: \n', sample_input)
    output = aci_service.run(sample_input)
    print('----------------output: \n', output)
    return output

def test_aci(aci_config):
    # Authenticate using CLI
    cli_auth = AzureCliAuthentication()
    # Get workspace
    ws = Workspace.from_config(auth=cli_auth)

    # Get ACI
    aci_name = aci_config['name']
    aci_service = Webservice(ws, aci_name)  
    output = inference_aci_webservice(aci_service)

    # return test successful
    return output

def main():
    args = sys.argv[1:]

    if len(args) >= 2 and args[0] == '-config':
        # execute load config file and model registration
        aci_config = read_config(args[1])
        # test model on ACI
        pass_test = test_aci(aci_config)
        if pass_test:
            print('ACI pass testing')
        else:
            sys.exit('ACI failed testing')
    else:
        print('Usage: -config <config file name> [-outfolder <Azure ML config folder>]')
    
if __name__ == '__main__':
    main()
# CFL

## Install CFL (Python 3.10.13):

> ```pip install </path/to/cfl-tmlcn/> -e . ```

## Input data
> Save your data per client as a .pickle file
>> Example:
>>> client 0: ./path_to_your_data/0.pickle

>>> client 1: ./path_to_your_data/1.pickle

> Provide the training data as a nested dict as: data = {'dataset': {'X': np.ndarray, 'Y': np.ndarray}, 'id': str}
>> dataset itself is a dict with keys 'X' and 'Y'

>> 'id' is a string corresponding to the name of the client

## A template for use of CFL in FMNIST classification:

> There is a dataclass named ExpConfig which needs to be modified

> There is a dataclass named LearningConfig which contains the learning configs

### Notes:

> CFL generally prefers to have number of epochs per round set to a larger value 

> You can make the final predictions either from the global model or the local model at the clinet side. 
>> This is done by setting "ExpConfig.make_predictions_using_local_model" to True

### To run the script:

> ``` python -m cfl.exp_scripts.exp_fmnist ```
>> Note: For now data from FMNIST is used as stored in ./cfl/data/fg_traces/

>> Please don't forget to replace with your own data.

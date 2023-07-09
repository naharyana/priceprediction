import pandas as pd
import util as utils
from sklearn.model_selection import train_test_split

def read_data(return_file = False):
    # Read data
    data = pd.read_csv(config["dataset_raw_path"], 
                       sep=',')

    # Print data
    print('data shape   :', data.shape)

    # Dump data
    utils.pickle_dump(data, config['dataset_path'])

    # Return data
    if return_file:
        return data

def check_data(input_data, params):
    # check range of company laptop data
    assert set(input_data.Company).issubset(set(params["range_company"])), "an error occurs in Company range."
    
def split_input_output(config_data, return_file=False):
    # Read data
    data = utils.pickle_load(config_data['dataset_path'])

    # Split input & output
    y = data[config_data['label']]
    X = data.drop([config_data['label']], axis=1)

    # Print splitting
    print('Input shape  :', X.shape)
    print('Output shape :', y.shape)
    print('Input NAN    :')
    
    print(X.isnull().sum())
    print('Benchmark    :')
    print(y.mean())
    
    # Dump file
    utils.pickle_dump(X, config_data['input_set_path'])
    utils.pickle_dump(y, config_data['output_set_path'])

    if return_file:
        return X, y
        
def split_train_test(config_data, return_file=True):
    # Load data
    X = utils.pickle_load(config_data['input_set_path'])
    y = utils.pickle_load(config_data['output_set_path'])

    # Split test & rest (train & valid)
    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y,
                                            test_size = config_data['test_size'],
                                            random_state = config_data['seed']
                                        )
    
    # Split train & valid
    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size = config_data['test_size'],
                                            random_state = config_data['seed']
                                        )

    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump file
    utils.pickle_dump(X_train, config_data['train_set_path'][0])
    utils.pickle_dump(y_train, config_data['train_set_path'][1])
    utils.pickle_dump(X_valid, config_data['valid_set_path'][0])
    utils.pickle_dump(y_valid, config_data['valid_set_path'][1])
    utils.pickle_dump(X_test, config_data['test_set_path'][0])
    utils.pickle_dump(y_test, config_data['test_set_path'][1])

    if return_file:
        return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == '__main__':
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    dataset = read_data(return_file = True)
    check_data(dataset, config)
    
    split_input_output(config)
    split_train_test(config)
    
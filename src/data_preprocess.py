import util as utils
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict):
    # Load every set of data
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set

def check_touchsreen(data):
    if "Touchscreen" in data:
        return 1
    else:
        return 0
    
def check_ips(data):
    if "IPS" in data:
        return 1
    else:
        return 0

def get_cpu_name(data):
    # get the first three words in cpu name 
    
    first_three_words = data.split()[0:3]
    output = " ".join(first_three_words)
    
    return output

def fetch_processor(text):
    if (text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3'):
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
    
def clean_ram(data):
    # get the numeric of RAM
    replacing_gb = data.str.replace("GB","")
    numeric_form = replacing_gb.astype("int")
    
    return numeric_form

def brand_gpu(data):
    # get the brand of gpu
    brand = data.split()[0]
    
    return brand

def cat_os(data):
    if data == 'Windows 10' or data == 'Windows 7' or data == 'Windows 10 S':
        return 'Windows'
    elif data == 'macOS' or data == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
    
def clean_weight(data):
    # get the numeric of RAM
    replacing_kg = data.str.replace("kg","")
    numeric_form_weight = replacing_kg.astype("float")
    return numeric_form_weight

def get_xresolution(data):
    temp = data.str.split("x", n = 1, expand = True)
    x_res = temp[0].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
    return x_res

def get_yresolution(data):
    temp = data.str.split("x", n = 1, expand = True)
    y_res = temp[1].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
    return y_res

def fit_standardize(data, config_data, return_file=True):
    """Dump ohe object"""
    standardizer = StandardScaler()

    # Fit standardizer
    standardizer.fit(data)

    # Dump standardizer
    utils.pickle_dump(standardizer, config_data['standardizer_path'])
    
    if return_file:
        return standardizer
    
def transform_standardize(data, standardizer):
    """Function to standardize data"""
    data_standard = pd.DataFrame(standardizer.transform(data))
    data_standard.columns = data.columns
    data_standard.index = data.index 
    
    return data_standard
    
def fit_ohe(data, config_data, return_file=True):
    """Dump ohe object"""
    ohe = OneHotEncoder(handle_unknown = "ignore")

    # Fit One Hot Encoder
    ohe.fit(data)

    # Dump One Hot Encoder
    utils.pickle_dump(ohe, config_data['ohe_path'])
    
    if return_file:
        return ohe
    
def transform_ohe(data, ohe):
    """Function to standardize data"""
    data_ohe = pd.DataFrame(ohe.transform(data).toarray())
    data_ohe.columns = ohe.get_feature_names_out(data.columns)
    data_ohe.index = data.index
    return data_ohe

def feature_engineering(data):
    """Function to do feature engineering"""
    
    # Add new columns
    data["Touchscreen"] = data["ScreenResolution"].apply(check_touchsreen)
    data["IPS"] = data["ScreenResolution"].apply(check_ips)
    data["X_res"] = get_xresolution(data["ScreenResolution"])
    data["Y_res"] = get_yresolution(data["ScreenResolution"])
    
    # Clearning columns
    data['Cpu'] = data['Cpu'].apply(get_cpu_name)
    data['Cpu Name'] = data['Cpu'].apply(fetch_processor)
    data['Gpu'] = data['Gpu'].apply(brand_gpu)
    data["Ram"] = clean_ram(data["Ram"])
    data['OpSys'] = data['OpSys'].apply(cat_os)
    data["Weight"] = clean_weight(data["Weight"])
    
    return data

def generate_preprocessor(train_data, config_data, return_file=True):
    """Function to generate preprocessor"""
    # Load data
    
    # Generate preprocessor: standardizer
    standardizer = fit_standardize(train_data[config_data["numerical_columns"]], config_data)
    
    # Generate preprocessor: onehotencoden
    ohe = fit_ohe(train_data[config_data["cat_columns"]], config_data)

    # Dump file
    preprocessor = {'standardizer': standardizer,
                    'ohe': ohe}
    utils.pickle_dump(preprocessor, config_data['preprocessor_path'])
    utils.pickle_dump(preprocessor, config_data['ohe_path'])
    
    if return_file:
        return preprocessor
    
def preprocess_data(config_data, type_data = 'train' , return_file=True):
    """Function to preprocess train data"""
    # Load data
    X = utils.pickle_load(config_data[f'{type_data}_set_path'][0])
    y = utils.pickle_load(config_data[f'{type_data}_set_path'][1])
    
    # Feature Engineering
    X_fe = feature_engineering(X)
    
    # Load preprocessor
    if type_data == "train":
        preprocessor = generate_preprocessor(X_fe, config_data)
    else:
        preprocessor = utils.pickle_load(config_data['preprocessor_path'])
    
    
    # Standardization
    standardizer = preprocessor['standardizer']
    X_clean_numerical = transform_standardize(X_fe[config_data['numerical_columns']], standardizer)
    
    # One Hot Encoder
    ohe = preprocessor['ohe']
    X_clean_categorical = transform_ohe(X_fe[config_data['cat_columns']], ohe)
    
    # Combine numerical and categorical columns
    X_clean = pd.concat([X_clean_numerical, X_clean_categorical], axis = 1)
    
    y_clean = y

    # Print shape
    print("X clean shape:", X_clean.shape)
    print("y clean shape:", y_clean.shape)

    # Dump file
    utils.pickle_dump(X_clean, config_data[f'{type_data}_clean_path'][0])
    utils.pickle_dump(y_clean, config_data[f'{type_data}_clean_path'][1])

    if return_file:
        return X_clean, y_clean   

if __name__ == '__main__':
    # 1. Load configuration file
    config_data = utils.load_config()

    # 3. Preprocess Data
    preprocess_data(config_data, type_data = 'train')
    preprocess_data(config_data, type_data = 'valid')
    preprocess_data(config_data, type_data = 'test')
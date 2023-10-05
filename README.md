# 🌊 SST Prediction App 🌡️

## Introduction

Welcome to the SST Prediction App! 🎉

This application enables users to predict Sea Surface Temperatures (SST) based on historical data. With a specialized ConvLSTM model, the app processes 5 days of SST data to predict the SST for the 6th day.

### Model Overview: 
🔧 Trained by [Boris](https://github.com/boryasbora)
Layer (type) | Output Shape | Param #
---|---|---
ConvLSTM2D | (None, 401, 451, 32) | 38,144
BatchNormalization | (None, 401, 451, 32) | 128
Dropout | (None, 401, 451, 32) | 0
Conv2D | (None, 401, 451, 64) | 18,496
BatchNormalization_1 | (None, 401, 451, 64) | 256
Dropout_1 | (None, 401, 451, 64) | 0
Conv2D_1 | (None, 401, 451, 32) | 18,464
BatchNormalization_2 | (None, 401, 451, 32) | 128
Dropout_2 | (None, 401, 451, 32) | 0
Conv2D_2 | (None, 401, 451, 1) | 289

**Total params:** 75,905  
**Trainable params:** 75,649  
**Non-trainable params:** 256

The data used hails from the [MUR-JPL-L4-GLOB-v4.1](https://search.earthdata.nasa.gov/search/granules?p=C1996881146-POCLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&q=MUR-JPL-L4-GLOB-v4.1&fi=MODIS&as[instrument][0]=MODIS&tl=1686459841!3!!&zoom=0) dataset on earthdata.nasa.gov.

## Setup Steps 🚀:

1. Clone the repo:
```bash
git clone https://github.com/NaNa7Miiii/upwelling_prediction.git
```
2. Navigate to the directory：
```bash
cd upwelling_prediction
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Launch the app:
```bash
streamlit run app.py
```

5. Please comment out this chunk of code which aims to create a .netrc file to access NASA's data, but if you want to deploy the repo in streamlit cloud please don't do anything on that:
```
NETRC_PATH = os.path.expanduser("~/.netrc")

def create_netrc(machine, login, password, path=NETRC_PATH):
    netrc_content = f"""
    machine {machine}
    login {login}
    password {password}
    """

    with open(path, "w") as file:
        file.write(netrc_content)

    os.chmod(path, 0o600)

secrets = st.secrets["earthdata_test"]
create_netrc(secrets["machine"], secrets["login"], secrets["password"])
```
After all that, you are ready to go!

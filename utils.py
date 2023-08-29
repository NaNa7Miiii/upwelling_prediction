from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import requests
import xarray as xr
import os


def load_your_model(model_path):
    model = load_model(model_path)
    return model


def process_nc(ds):
    sst_values = ds.values
    sst_values = np.where(sst_values == 9.96921e+36, np.nan, sst_values)
    return sst_values


def preprocess_vis_input_data(day_data):
    day_data = np.squeeze(day_data)
    mean_val = np.nanmean(day_data)
    processed_data = day_data - mean_val
    # Replace NaNs with 0.0
    processed_data = np.where(np.isnan(processed_data), 0.0, processed_data)
    return processed_data


def postprocess_prediction(prediction, input_data_day):
    land_mask = create_land_mask(input_data_day)
    prediction_squeezed = np.squeeze(prediction)
    prediction_squeezed[land_mask] = np.nan
    mean_val = np.nanmean(input_data_day)
    prediction_post = np.where(np.isnan(prediction_squeezed), np.nan, prediction_squeezed + mean_val)
    return prediction_post


def create_land_mask(data):
    data_copy = data.copy()
    data_copy[data_copy == 0] = np.nan
    return np.isnan(data_copy)


def plot_sst(sst_values, ds, title_text):
    fig, ax = plt.subplots()
    lons = ds['lon'].values
    lats = ds['lat'].values
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    img = ax.imshow(sst_values, origin="lower", cmap="viridis", extent=extent)
    ax.set_xticks(np.linspace(lons.min(), lons.max(), 6))
    ax.set_yticks(np.linspace(lats.min(), lats.max(), 6))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title_text)
    plt.colorbar(img)
    st.pyplot(fig)


def downsample_data(data, step=10):
    return data[::step, ::step]


def plot_sst_interactive(sst_values, ds, title_text):
    sst_values_downsampled = downsample_data(sst_values)
    lons = ds['lon'].values[::10]
    lats = ds['lat'].values[::10]
    fig = go.Figure(data=go.Heatmap(
        z=sst_values_downsampled,
        x=lons,
        y=lats,
        colorscale="Viridis",
        colorbar={"title": "SST Values"}
    ))
    fig.update_layout(title=title_text, xaxis_title="Longitude", yaxis_title="Latitude")
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig)


def plot_sst_interactive_init(sst_values, ds, title_text):
    lons = ds['lon'].values
    lats = ds['lat'].values
    fig = go.Figure(data=go.Heatmap(
        z=sst_values,
        x=lons,
        y=lats,
        colorscale="Viridis",
        colorbar={"title": "SST Values"}
    ))
    fig.update_layout(title=title_text, xaxis_title="Longitude", yaxis_title="Latitude")
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig)


def fetch_data_url(start_time, end_time, lats, lons):
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
    response = requests.get(cmr_url,
                            params={
                                'provider': 'POCLOUD',
                                'short_name': 'MUR-JPL-L4-GLOB-v4.1',
                                'temporal': f'{start_time},{end_time}',
                                'bounding_box': f'{lons.start},{lats.start},{lons.stop},{lats.stop}',
                                'page_size': 2000,
                            }
                            )
    granules = response.json()['feed']['entry']
    return [next((link['href'] for link in granule['links'] if link['rel'].endswith('/data#')), None) for granule in
            granules]


def fetch_and_load_sst(url, lats, lons):
    filename = url.split('/')[-1]
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download from {url}. Status code: {response.status_code}")
            return None
        with open(filename, 'wb') as f:
            f.write(response.content)
        ds = xr.open_dataset(filename)
        sst_data = ds['analysed_sst'].sel(lat=slice(lats.start, lats.stop), lon=slice(lons.start, lons.stop))
        os.remove(filename)
        return sst_data
    except Exception as e:
        print(f"Error fetching and loading SST data from {url}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return None


def fetch_mur_data(start_time, end_time, lats, lons):
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
    response = requests.get(cmr_url,
                            params={
                                'provider': 'POCLOUD',
                                'short_name': 'MUR-JPL-L4-GLOB-v4.1',
                                'temporal': f'{start_time},{end_time}',
                                'bounding_box': f'{lons.start},{lats.start},{lons.stop},{lats.stop}',
                                'page_size': 2000,
                            }
                            )
    granules = response.json()['feed']['entry']
    urls = [next((link['href'] for link in granule['links'] if link['rel'].endswith('/data#')), None) for granule in
            granules]
    sst_data_list = [fetch_and_load_sst(url, lats, lons) for url in urls]
    st.write("help")
    st.write(sst_data_list)
    # Then, concatenate all the sst_data elements from the list
    resulting_dataset = xr.concat(sst_data_list, dim="time")
    return resulting_dataset


INTRO_TEXT = """
    Welcome to the SST Prediction App!
    
    This application allows users to predict Sea Surface Temperatures (SST) based on historical data.
    Using a specialized ConvLSTM model, this app ingests 5 days of SST data and predicts the SST for the 6th day.
    
    ### Model Overview:
    
    Layer (type) | Output Shape | Param #
    ---|---|---
    ConvLSTM2D | (None, 401, 451, 32) | 38144
    BatchNormalization | (None, 401, 451, 32) | 128
    Dropout | (None, 401, 451, 32) | 0
    Conv2D | (None, 401, 451, 64) | 18496
    BatchNormalization_1 | (None, 401, 451, 64) | 256
    Dropout_1 | (None, 401, 451, 64) | 0
    Conv2D_1 | (None, 401, 451, 32) | 18464
    BatchNormalization_2 | (None, 401, 451, 32) | 128
    Dropout_2 | (None, 401, 451, 32) | 0
    Conv2D_2 | (None, 401, 451, 1) | 289
    
    **Total params:** 75,905
    **Trainable params:** 75,649
    **Non-trainable params:** 256
    
    The data utilized originates from the [MUR-JPL-L4-GLOB-v4.1](https://search.earthdata.nasa.gov/search/granules?p=C1996881146-POCLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&q=MUR-JPL-L4-GLOB-v4.1&fi=MODIS&as[instrument][0]=MODIS&tl=1686459841!3!!&zoom=0) dataset on earthdata.nasa.gov.
    
    **Please fetch the data for 5 days at the left sidebar first.**
    """

SIDEBAR_DESC = """
    ### 1. Select Dates
    
    The data for this app is sourced from the MUR-JPL-L4-GLOB-v4.1 dataset available at [earthdata.nasa.gov](https://search.earthdata.nasa.gov/search/granules?p=C1996881146-POCLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&q=MUR-JPL-L4-GLOB-v4.1&fi=MODIS&as[instrument][0]=MODIS&tl=1686459841!3!!&zoom=0).
    
    By default, the 'end date' is set as yesterday. The range between the start and end date encompasses 5 days of data.
    """

DIFFMAP_DESC = """
    ## Difference Map Explanation

    The difference map visualizes the discrepancy between predicted and true values. It's calculated by subtracting the true values from the predicted values. 

    - **Blue regions**: The predicted value is lower than the true value.
    - **Red regions**: The predicted value is higher than the true value.
    - **White regions**: The predicted value is very close or equal to the true value.

    Difference maps are valuable in machine learning and data analytics to visualize prediction errors, thereby guiding researchers and developers towards potential areas of improvement.
    """


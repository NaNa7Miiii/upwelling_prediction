import datetime
import streamlit as st

from utils import *

# secrets = st.secrets["earthdata_test"]
# machine = secrets["machine"]
# login = secrets["login"]
# password = secrets["password"]

def create_netrc(machine, login, password):
    netrc_content = f"""
    machine {machine}
    login {login}
    password {password}
    """
    
    with open(".netrc", "w") as file:
        file.write(netrc_content)
        
    os.chmod(".netrc", 0o600) 

with open(".netrc", "r") as f:
    content = f.read()
st.text(content)

secrets = st.secrets["earthdata_test"]
create_netrc(secrets["machine"], secrets["login"], secrets["password"])

# Set up session state variables
session_vars = ['data_fetched', 'step', 'view', 'sst_data', 'ds_list']
default_values = [False, 0, 'main', [], []]
for var, default in zip(session_vars, default_values):
    st.session_state.setdefault(var, default)

# Load model
model = load_your_model('ConvLSTM_2002_2017.keras')

# App Title and Sidebar
st.title('🌊 SST Prediction App 🌊')
intro_placeholder = st.empty()

if not st.session_state.get("data_fetched", False):
    intro_placeholder.write(INTRO_TEXT)

st.sidebar.title("Options")
st.sidebar.write(SIDEBAR_DESC)
default_end_date = datetime.date.today() - datetime.timedelta(days=2)
default_start_date = default_end_date - datetime.timedelta(days=4)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", start_date + datetime.timedelta(days=4))
start_time = f"{start_date}T21:00:00Z"
end_time = f"{end_date}T20:59:59Z"

# Fetching Data
with st.sidebar.expander("Fetching Data"):
    if st.sidebar.button("Fetch Data"):
        intro_placeholder.empty()
        with st.spinner('Fetching Data...'):
            ds = fetch_mur_data(start_time, end_time, slice(-5, 35), slice(45, 90))
        if len(ds['time']) == 5:
            st.session_state.data_fetched = True
            st.session_state.sst_data = [process_nc(ds.isel(time=i)) for i in range(5)]
            st.session_state.ds_list = [ds.isel(time=i) for i in range(5)]
        else:
            st.warning("Please select a date range with exactly 5 days of available data!")

# Display Data
for idx, (sst_value, ds) in enumerate(zip(st.session_state.sst_data, st.session_state.ds_list)):
    if sst_value.all() and ds.all():
        plot_sst(sst_value, ds, f"Sea Surface Temperature for File #{idx + 1}")
        if st.button(f"See Details for File #{idx + 1}"):
            st.session_state.view = 'details'
            st.session_state.detail_idx = idx
            st.experimental_rerun()

if st.session_state.view == 'details':
    sst_value, ds = st.session_state.sst_data[st.session_state.detail_idx], st.session_state.ds_list[st.session_state.detail_idx]
    if sst_value.all() and ds.all():
        plot_sst_interactive(sst_value, ds, f"Sea Surface Temperature for File #{st.session_state.detail_idx + 1}")
        if st.button("Return from Details"):
            st.session_state.view = 'main'
            st.experimental_rerun()

# Prediction
prediction_container = st.empty()
if st.session_state.data_fetched:
    with st.spinner('Running Prediction...'):
        if st.button('🔍 Run Prediction 🔍'):
            preprocessed_data = [preprocess_vis_input_data(day) for day in st.session_state.sst_data]
            input_data = np.array(preprocessed_data)[:, ::10, ::10, np.newaxis]
            prediction = model.predict(input_data[np.newaxis, ...])[0]
            st.session_state.prediction_postprocessed = postprocess_prediction(prediction, st.session_state.sst_data[-1][::10, ::10])
            st.session_state.step = 1

# Display Predicted Data Details
if "prediction_postprocessed" in st.session_state:
    last_ds = st.session_state.ds_list[-1]
    plot_sst(st.session_state.prediction_postprocessed, last_ds, "Predicted Sea Surface Temperature")
    if st.button('See Details for Predicted Data'):
        st.session_state.view = 'predicted_details'
        st.experimental_rerun()

if st.session_state.view == 'predicted_details':
    last_ds = st.session_state.ds_list[-1]
    plot_sst_interactive_init(st.session_state.prediction_postprocessed, last_ds, "Detailed Predicted Sea Surface Temperature")
    if st.button("Return from Predicted Details"):
        st.session_state.view = 'main'
        st.experimental_rerun()

# True Data Section
st.session_state.setdefault("true_values", None)

if st.session_state.view == 'true_details' and st.session_state.true_values:
    last_ds = st.session_state.ds_list[-1]
    plot_sst_interactive(st.session_state.true_values_fullres, last_ds, "Detailed Real Image")
    if st.button("Return from True Details"):
        st.session_state.view = 'main'
        st.experimental_rerun()



if st.session_state.step == 1:
    date_input = st.date_input("Choose a date (The Default Day is The Following Day)", end_date + datetime.timedelta(days=1))
    if date_input:
        with st.spinner("Plotting the True Data..."):
            start_time, end_time = f"{date_input}T09:00:00.000000000", f"{date_input}T09:23:59.000000000"
            true_data = fetch_mur_data(start_time, end_time, slice(-5.0, 35.0), slice(45.0, 90.0))
            true_values = np.squeeze(process_nc(true_data))
            st.session_state.true_values, st.session_state.true_values_fullres = true_values[::10, ::10], true_values
            plot_sst(true_values, true_data, "Real Image")
            st.session_state.step = 2

# Evaluation Metrics
if st.session_state.step == 2:
    metrics = st.multiselect("Choose metrics for evaluation", ["MSE", "MAE"])

    prediction_postprocessed = st.session_state.prediction_postprocessed
    true_values = st.session_state.true_values
    results = {}

    if st.button("See Difference between Predicted and True Data"):
        st.session_state.view = 'difference_map'
        st.experimental_rerun()

    if "MSE" in metrics:
        results["MSE: Prediction Vs True Data"] = np.nanmean((prediction_postprocessed - true_values) ** 2)
        results["MSE: Last Input Day Vs True Data "] = np.nanmean((st.session_state.sst_data[-1][::10, ::10] - true_values) ** 2)

    if "MAE" in metrics:
        results["MAE: Prediction Vs True Data"] = np.nanmean(np.abs(prediction_postprocessed - true_values))
        results["MAE: Last Input Day Vs True Data "] = np.nanmean(np.abs(st.session_state.sst_data[-1][::10, ::10] - true_values))

    for key, val in results.items():
        st.write(f"{key}: {val:.4f}")

# Difference Map View
if st.session_state.view == 'difference_map':
    diff_map = prediction_postprocessed - true_values

    # Find the absolute maximum difference to ensure the color scale is centered at zero
    max_diff = np.nanmax(np.abs(diff_map))

    fig, ax = plt.subplots()
    im = ax.imshow(diff_map, cmap='bwr', interpolation='none', origin='lower', vmin=-max_diff, vmax=max_diff)
    ax.set_title("Difference Map: Predicted - True")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.write(DIFFMAP_DESC)

    if st.button("Return from Difference Map"):
        st.session_state.view = 'main'
        st.experimental_rerun()

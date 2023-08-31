import datetime
from utils import *

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

st.sidebar.title("🛠️ Control Panel 🛠️")
st.sidebar.write(SIDEBAR_DESC)
default_end_date = datetime.date.today() - datetime.timedelta(days=2)
default_start_date = default_end_date - datetime.timedelta(days=4)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", start_date + datetime.timedelta(days=4))
start_time = f"{start_date}T21:00:00Z"
end_time = f"{end_date}T20:59:59Z"

# Fetching Data
st.sidebar.header("📥 Fetch Data")
with st.sidebar.expander("Click to expand", expanded=True):
    if st.sidebar.button("🔄 Fetch Data"):
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
if st.session_state.view == 'main':
    for idx, (sst_value, ds) in enumerate(zip(st.session_state.sst_data, st.session_state.ds_list)):
        if sst_value.all() and ds.all():

            date_str = ds.time.values.astype('datetime64[D]').astype(str)
            with st.expander(f"📅 {date_str}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Show SST for {date_str}"):
                        plot_sst(sst_value, ds, f"Sea Surface Temperature for {date_str}")
                        if st.button(f"Return from Plotting for {date_str}", key=f"return_{idx}"):
                            st.session_state.view = 'main'
                            st.experimental_rerun()
                with col2:
                    if st.button(f"See Details for {date_str}", key=f"details_{idx}"):
                        st.session_state.view = 'details'
                        st.session_state.detail_idx = idx
                        st.experimental_rerun()

                with col3:
                    bytes_data = create_nc_download_link(ds, f"{date_str}.nc")
                    st.download_button(
                        label=f"Download {date_str}.nc",
                        data=bytes_data,
                        file_name=f"{date_str}.nc",
                        mime="application/netcdf"
                    )

if st.session_state.view == 'details':
    sst_value, ds = st.session_state.sst_data[st.session_state.detail_idx], st.session_state.ds_list[st.session_state.detail_idx]
    date_str = ds.time.values.astype('datetime64[D]').astype(str)
    if sst_value.all() and ds.all():
        plot_sst_interactive(sst_value, ds, f"Sea Surface Temperature Details for {date_str}")
        if st.button("Return from Details"):
            st.session_state.view = 'main'
            st.experimental_rerun()

# Prediction
if 'view' not in st.session_state:
    st.session_state.view = 'main'

if st.session_state.view == 'main':
    prediction_container = st.empty()
    if st.session_state.data_fetched:
        st.header("🔮 Prediction Section 🔮")
        with st.spinner('Waiting for user input...'):
            with st.expander("🔍 Single Day Prediction"):
                if st.button('Run Prediction For The Next Day'):
                    preprocessed_data = [preprocess_vis_input_data(day) for day in st.session_state.sst_data]
                    input_data = np.array(preprocessed_data)[:, ::10, ::10, np.newaxis]
                    prediction = model.predict(input_data[np.newaxis, ...])[0]
                    st.session_state.prediction_postprocessed = postprocess_prediction(prediction,
                                                                                       st.session_state.sst_data[-1][::10,
                                                                                   ::10])
                    st.session_state.step = 1

                # Display Predicted Data Details
                if "prediction_postprocessed" in st.session_state:
                    last_ds = st.session_state.ds_list[-1]
                    plot_sst(st.session_state.prediction_postprocessed, last_ds, "Predicted Sea Surface Temperature")

                    download_ds = create_dataset_from_values(st.session_state.prediction_postprocessed, last_ds)
                    bytes_data = create_nc_download_link(download_ds, "predicted_sst_data.nc")
                    st.download_button("Download Predicted SST Data", bytes_data, "predicted_sst_data.nc")

                    if st.button('See Details for Predicted Data'):
                        st.session_state.view = 'predicted_details'
                        st.experimental_rerun()

                # True Data Section
                st.session_state.setdefault("true_values", None)

                if st.session_state.step == 1:
                    date_input = st.date_input("Choose a date (The Default Day is The Following Day)",
                                               end_date + datetime.timedelta(days=1))
                    if date_input:
                        with st.spinner("Plotting the True Data..."):
                            start_time, end_time = f"{date_input}T09:00:00.000000000", f"{date_input}T09:23:59.000000000"
                            true_data = fetch_mur_data(start_time, end_time, slice(-5.0, 35.0), slice(45.0, 90.0))
                            true_values = np.squeeze(process_nc(true_data))
                            st.session_state.true_values, st.session_state.true_values_fullres = true_values[::10,
                                                                                                 ::10], true_values
                            plot_sst(true_values, true_data, "Real Image")
                            download_ds = create_dataset_from_values(st.session_state.true_values_fullres, true_data)
                            bytes_data = create_nc_download_link(download_ds, "true_sst_data.nc")
                            st.download_button("Download True SST Data", bytes_data, "true_sst_data.nc")

                            st.session_state.step = 2

                # Evaluation Metrics
                if st.session_state.step == 2:
                    st.header("📊 Evaluation Metrics 📊")
                    metrics = st.multiselect("Choose metrics for evaluation", ["MSE", "MAE"])
                    prediction_postprocessed = st.session_state.prediction_postprocessed
                    true_values = st.session_state.true_values
                    results = {}
                    if st.button("See Difference between Predicted and True Data"):
                        st.session_state.view = 'difference_map'
                        st.experimental_rerun()

                    if "MSE" in metrics:
                        results["MSE: Prediction Vs True Data"] = np.nanmean(
                                        (prediction_postprocessed - true_values) ** 2)
                        results["MSE: Last Input Day Vs True Data "] = np.nanmean(
                                        (st.session_state.sst_data[-1][::10, ::10] - true_values) ** 2)

                    if "MAE" in metrics:
                        results["MAE: Prediction Vs True Data"] = np.nanmean(
                                        np.abs(prediction_postprocessed - true_values))
                        results["MAE: Last Input Day Vs True Data "] = np.nanmean(
                                        np.abs(st.session_state.sst_data[-1][::10, ::10] - true_values))

                    for key, val in results.items():
                        st.write(f"{key}: {val:.4f}")

            with st.expander("🔍 Multiple Days Prediction"):
                days_to_predict = st.slider("Number of days to predict", 1, 30, 1)
                if st.button(f'Predict next {days_to_predict} days'):
                    preprocessed_data = [preprocess_vis_input_data(day) for day in
                                                     st.session_state.sst_data]
                    input_data = np.array(preprocessed_data)[:, ::10, ::10, np.newaxis]
                    predictions = sliding_window_prediction(model, input_data, days_to_predict)
                    st.session_state.predictions = predictions
                    st.session_state.prediction_end_date = end_date

                if 'predictions' in st.session_state:
                    if st.button('🔍 View Predicted SST for Selected Days 🔍'):
                        last_ds = st.session_state.ds_list[-1]
                        for i, prediction in enumerate(st.session_state.predictions[:days_to_predict]):
                            date = st.session_state.prediction_end_date + datetime.timedelta(days=i + 1)
                            prediction = postprocess_prediction(prediction, st.session_state.sst_data[-1][::10, ::10])
                            plot_sst(prediction, last_ds, f"Predicted Sea Surface Temperature for {date}")

                            download_ds = create_dataset_from_values(prediction, last_ds)
                            bytes_data = create_nc_download_link(download_ds, f"predicted_sst_data_{date}.nc")
                            st.download_button("Download Predicted SST Data", bytes_data, f"predicted_sst_data_{date}.nc")

                        if st.button("Return from Predicted Images"):
                            st.session_state.view = 'main'
                            st.experimental_rerun()

if st.session_state.view == 'predicted_details':
    last_ds = st.session_state.ds_list[-1]
    plot_sst_interactive_init(st.session_state.prediction_postprocessed, last_ds,
                                          "Detailed Predicted Sea Surface Temperature")
    if st.button("Return from Predicted Details"):
        st.session_state.view = 'main'
        st.experimental_rerun()

elif st.session_state.view == 'true_details' and st.session_state.true_values:
    last_ds = st.session_state.ds_list[-1]
    plot_sst_interactive(st.session_state.true_values_fullres, last_ds, "Detailed Real Image")
    if st.button("Return from True Details"):
        st.session_state.view = 'main'
        st.experimental_rerun()

# Difference Map View
elif st.session_state.view == 'difference_map':
    prediction_postprocessed = st.session_state.prediction_postprocessed
    true_values = st.session_state.true_values
    diff_map = prediction_postprocessed - true_values
    max_diff = np.nanmax(np.abs(diff_map))
    fig, ax = plt.subplots()
    im = ax.imshow(diff_map, cmap='bwr', interpolation='none', origin='lower', vmin=-max_diff,
                           vmax=max_diff)
    ax.set_title("Difference Map: Predicted - True")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.write(DIFFMAP_DESC)
    if st.button("Return from Difference Map"):
        st.session_state.view = 'main'
        st.experimental_rerun()
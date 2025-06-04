import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Pastikan TensorFlow menggunakan mode eager untuk debugging jika diperlukan
tf.config.run_functions_eagerly(True)

# Fungsi untuk membuat sequence data untuk TCN
def create_sequences(data, look_back):
    """
    Membuat sequence input (X) dan output (y) untuk model TCN.

    Args:
        data (np.array): Data deret waktu.
        look_back (int): Jumlah langkah waktu sebelumnya untuk digunakan sebagai input.

    Returns:
        tuple: Tuple berisi array numpy X (input) dan y (output).
               y akan dikembalikan dalam bentuk 2D (num_samples, 1).
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y).reshape(-1, 1) # Pastikan y selalu 2D (num_samples, 1)

# Fungsi untuk membangun model TCN
def build_tcn_model(input_shape, filters=64, kernel_size=2, dilations=[1, 2, 4], dropout_rate=0.2):
    """
    Membangun model Temporal Convolutional Network (TCN).

    Args:
        input_shape (tuple): Bentuk input data (look_back, 1).
        filters (int): Jumlah filter untuk setiap lapisan konvolusi.
        kernel_size (int): Ukuran kernel untuk lapisan konvolusi.
        dilations (list): Daftar faktor dilasi untuk lapisan TCN.
        dropout_rate (float): Tingkat dropout.

    Returns:
        tf.keras.Model: Model TCN yang telah dikompilasi.
    """
    model = Sequential()
    model.add(Input(shape=input_shape)) # Menambahkan Input layer secara eksplisit

    # Lapisan TCN (Conv1D dengan dilasi)
    for i, d in enumerate(dilations):
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         dilation_rate=d,
                         padding='causal', # Penting untuk TCN agar tidak "melihat" masa depan
                         activation='relu',
                         name=f'tcn_conv1d_{i+1}'))
        model.add(Dropout(dropout_rate))

    # Lapisan output
    model.add(Dense(1, name='output_dense')) # Output tunggal untuk prediksi nilai berikutnya

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Fungsi untuk menghitung EWMA dan batas kontrol
def calculate_ewma(residuals, lambda_param, L):
    """
    Menghitung statistik EWMA dan batas kontrol.

    Args:
        residuals (np.array): Residual dari model forecasting (1D array).
        lambda_param (float): Faktor smoothing EWMA (0 < lambda_param <= 1).
        L (float): Multiplier untuk batas kontrol (misalnya, 3 untuk 3-sigma).

    Returns:
        tuple: Tuple berisi array EWMA (Z_t), Upper Control Limit (UCL),
               Lower Control Limit (LCL), dan daftar titik out-of-control.
    """
    n = len(residuals)
    Z_t = np.zeros(n)
    sigma_residuals = np.std(residuals) # Estimasi standar deviasi residual

    # Inisialisasi Z_0 dengan rata-rata residual (atau 0 jika residualnya di sekitar 0)
    # Untuk EWMA, Z_0 sering diinisialisasi dengan rata-rata historis atau target proses (0 untuk residual)
    Z_t[0] = residuals[0] if n > 0 else 0 # Handle empty residuals case

    for i in range(1, n):
        Z_t[i] = lambda_param * residuals[i] + (1 - lambda_param) * Z_t[i-1]

    # Batas kontrol (UCL, LCL)
    # Variansi EWMA mendekati lambda_param / (2 - lambda_param) * sigma_residuals^2
    # Untuk n besar, ini adalah estimasi yang baik.
    UCL = L * sigma_residuals * np.sqrt(lambda_param / (2 - lambda_param)) if (2 - lambda_param) != 0 else np.inf
    LCL = -UCL # Asumsi target proses adalah 0 (residual ideal)

    out_of_control_points = []
    for i in range(n):
        if Z_t[i] > UCL or Z_t[i] < LCL:
            out_of_control_points.append(i)

    return Z_t, UCL, LCL, out_of_control_points

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="TCN Forecasting & EWMA Monitoring")

st.title("Aplikasi Forecasting TCN dan Monitoring Residual EWMA")
st.write("Unggah file Excel Anda yang berisi data deret waktu untuk melakukan forecasting menggunakan Temporal Convolutional Network (TCN) dan memantau residual dengan Exponentially Weighted Moving Average (EWMA).")

# Sidebar untuk parameter
st.sidebar.header("Pengaturan Data & Model")
uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.sidebar.success("File berhasil diunggah!")

        # Asumsi kolom 'Tanggal' dan 'Volume' seperti di notebook
        if 'Tanggal' in df.columns and 'Volume' in df.columns:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            df.set_index('Tanggal', inplace=True)
            df_volume = df[['Volume']].copy() # Pastikan hanya kolom 'Volume' yang digunakan

            st.subheader("Pratinjau Data")
            st.write(df_volume.head())
            st.write(f"Jumlah baris data: {len(df_volume)}")

            # Parameter TCN
            st.sidebar.subheader("Parameter TCN")
            look_back = st.sidebar.slider("Look-back (jumlah langkah waktu input)", 1, 24, 12)
            train_ratio = st.sidebar.slider("Rasio data training", 0.5, 0.9, 0.8, 0.05)
            epochs = st.sidebar.slider("Epochs Training TCN", 10, 200, 50)
            batch_size = st.sidebar.slider("Batch Size TCN", 16, 128, 32)
            filters = st.sidebar.slider("Filters TCN", 32, 128, 64)
            kernel_size = st.sidebar.slider("Kernel Size TCN", 2, 5, 2)
            dilations = st.sidebar.multiselect("Dilations TCN", [1, 2, 4, 8, 16], [1, 2, 4])
            dropout_rate = st.sidebar.slider("Dropout Rate TCN", 0.0, 0.5, 0.2, 0.05)

            # Parameter EWMA
            st.sidebar.subheader("Parameter EWMA")
            lambda_ewma = st.sidebar.slider("Lambda EWMA (faktor smoothing)", 0.05, 1.0, 0.2, 0.05)
            L_ewma = st.sidebar.slider("L EWMA (multiplier batas kontrol)", 1.0, 5.0, 3.0, 0.1)

            if st.sidebar.button("Jalankan Analisis"):
                st.subheader("Memproses Data dan Melatih Model TCN...")
                # Normalisasi data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df_volume)

                # Split data training dan testing
                train_size = int(len(scaled_data) * train_ratio)
                # Pastikan test_data memiliki cukup data untuk look_back
                # Jika train_size - look_back < 0, atur ke 0
                test_start_index = max(0, train_size - look_back)
                train_data = scaled_data[0:train_size, :]
                test_data = scaled_data[test_start_index:, :]

                X_train, y_train = create_sequences(train_data, look_back)
                X_test, y_test = create_sequences(test_data, look_back)

                # Reshape input untuk TCN (samples, timesteps, features)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # Bangun dan latih model TCN
                model = build_tcn_model(input_shape=(look_back, 1),
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        dilations=dilations,
                                        dropout_rate=dropout_rate)

                with st.spinner("Melatih model TCN..."):
                    history = model.fit(X_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose=0) # verbose=0 agar tidak terlalu banyak output di Streamlit
                st.success("Pelatihan model TCN selesai!")

                # Prediksi
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)

                # Inverse transform prediksi ke skala asli
                train_predict = scaler.inverse_transform(train_predict)
                test_predict = scaler.inverse_transform(test_predict)
                y_train_actual = scaler.inverse_transform(y_train) # y_train sudah 2D dari create_sequences
                y_test_actual = scaler.inverse_transform(y_test)   # y_test sudah 2D dari create_sequences

                # Plot hasil forecasting
                st.subheader("Hasil Forecasting TCN")
                fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                ax_forecast.plot(df_volume.index, df_volume['Volume'], label='Aktual', color='black')

                # Buat array kosong dengan NaN untuk plotting yang lebih bersih
                # Ini memastikan plot prediksi muncul di indeks waktu yang benar
                train_predict_plot = np.empty_like(df_volume['Volume'], dtype=float) * np.nan
                # Sesuaikan indeks untuk prediksi training
                train_predict_start_idx = look_back
                train_predict_end_idx = look_back + len(train_predict)
                if train_predict_end_idx <= len(df_volume):
                    train_predict_plot[train_predict_start_idx : train_predict_end_idx] = train_predict.flatten()

                test_predict_plot = np.empty_like(df_volume['Volume'], dtype=float) * np.nan
                # Sesuaikan indeks untuk prediksi testing
                # Indeks aktual untuk y_test_actual dimulai dari train_size + look_back
                test_predict_start_idx = train_size + look_back
                test_predict_end_idx = train_size + look_back + len(test_predict)
                if test_predict_end_idx <= len(df_volume):
                    test_predict_plot[test_predict_start_idx : test_predict_end_idx] = test_predict.flatten()


                ax_forecast.plot(df_volume.index, train_predict_plot, label='Prediksi Training TCN', color='blue')
                ax_forecast.plot(df_volume.index, test_predict_plot, label='Prediksi Testing TCN', color='red')
                ax_forecast.set_title('Forecasting Volume dengan TCN')
                ax_forecast.set_xlabel('Tanggal')
                ax_forecast.set_ylabel('Volume')
                ax_forecast.legend()
                ax_forecast.grid(True)
                st.pyplot(fig_forecast)

                # Hitung Residual
                # Pastikan residual dihitung dari data yang diprediksi dan aktual yang sesuai
                # Untuk kesederhanaan, kita akan menghitung residual dari data test
                residuals = y_test_actual.flatten() - test_predict.flatten()
                residuals_df = pd.DataFrame(residuals, index=df_volume.index[test_predict_start_idx:test_predict_end_idx], columns=['Residual'])

                st.subheader("Residual Model TCN")
                fig_residuals, ax_residuals = plt.subplots(figsize=(12, 4))
                ax_residuals.plot(residuals_df.index, residuals_df['Residual'], label='Residual', color='purple', alpha=0.7)
                ax_residuals.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax_residuals.set_title('Residual Model TCN (Aktual - Prediksi)')
                ax_residuals.set_xlabel('Tanggal')
                ax_residuals.set_ylabel('Residual')
                ax_residuals.legend()
                ax_residuals.grid(True)
                st.pyplot(fig_residuals)

                # Monitoring Residual dengan EWMA
                st.subheader("Monitoring Residual dengan EWMA")
                if len(residuals) > 0: # Pastikan ada residual untuk dihitung EWMA
                    Z_t, UCL, LCL, out_of_control_points = calculate_ewma(residuals, lambda_ewma, L_ewma)

                    fig_ewma, ax_ewma = plt.subplots(figsize=(12, 6))
                    ax_ewma.plot(residuals_df.index, Z_t, label='EWMA', color='green', marker='o', markersize=4)
                    ax_ewma.axhline(UCL, color='red', linestyle='--', label=f'UCL ({UCL:.2f})')
                    ax_ewma.axhline(LCL, color='red', linestyle='--', label=f'LCL ({LCL:.2f})')
                    ax_ewma.axhline(0, color='gray', linestyle='-', linewidth=0.8) # Center line
                    ax_ewma.set_title('EWMA Chart untuk Residual')
                    ax_ewma.set_xlabel('Tanggal')
                    ax_ewma.set_ylabel('EWMA Statistik')
                    ax_ewma.legend()
                    ax_ewma.grid(True)

                    if out_of_control_points:
                        out_of_control_dates = residuals_df.index[out_of_control_points]
                        ax_ewma.plot(out_of_control_dates, Z_t[out_of_control_points], 'rx', markersize=10, label='Out-of-Control')
                        st.error(f"**Peringatan:** Ditemukan {len(out_of_control_points)} titik out-of-control pada EWMA chart!")
                        st.write("Tanggal out-of-control:")
                        st.write(out_of_control_dates.strftime('%Y-%m-%d').tolist())
                    else:
                        st.success("Semua titik berada dalam batas kontrol. Proses terkendali.")

                    st.pyplot(fig_ewma)
                else:
                    st.warning("Tidak ada residual untuk dihitung EWMA. Pastikan data testing memiliki cukup data.")

        else:
            st.error("File Excel harus mengandung kolom 'Tanggal' dan 'Volume'.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.write("Pastikan file Excel Anda memiliki format yang benar dan kolom 'Tanggal' serta 'Volume' tersedia.")
        st.write("Jika Anda mendapatkan error 'Found array with dim 3. None expected <= 2.',")
        st.write("ini biasanya berarti ada masalah dengan dimensi data yang tidak sesuai dengan yang diharapkan oleh fungsi.")
        st.write("Pastikan lingkungan Python Anda kompatibel dengan TensorFlow (disarankan Python 3.9-3.11).")
        st.write("Periksa kembali format file Excel Anda dan pastikan tidak ada data yang tidak terduga.")
else:
    st.info("Silakan unggah file Excel Anda untuk memulai.")
```

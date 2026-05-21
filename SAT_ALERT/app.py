import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.models import MultiModalFusion, TerrainCNN, PredictiveSituationalAwareness
from src.utils import load_model, plot_image, plot_mask
from src.data_generation import generate_optical_images, generate_time_series, generate_terrain_images

st.title("SAT_ALERT AI Sentinel Defence MVP")

st.header("Multi-Modal Data Fusion")
st.write("Integrating optical imagery and time series for risk assessment.")

if st.button("Generate Sample and Predict Risk"):
    sample_optical = generate_optical_images(1)[0]
    sample_ts = generate_time_series(1)[0]

    fig, ax = plt.subplots()
    ax.imshow(sample_optical)
    ax.set_title("Sample Optical Image")
    ax.axis('off')
    st.pyplot(fig)

    # Load model (assume trained)
    try:
        fusion_model = load_model('fusion')
        risk = fusion_model.predict([np.expand_dims(sample_optical, 0), np.expand_dims(sample_ts, 0)])
        st.write(f"Predicted Risk Score: {risk[0][0]:.4f}")
    except:
        st.write("Model not found. Please train models first by running lab.py")

st.header("Computer Vision for Terrain Analysis")
st.write("Detecting inundation risk zones in terrain images.")

if st.button("Analyze Terrain"):
    sample_terrain = generate_terrain_images(1)[0]

    fig, ax = plt.subplots()
    ax.imshow(sample_terrain)
    ax.set_title("Sample Terrain Image")
    ax.axis('off')
    st.pyplot(fig)

    try:
        cnn_model = load_model('terrain_cnn')
        mask = cnn_model.predict(np.expand_dims(sample_terrain, 0))[0]

        fig2, ax2 = plt.subplots()
        ax2.imshow(mask[:, :, 0], cmap='Reds', alpha=0.7)
        ax2.set_title("Detected Risk Zones")
        ax2.axis('off')
        st.pyplot(fig2)
    except:
        st.write("Model not found. Please train models first.")

st.header("Predictive Situational Awareness")
st.write("72-Hour Inundation Forecast Maps.")

if st.button("Generate Forecast"):
    sample_terrain = generate_terrain_images(1)[0]
    sample_ts = generate_time_series(1)[0]

    fig, ax = plt.subplots()
    ax.imshow(sample_terrain)
    ax.set_title("Current Terrain")
    ax.axis('off')
    st.pyplot(fig)

    try:
        psa_model = load_model('psa')
        forecast = psa_model.predict([np.expand_dims(sample_terrain, 0), np.expand_dims(sample_ts, 0)])[0]

        fig2, ax2 = plt.subplots()
        ax2.imshow(forecast[:, :, 0], cmap='Blues', alpha=0.7)
        ax2.set_title("Forecast Inundation Map")
        ax2.axis('off')
        st.pyplot(fig2)
    except:
        st.write("Model not found. Please train models first.")
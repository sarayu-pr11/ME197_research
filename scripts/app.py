import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.emg_pose_model import EMGPoseNet

st.title("🤖 EMG to Hand Pose Predictor")

# Load model
model = EMGPoseNet()
model.load_state_dict(torch.load("emg_model.pth", map_location=torch.device("cpu")))
model.eval()

uploaded_file = st.file_uploader("Upload a NumPy (.npy) EMG file (shape: channels x time)", type="npy")

if uploaded_file is not None:
    emg_data = np.load(uploaded_file)
    st.write("Shape of uploaded EMG:", emg_data.shape)
    
    # Convert to tensor
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32).unsqueeze(0)  # (1, C, T)

    with torch.no_grad():
        pose_prediction = model(emg_tensor).numpy()[0]

    st.subheader("🔮 Predicted Pose (First 10 joints)")
    st.write(pose_prediction[:10])

    # Plot result
    fig, ax = plt.subplots()
    ax.bar(range(len(pose_prediction)), pose_prediction)
    ax.set_title("Predicted Joint Angles")
    st.pyplot(fig)

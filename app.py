import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# Title and header image
st.title("🩺 Breast Cancer Prediction Model")
st.image("static/image1.jpeg", width=700)

st.write(
    "Enter the **values for 31 features** (comma-separated) "
    "as given by your medical report/test results."
)

# Example inputs (with 31 features each)
benign_example = (
    "0.42469734, -0.98179678, 1.41622208, -0.98258746, -0.86694414,"
    "0.05938999, -0.59678772, -0.82020317, -0.84511471, 0.31326409,"
    "0.07404147, -0.53850473, 0.53647286, -0.65795, -0.49659014,"
    "0.0654747, -0.82240418, -0.68556537, -0.89848456, 0.12329928,"
    "-0.43154667, -0.8291973, 1.59353039, -0.87357215, -0.74294685,"
    "0.79662437, -0.7293916, -0.77494969, -0.80948314, 0.79892783,"
    "-0.1344968"
)

malignant_example = (
    "-0.23717767, 0.13091005, 0.550292, 0.22752392, -0.02701312,"
    "0.69429016, 1.58702974, 0.67525209, 1.04029042, 1.60619085,"
    "0.90022553, -0.52153458, -0.40559204, -0.36166178, -0.40312701,"
    "-0.83088147, 0.26441801, -0.18728204, 0.25863065, -0.73913229,"
    "-0.03931498, 0.28816132, 1.26671357, 0.45884257, 0.03511686,"
    "0.90938777, 2.65789184, 1.33531714, 2.38938856, 2.15431093,"
    "2.44735687"
)

# Sidebar showing example inputs for copy-paste
st.sidebar.header("📋 Example Inputs (Copy & Paste)")
st.sidebar.subheader("Benign Example")
st.sidebar.code(benign_example, language="text")

st.sidebar.subheader("Malignant Example")
st.sidebar.code(malignant_example, language="text")

# Text area for user input
features_input = st.text_area(
    "Paste comma-separated values below:",
    placeholder="Enter 31 comma-separated feature values here..."
)

# Predict button
if st.button("🔮 Predict"):
    try:
        # Convert input into numbers
        features_list = [float(x.strip()) for x in features_input.split(",") if x.strip()]

        # Fix input size to exactly 31
        if len(features_list) < 31:
            features_list += [0.0] * (31 - len(features_list))
        elif len(features_list) > 31:
            features_list = features_list[:31]

        features_array = np.array(features_list, dtype=np.float32).reshape(1, -1)

        # Prediction
        pred = model.predict(features_array)
        output = "Cancerous" if pred[0] == 1 else "Not Cancerous"

        # Show result with styled output
        if output == "Cancerous":
            st.error("⚠️ The model predicts: **Cancerous**")
            st.image("static/img1.jpeg", width=300)
            st.markdown(
                "<p style='color:red; font-weight:bold;'>⚡ Please consult a doctor immediately for further diagnosis.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ The model predicts: **Not Cancerous**")
            st.image("static/img2.jpeg", width=300)
            st.markdown(
                "<p style='color:green; font-weight:bold;'>🎉 You are safe, but always keep regular check-ups.</p>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.warning("⚠️ Invalid input! Please enter numeric values only.")
        st.text(f"Error: {e}")
    
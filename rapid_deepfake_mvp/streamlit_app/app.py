# streamlit_app/app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Deepfake MVP Agent", layout="centered")

st.title("⚡ Rapid Deepfake Detector (Agentic MVP)")
st.markdown("Upload Media → Route (Img/Vid) → Smart Confidence Branching → Output")

uploaded_file = st.file_uploader("Upload Image or Video (JPG, PNG, MP4)", type=["jpg", "jpeg", "png", "mp4"])

intent = st.radio("Select Output Detail (User Intent)", ["quick", "detailed", "explanation"], horizontal=True)

if uploaded_file is not None:
    # Display preview
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Media", use_container_width=True)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        
    if st.button("Run Analysis"):
        with st.spinner("Analyzing Pipeline (Model A → Confidence Check → Escalation)..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"intent": intent}
            
            try:
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    res_json = response.json()
                    
                    if "error" in res_json:
                        st.error(res_json["error"])
                    else:
                        st.success("Analysis Complete!")
                        
                        summary = res_json.get("summary", "")
                        st.markdown("### Agent Output")
                        st.info(summary)
                        
                        warning = res_json.get("ethical_warning")
                        if warning:
                            st.warning(warning)
                            
                        with st.expander("Show Agent Log (Internal State)"):
                            st.json(res_json.get("raw", {}))
                else:
                    st.error(f"API Error {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Did you run `python -m uvicorn src.api.main:app`?")

from typing import Optional
import streamlit as st
from generate import ttf_to_image
from PIL import Image
import os

LOADED_TTF_KEY = "loaded_ttf"
SET_IMG_KEY = "set_img"
OUTPUT_IMG_KEY = "output_img"

def get_ttf(key: str) -> Optional[any]:
    if key in st.session_state:
        return st.session_state[key]
    return None

def get_img(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None

def set_img(key: str, img: Image.Image):
    st.session_state[key] = img

def ttf_uploader(prefix):
    file = st.file_uploader("TTF, OTF", ["ttf", "otf"], key=f"{prefix}-uploader")
    if file:    
        return file
        
    return get_ttf(LOADED_TTF_KEY)

def generate_button(prefix, file_input, version, **kwargs):

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider(
            "Number of inference sample",
            min_value=1,
            max_value=200,
            value=20,
            key=f"{prefix}-inference-sample",
        )
    with col2:
        ref_char_ids = st.text_area(
        "ref_char_ids",
        value="1,2,3,4,5,6,7,8",
        key=f"{prefix}-ref_char_ids",
        )
    enable_attention_slicing = st.checkbox(
        "Enable attention slicing (enables higher resolutions but is slower)",
        key=f"{prefix}-attention-slicing",
    )
    enable_cpu_offload = st.checkbox(
        "Enable CPU offload (if you run out of memory, e.g. for XL model)",
        key=f"{prefix}-cpu-offload",
        value=False,
    )

    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("⏳ Generating image..."):
            image = ttf_to_image(file_input, n_samples, ref_char_ids, version)
            set_img(OUTPUT_IMG_KEY, image.copy())
        st.image(image)

    test_font = st.text_area(
        "test font",
        value="กขคง",
        key=f"{prefix}-prompt",
    )

def generate_tab():
    prefix = "ttf2img"
    col1, col2 = st.columns(2)

    with col1:
        sample_choose = st.selectbox(
                "Choose Sample", ["Custom"] + [i for i in os.listdir("font_sample/")], key=f"{prefix}-sample_choose"
            )
        if sample_choose == "Custom":
            uploaded_file = ttf_uploader(prefix)
            if uploaded_file:
                st.write("filename:", uploaded_file.name)
                uploaded_file = uploaded_file.getbuffer() # Send file as Buffer

        else:
            st.write("filename:", sample_choose)
            uploaded_file = os.path.join("font_sample", sample_choose)

    with col2:
        if uploaded_file:
            version = st.selectbox(
                "Model version", ["TH2TH", "ENG2TH"], key=f"{prefix}-version"
            )
            generate_button(
                prefix, file_input=uploaded_file, version=version
            )

def main():
    st.set_page_config(layout="wide")
    st.title("ThaiVecFont Playground")

    generate_tab()

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_img(OUTPUT_IMG_KEY)
        if output_image:
            st.image(output_image)
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
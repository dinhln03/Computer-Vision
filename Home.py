import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)

# Thiết lập màu nền xám nhạt bằng CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f2f2f2;
    }
    </style>
    """,
    unsafe_allow_html=True
)


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: "images/img1.jpg";
    background-position: center;
}
</style>
"""


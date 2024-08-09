import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
st.set_page_config(page_title="Nhận dạng tế bào máu", page_icon="😶‍🌫️")

# Background and header styling
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
    background-image: url("https://i.pinimg.com/736x/1b/e2/91/1be2919a288c48fe59ba448f92898bcc.jpg");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define load model function
def load_model():
    # load model
    model = YOLO(r"./utility/DetectBloodCell/model.pt")
    return model

def main():
    # Red title with subtitle
    st.markdown(
        """
        <h1 style='color: red;'>Nhận diện tế bào máu</h1>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Phát hiện và đếm số lượng tế bào máu(hồng cầu và bạch cầu) từ hình ảnh được tải lên")

    st.markdown(
        """
        <style>
        .uploader {
            display: block;
            width: 60%;
            padding: 20px;
            margin: auto;
            border: 2px dashed #ddd;
            border-radius: 8px;
            text-align: center;
        }
        .result-container {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("Nhấp vào để tải ảnh lên", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        if uploaded_image.type.startswith('image/'):
            model = load_model()

        # Hiển thị hình ảnh đầu vào
        input_image = Image.open(uploaded_image)
        st.image(input_image, caption='Ảnh đã tải', use_column_width=True)

        # Tạo nút "Predict" để thực hiện dự đoán khi được nhấn
        if st.button("Predict"):
            # Đảm bảo rằng dữ liệu hình ảnh được chuyển thành numpy array
            input_image_np = np.array(input_image)

            # Dự đoán và hiển thị hình ảnh đầu ra
            output_image, wbc, rbc = predict(model, input_image_np)

            # Hiển thị hình ảnh đầu ra
            st.image(output_image, caption='Ảnh sau khi xử lí', use_column_width=True)

            # Hiển thị kết quả trong một container với màu nền
            with st.container():
                st.markdown(f'<p style="color:blue; font-size:20px;">White Blood Cell: {wbc}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:blue; font-size:20px;">Red Blood Cell: {rbc}</p>', unsafe_allow_html=True)

def predict(model, image_data):
    wbc = 0
    rbc = 0
    results = model(image_data)
    output_image = results[0].plot()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    for box in results[0].boxes.cls:
        if np.array(box) == 0:
            rbc += 1
        else:
            wbc += 1
    return output_image, wbc, rbc

if __name__ == '__main__':
    main()

import streamlit as st
from PIL import Image

from utils import transforms, vis

# app = Flask(__name__)

st.title("Machine Vision Spring 2023")

ALLOWED_FORMATS = ['png', 'jpeg', 'jpg', 'bmp']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FORMATS


uploaded_file = st.sidebar.file_uploader(
    'Choose Image',
)
if uploaded_file is None or uploaded_file.name == "":
    st.write('no file')
elif not allowed_file(uploaded_file.name):
    st.write('file not supported')
else:
    st.write(f"file {uploaded_file.name} selected")
    st.write("### Source Image")
    img = Image.open(uploaded_file)
    st.image(img, width=300)

q_level = st.sidebar.selectbox("Quantize Level",
                               [64, 16, 8, 4, 2])
st.write('You selected:', q_level)

clicked = st.sidebar.button("Run!!!")

if clicked:
    q_image = transforms.quantize(image=img, quantize_level=[q_level])['image'][1]
    out = 'homework1/' + 'out' + uploaded_file.name
    vis.rescale(q_image, out)
    st.write("### Out Image")
    out_image = Image.open(out)
    st.image(out, width=300, clamp=True)
    st.balloons()

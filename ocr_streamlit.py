import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import re

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=180)
st.sidebar.markdown("## OCR Keyword Underliner")
st.sidebar.info(
    "Upload up to 5 images and/or paste text below. "
    "Enter a keyword to highlight it in your images or text. "
    "Image preprocessing is applied for better OCR accuracy."
)
show_debug = st.sidebar.checkbox("Show OCR debug output", value=False)

# --- Main Title ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔎 OCR Keyword Underliner</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ..

# --- Image Section ---
st.markdown("### 📷 Image Keyword Underliner")
uploaded_files = st.file_uploader(
    "Upload up to 5 images", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=True
)
keyword_img_input = st.text_input("Enter keyword(s) to underline in images (comma-separated)").strip().lower()
keywords_img = [k.strip() for k in keyword_img_input.split(",") if k.strip()]

if uploaded_files and keywords_img:
    cols = st.columns(min(2, len(uploaded_files)))
    for idx, uploaded_file in enumerate(uploaded_files[:5]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Image {idx+1}**")
            img_pil = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            #
            _, thresh = cv2.threshold(img_bgr, 150, 255, cv2.THRESH_BINARY)
            denoised = cv2.fastNlMeansDenoisingColored(thresh, None, 10, 10, 7, 21)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            data = pytesseract.image_to_data(sharpened, output_type=pytesseract.Output.DICT)
            if show_debug:
                st.expander("OCR detected words").write(data["text"])

            found = False
            for i, word in enumerate(data["text"]):
                word_clean = word.strip().lower()
                if word_clean and any(k in word_clean for k in keywords_img):
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    cv2.line(img_bgr, (x, y + h + 2), (x + w, y + h + 2), (0, 0, 255), 4)
                    found = True

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_out = Image.fromarray(img_rgb)
            st.image(img_out, caption=f"Underlined Image {idx+1}", use_container_width=True)
            if found:
                st.success("✅ Underlined version shown")
            else:
                st.warning("⚠️ Keyword(s) not found in image.")

st.markdown("<hr>", unsafe_allow_html=True)

#Text Section ---
st.markdown("### 📝 Text Keyword Underliner")
text_input = st.text_area("Paste your text here", height=200)
keyword_text_input = st.text_input("Enter keyword(s) to underline in text (comma-separated)", key="text_keyword").strip()
keywords_text = [k.strip() for k in keyword_text_input.split(",") if k.strip()]

if text_input and keywords_text:
    def underline_keywords_in_text(text, keywords):
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<u style='color:#FF4B4B'>{m.group(0)}</u>", text)
        return text

    underlined_text = underline_keywords_in_text(text_input, keywords_text)
    st.markdown(underlined_text, unsafe_allow_html=True)
    st.success("✅ Underlined keyword(s) in text!")

# 

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #888;'>Made with ❤️ using Streamlit & Tesseract OCR</div>",
    unsafe_allow_html=True

)

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os # For checking model path

# --- Model Definition (Copied from your model.py) ---
import torch.nn as nn

class FullyConvolutionalAE(nn.Module):
    def __init__(self, latent_channels=64): # Number of channels in the bottleneck
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
# --- End Model Definition ---

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fcn_compression_ae.pth"  # IMPORTANT: Ensure this model exists!
                                     # Or change to "fcn_compression_ae2.pth" if that's your file
LATENT_CHANNELS = 64 # Must match the latent_channels used during training

# --- Model Loading (Cached) ---
@st.cache_resource # Use st.cache_resource for PyTorch models
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Please make sure it's in the same directory as app.py.")
        return None
    model = FullyConvolutionalAE(latent_channels=LATENT_CHANNELS)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval().to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Compression and Decompression Logic (Adapted) ---
def process_image(pil_image, model_instance, target_quality=50):
    """
    Compresses and then decompresses a PIL image using the loaded model.
    Returns the original PIL image, the reconstructed PIL image, and bytes for download.
    """
    if model_instance is None:
        return pil_image, None, None # Return if model failed to load

    original_pil_image = pil_image.copy() # Keep a copy

    # 1. Prepare image for compression
    img = original_pil_image.convert('RGB')
    width, height = img.size
    pad_w = (8 - width % 8) % 8
    pad_h = (8 - height % 8) % 8

    if pad_w > 0 or pad_h > 0:
        padded_img = Image.new(img.mode, (width + pad_w, height + pad_h), (0,0,0))
        padded_img.paste(img, (0,0))
        img_to_process_compression = padded_img
    else:
        img_to_process_compression = img

    img_array = np.array(img_to_process_compression, dtype=np.float32) / 255.0
    tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. Compress (Encode)
    with torch.no_grad():
        latent_tensor = model_instance.encoder(tensor)
    
    # Latent tensor is kept in memory (no need to save to .npy for this workflow)
    # original_size_before_padding = (width, height) # Stored as width, height

    # 3. Decompress (Decode)
    with torch.no_grad():
        reconstructed_padded_tensor = model_instance.decoder(latent_tensor)

    output_array_padded = reconstructed_padded_tensor.squeeze().cpu().numpy()
    output_array_padded = np.transpose(output_array_padded, (1, 2, 0))
    output_array_padded = np.clip(output_array_padded * 255, 0, 255).astype(np.uint8)
    
    reconstructed_pil_padded = Image.fromarray(output_array_padded, 'RGB')
    
    # Crop to original_w, original_h
    reconstructed_pil_final = reconstructed_pil_padded.crop((0, 0, width, height))

    # Prepare for download
    img_byte_arr = io.BytesIO()
    reconstructed_pil_final.save(img_byte_arr, format='JPEG', quality=target_quality)
    download_bytes = img_byte_arr.getvalue()

    return original_pil_image, reconstructed_pil_final, download_bytes

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🖼️ DeepCompress FCN: Image Compression & Decompression")
st.markdown("Upload an image, and the app will compress it using a Fully Convolutional Autoencoder, then decompress it. The reconstructed image will be available for download.")

# Load the model
model = load_model()

if model: # Only proceed if model is loaded
    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    quality = st.sidebar.slider("Output JPEG Quality", min_value=1, max_value=100, value=50, step=1)

    if uploaded_file is not None:
        # Display images in columns
        col1, col2 = st.columns(2)

        try:
            input_image_pil = Image.open(uploaded_file)
            
            with col1:
                st.subheader("Original Image")
                st.image(input_image_pil, caption=f"Original: {input_image_pil.width}x{input_image_pil.height}", use_column_width=True)

            if st.sidebar.button("✨ Compress & Decompress"):
                with st.spinner("Processing image... This might take a moment."):
                    original_img, reconstructed_img, download_bytes = process_image(input_image_pil, model, quality)
                
                if reconstructed_img:
                    with col2:
                        st.subheader("Reconstructed Image")
                        st.image(reconstructed_img, caption=f"Reconstructed (JPEG Q={quality}): {reconstructed_img.width}x{reconstructed_img.height}", use_column_width=True)
                    
                    st.success("Image processed successfully!")
                    
                    # Prepare filename for download
                    original_filename = os.path.splitext(uploaded_file.name)[0]
                    download_filename = f"{original_filename}_reconstructed_q{quality}.jpg"
                    
                    st.download_button(
                        label="📥 Download Reconstructed Image",
                        data=download_bytes,
                        file_name=download_filename,
                        mime="image/jpeg"
                    )
                else:
                    st.error("Failed to process the image.")
        except Exception as e:
            st.error(f"An error occurred while opening or processing the image: {e}")
            col2.empty() # Clear the second column if an error occurs before processing

else:
    st.warning("Model could not be loaded. Please check the console for errors and ensure the model file is correctly placed.")

st.markdown("---")
st.markdown("Powered by PyTorch & Streamlit. Model based on a Fully Convolutional Autoencoder.")
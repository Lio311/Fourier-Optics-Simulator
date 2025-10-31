import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Fourier Optics Simulator")

st.title("Fourier Optics Simulator: 2D Spatial Filtering")
st.markdown("This app simulates a 4f optical system. An uploaded image is Fourier transformed (like passing through a lens), "
            "filtered in the frequency domain (the 'Fourier plane'), and then inverse transformed (like passing through a second lens).")

# --- Helper Functions ---

def create_mask(shape, filter_type="Low-Pass", radius=30):
    """
    Creates a frequency-domain mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates
    
    mask = np.zeros((rows, cols), np.uint8)
    
    if filter_type == "Low-Pass":
        # Create a circular mask (aperture)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        return mask

    elif filter_type == "High-Pass (Edge Detection)":
        # Create an inverted circular mask (central block)
        mask.fill(1)
        cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)
        return mask
        
    elif filter_type == "Band-Pass (Ring)":
        # Create a ring mask
        outer_radius = radius + 10
        cv2.circle(mask, (ccol, crow), outer_radius, 1, thickness=-1)
        cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)
        return mask

    return mask

def process_image(image, filter_type, radius):
    """
    Applies the full Fourier filtering process to an image.
    """
    # 1. Convert image to grayscale float
    img_gray = np.array(image.convert('L'), dtype=np.float64)
    
    # --- Step 1: Fourier Transform (Lens 1) ---
    # Perform 2D FFT and shift the zero-frequency component to the center
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum for visualization (log scale)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9) # +1e-9 to avoid log(0)
    
    # --- Step 2: Create Filter Mask ---
    mask = create_mask(img_gray.shape, filter_type, radius)
    
    # --- Step 3: Apply Mask in Fourier Plane ---
    f_shift_filtered = f_shift * mask
    
    # Calculate filtered spectrum for visualization
    magnitude_spectrum_filtered = 20 * np.log(np.abs(f_shift_filtered) + 1e-9)

    # --- Step 4: Inverse Fourier Transform (Lens 2) ---
    # Inverse shift to move zero-frequency back to corner
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    
    # Inverse FFT
    img_filtered = np.fft.ifft2(f_ishift)
    
    # Get the magnitude (real part) of the complex result
    img_filtered = np.abs(img_filtered)
    
    # Normalize images for display
    def normalize(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return (img * 255).astype(np.uint8)

    return (
        normalize(img_gray),
        normalize(magnitude_spectrum),
        mask * 255,  # Mask is already 0 or 1, just scale to 255
        normalize(magnitude_spectrum_filtered),
        normalize(img_filtered)
    )


# --- Streamlit UI ---

# Sidebar Controls
st.sidebar.header("Filter Controls")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])

filter_type = st.sidebar.selectbox(
    "Select Filter Type",
    ["Low-Pass", "High-Pass (Edge Detection)", "Band-Pass (Ring)"]
)

radius = st.sidebar.slider(
    "Filter Size (Radius)", 
    min_value=1, 
    max_value=200, 
    value=30
)

# Main Page Display
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Process the image
    img_orig, spec_orig, mask_img, spec_filtered, img_final = process_image(image, filter_type, radius)
    
    st.subheader("The Optical Filtering Process")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_orig, caption="1. Original Image (Input Plane)", use_column_width=True)
    
    with col2:
        st.image(spec_orig, caption="2. Fourier Transform (Fourier Plane)", use_column_width=True)
        
    with col3:
        st.image(mask_img, caption="3. Spatial Filter (Mask)", use_column_width=True)

    st.divider()
    
    col4, col5 = st.columns(2)
    with col4:
        st.image(spec_filtered, caption="4. Filtered Fourier Transform", use_column_width=True)
        
    with col5:
        st.image(img_final, caption="5. Final Image (Image Plane)", use_column_width=True)

    # --- Explanation Expander ---
    with st.expander("What am I looking at?"):
        st.markdown(
            """
            This simulation demonstrates **Spatial Filtering**, a core concept in Fourier Optics.

            1.  **Original Image:** Your input.
            2.  **Fourier Transform:** This is what a lens "sees" at its back focal plane.
                - **The bright center** represents low spatial frequencies (overall brightness, large structures).
                - **The outer regions** represent high spatial frequencies (sharp edges, fine details).
            3.  **Spatial Filter (Mask):** This is a physical "stop" or "aperture" we place in the Fourier plane.
                - **Low-Pass:** A hole that only lets the center (low frequencies) pass.
                - **High-Pass:** A block that only lets the edges (high frequencies) pass.
            4.  **Filtered Fourier Transform:** The light *after* it has passed through the mask.
            5.  **Final Image:** An inverse Fourier transform (performed by a second lens) reconstructs the image from the remaining frequencies.

            ### What to Observe:
            - **Low-Pass Filter:** By blocking high frequencies, we **lose sharp edges**. The result is a **blurred image**.
            - **High-Pass Filter:** By blocking low frequencies, we **lose the "substance"** of the image and are left with **only the sharp edges**. This is a fundamental method for **edge detection**.
            - **Band-Pass Filter:** We keep only a "ring" of frequencies, resulting in an image that shows details of a specific size.
            """
        )

else:
    st.info("Please upload an image using the sidebar to begin the simulation.")

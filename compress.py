import argparse
import torch
import numpy as np
from PIL import Image
from model import FullyConvolutionalAE # Use the FCN model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fcn_compression_ae.pth" # Path to the trained FCN model
LATENT_CHANNELS = 64 # Must match the latent_channels used during training

def compress_to_latent(input_path, latent_path):
    """
    Compress an image to its latent representation using a trained Fully Convolutional Autoencoder (FCN).

    Args:
        input_path (str): Path to the input image file.
        latent_path (str): Path to save the output latent .npy file.

    Returns:
        None. Saves the latent representation and original image size to a .npy file.
    """
    # Load the trained model
    model = FullyConvolutionalAE(latent_channels=LATENT_CHANNELS)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval().to(device)

    # Load and preprocess the image
    img = Image.open(input_path).convert('RGB')
    
    # Ensure image dimensions are divisible by 8 (for 3 downsampling layers)
    width, height = img.size
    pad_w = (8 - width % 8) % 8
    pad_h = (8 - height % 8) % 8

    if pad_w > 0 or pad_h > 0:
        # Pad image with black pixels if necessary
        padded_img = Image.new(img.mode, (width + pad_w, height + pad_h), (0,0,0))
        padded_img.paste(img, (0,0))
        img_to_process = padded_img
        print(f"Padded image from ({width}x{height}) to ({width+pad_w}x{height+pad_h}) for model compatibility.")
    else:
        img_to_process = img

    # Convert image to tensor
    img_array = np.array(img_to_process, dtype=np.float32) / 255.0
    tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # Encode image to latent representation
    with torch.no_grad():
        latent = model.encoder(tensor) # Latent is now a spatial tensor

    # Save latent tensor and original size
    data_to_save = {
        'latent': latent.cpu().numpy(),
        'original_size_before_padding': (width, height) # Store original W, H
    }
    np.save(latent_path, data_to_save, allow_pickle=True)
    print(f"[✓] Latent representation (spatial tensor) saved to {latent_path}")


def decompress_from_latent(latent_path, output_path, quality=85):
    """
    Decompress an image from its latent representation using a trained Fully Convolutional Autoencoder (FCN).

    Args:
        latent_path (str): Path to the latent .npy file.
        output_path (str): Path to save the reconstructed image.
        quality (int, optional): JPEG quality for output image. Default is 85.

    Returns:
        None. Saves the reconstructed image to the specified path.
    """
    # Load the trained model
    model = FullyConvolutionalAE(latent_channels=LATENT_CHANNELS)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval().to(device)

    # Load latent data
    try:
        loaded_data = np.load(latent_path, allow_pickle=True).item()
        latent_np = loaded_data['latent']
        original_w, original_h = loaded_data['original_size_before_padding']
    except FileNotFoundError:
        print(f"Error: Latent file '{latent_path}' not found.")
        return
    except KeyError:
        print(f"Error: Latent file '{latent_path}' is in an old format or missing data.")
        return
        
    latent_tensor = torch.FloatTensor(latent_np).to(device)

    # Decode latent representation to image
    with torch.no_grad():
        reconstructed_padded = model.decoder(latent_tensor) # Output has dimensions of the padded input

    output_array_padded = reconstructed_padded.squeeze().cpu().numpy()
    output_array_padded = np.transpose(output_array_padded, (1, 2, 0))
    output_array_padded = np.clip(output_array_padded * 255, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and crop back to original size (if padding was applied)
    reconstructed_pil_padded = Image.fromarray(output_array_padded, 'RGB')
    
    # Crop to original_w, original_h (top-left crop)
    final_img = reconstructed_pil_padded.crop((0, 0, original_w, original_h))

    final_img.save(output_path, quality=quality)
    print(f"[✓] Decompressed image (original size {original_w}x{original_h}) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepCompress FCN - Autoencoder Image Compression")
    # LATENT_CHANNELS and MODEL_PATH are now script constants

    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_compress = subparsers.add_parser("compress")
    parser_compress.add_argument("input", help="Input image path")
    parser_compress.add_argument("-l", "--latent", default="fcn_compressed_latent.npy", help="Output latent file")

    parser_decompress = subparsers.add_parser("decompress")
    parser_decompress.add_argument("latent", help="Latent .npy file")
    parser_decompress.add_argument("-o", "--output", default="fcn_reconstructed.jpg", help="Output image path")
    parser_decompress.add_argument("-q", "--quality", type=int, default=85, help="JPEG quality for output")

    args = parser.parse_args()

    if args.mode == "compress":
        compress_to_latent(args.input, args.latent)
    elif args.mode == "decompress":
        decompress_from_latent(args.latent, args.output, args.quality)
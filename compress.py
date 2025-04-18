# File 4: compress.py (CLI Tool)
import argparse
import torch
import numpy as np
from PIL import Image
from model import CompressionAE

def compress_image(input_path, output_path, quality=85):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompressionAE()
    model.load_state_dict(torch.load("compression_ae.pth", map_location=device))
    model.eval()

    # Load and process image
    img = Image.open(input_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # Compression
    with torch.no_grad():
        compressed = model(tensor)

    # Convert back to image
    output_array = compressed.squeeze().permute(1, 2, 0).cpu().numpy()
    output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)
    Image.fromarray(output_array).save(output_path, quality=quality)

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Image Compressor")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", default="compressed.jpg",
                        help="Output path (jpg/png)")
    parser.add_argument("-q", "--quality", type=int, default=85,
                        help="Quality (1-100), higher=better")

    args = parser.parse_args()

    try:
        output = compress_image(args.input, args.output, args.quality)
        print(f"Successfully compressed image saved to {output}")
    except Exception as e:
        print(f"Error: {str(e)}")
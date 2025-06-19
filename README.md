# DeepCompress: Fully Convolutional Autoencoder for Image Compression

This project provides tools for compressing and decompressing images using a Fully Convolutional Autoencoder (FCN). You can use it as a web app (via Streamlit) or from the command line (CLI).

## Setup

1. **Clone the repository**

```bash
# Clone this repository
git clone <your-repo-url>
cd deepcompress
```

2. **Download large files (pretrained model) with Git LFS**

If you have not already, install [Git LFS](https://git-lfs.github.com/) and run:

```bash
git lfs pull
```

This will download the pretrained model file (`fcn_compression_ae.pth`) if it is tracked with Git LFS.

3. **Install dependencies**

It's recommended to use a virtual environment:

```bash
python -m venv venv
```

If the above command does not work, remove the created `venv` folder and run:

```bash
python -m venv venv --symlinks
```

On Windows, activate with:
```bash
venv\Scripts\activate
```
On macOS/Linux, activate with:
```bash
source venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web App (Streamlit)

The web app allows you to upload an image, compress it, and download the reconstructed image.

```bash
streamlit run app.py
```

- Open the provided local URL in your browser.
- Upload an image (JPG/PNG), set the output quality, and click "Compress & Decompress".
- Download the reconstructed image.

**Note:** Make sure the model file `fcn_compression_ae.pth` is present in the project directory. You can train your own model using `train_cifar10.py` or `train.py`.

## Running from the Command Line (CLI)

The CLI tool is provided in `compress.py` and supports both compression and decompression.

### Compress an image to latent representation

```bash
python compress.py compress path/to/input.jpg -l path/to/output_latent.npy
```
- `path/to/input.jpg`: Path to your input image.
- `-l path/to/output_latent.npy`: (Optional) Output file for the latent representation (default: `fcn_compressed_latent.npy`).

### Decompress from latent representation to image

```bash
python compress.py decompress path/to/output_latent.npy -o path/to/reconstructed.jpg -q 85
```
- `path/to/output_latent.npy`: Path to the latent file saved during compression.
- `-o path/to/reconstructed.jpg`: (Optional) Output image path (default: `fcn_reconstructed.jpg`).
- `-q 85`: (Optional) JPEG quality (default: 85).

## Training the Model

You can train the FCN model using CIFAR-10 or your own dataset:

- **CIFAR-10:**
  ```bash
  python train_cifar10.py
  ```
- **Custom Dataset:**
  Place your images in `./data/train` and run:
  ```bash
  python train.py
  ```

This will produce a `.pth` model file you can use for compression and decompression.

## Requirements
- Python 3.8 or higher
- See `requirements.txt` for all dependencies (PyTorch, Streamlit, Pillow, numpy, etc.)

## Notes
- The model expects images with dimensions divisible by 8. Padding is handled automatically.
- For best results, use images similar in size and content to those used for training.

---

**Authors:** Aviv Elbaz ([aviv000](https://github.com/aviv000)), Ron Butbul ([RonButbul626](https://github.com/RonButbul626))

---

For more comprehensive details, troubleshooting, and advanced usage, please refer to the instruction manual (available in `.md`, `.docx`, and `.pdf` formats) located in the `instruction_manual` folder.
import fitz  # PyMuPDF
import os
from PIL import Image

def pdf_to_images(pdf_path, output_dir="images", dpi=300):
    """
    Converts PDF pages to high-resolution PNG images using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the source PDF file.
        output_dir (str): Directory to save generated images.
        dpi (int): Dots per inch for image resolution.
        
    Returns:
        list: Paths to the generated images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        # Calculate matrix for DPI
        zoom = dpi / 72  # 72 is the default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
        
    doc.close()
    return image_paths

if __name__ == "__main__":
    # Test conversion
    test_pdf = "data.pdf" # Assuming a test file exists
    if os.path.exists(test_pdf):
        imgs = pdf_to_images(test_pdf)
        print(f"Generated {len(imgs)} images in 'images/' folder.")
    else:
        print(f"PDF {test_pdf} not found for testing.")

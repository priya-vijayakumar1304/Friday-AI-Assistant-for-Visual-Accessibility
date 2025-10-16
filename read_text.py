import torch
from PIL import Image
import numpy as np
import easyocr

# --- OCR Configuration ---
CONF_THRESHOLD = 0.40   # Ignore text below this confidence
Y_THRESHOLD = 15        # For grouping text in the same line

gpu = True if torch.cuda.is_available() else False
reader = easyocr.Reader(['en'], gpu=gpu)


# --- Group results into lines based on y-coordinate ---
def group_lines(results, y_threshold=Y_THRESHOLD):
    # Sort all results top-to-bottom
    results = sorted(results, key=lambda x: x[0][0][1])

    lines = []
    current_line = []
    last_y = None

    for (bbox, text, conf) in results:
        y = bbox[0][1]
        if last_y is None or abs(y - last_y) < y_threshold:
            current_line.append((bbox, text, conf))
        else:
            lines.append(current_line)
            current_line = [(bbox, text, conf)]
        last_y = y

    if current_line:
        lines.append(current_line)
    return lines


def read_text(image: Image.Image) -> str :
    """Extracts and returns text from an image using OCR."""
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # Run OCR
    results = reader.readtext(image_np)

    # --- Filter low-confidence results ---
    results = [r for r in results if r[2] >= CONF_THRESHOLD]

    if not results:
        return "no text found"

    # --- Reconstruct text line-by-line ---
    lines = group_lines(results)

    final_text = []
    for line in lines:
        # Sort left to right within a line
        line = sorted(line, key=lambda x: x[0][0][0])
        joined = " ".join([w[1] for w in line])
        final_text.append(joined)

    return "\n".join(final_text) 

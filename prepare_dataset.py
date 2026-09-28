import os
import cv2
import numpy as np
import zipfile
import shutil
from pathlib import Path

ZIP_PATH = r"C:\Users\nsgud\Downloads\AI4Shipwrecks.zip"
OUT_DIR = "data/tiled"
TILE_SIZE = 640
OVERLAP = 100

def get_contours_yolo_format(mask, tile_size):
    """Finds contours in a binary mask and converts them to YOLO format."""
    # Ensure binary mask
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_polygons = []
    for contour in contours:
        # Filter very small contours to avoid noise
        if cv2.contourArea(contour) < 10:
            continue
            
        # Normalize coordinates between 0 and 1
        normalized_contour = contour.astype(np.float32) / tile_size
        
        # Flatten and format for YOLO: class x1 y1 x2 y2 ...
        # Class is 0 (Shipwreck)
        poly_str = "0 " + " ".join([f"{pt[0][0]:.6f} {pt[0][1]:.6f}" for pt in normalized_contour])
        yolo_polygons.append(poly_str)
        
    return yolo_polygons

def process_image(img, mask, split, base_name):
    h, w = img.shape[:2]
    step = TILE_SIZE - OVERLAP
    
    out_img_dir = Path(OUT_DIR) / split / "images"
    out_lbl_dir = Path(OUT_DIR) / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # If image is smaller than TILE_SIZE, just pad it
    if h <= TILE_SIZE and w <= TILE_SIZE:
        img_tile = cv2.copyMakeBorder(img, 0, max(0, TILE_SIZE - h), 0, max(0, TILE_SIZE - w), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask_tile = cv2.copyMakeBorder(mask, 0, max(0, TILE_SIZE - h), 0, max(0, TILE_SIZE - w), cv2.BORDER_CONSTANT, value=[0])
        yolo_labels = get_contours_yolo_format(mask_tile, TILE_SIZE)
        tile_name = f"{base_name}_0_0"
        cv2.imwrite(str(out_img_dir / f"{tile_name}.png"), img_tile)
        with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
            if yolo_labels:
                f.write("\n".join(yolo_labels) + "\n")
        return

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate coordinates
            y1, x1 = y, x
            y2, x2 = y + TILE_SIZE, x + TILE_SIZE
            
            # Adjust if we go out of bounds
            if y2 > h:
                y1 = max(0, h - TILE_SIZE)
                y2 = h
            if x2 > w:
                x1 = max(0, w - TILE_SIZE)
                x2 = w
                
            # Crop
            img_tile = img[y1:y2, x1:x2]
            mask_tile = mask[y1:y2, x1:x2]
            
            # Pad if smaller than TILE_SIZE (only happens if original image is smaller than TILE_SIZE)
            tile_h, tile_w = img_tile.shape[:2]
            if tile_h < TILE_SIZE or tile_w < TILE_SIZE:
                img_tile = cv2.copyMakeBorder(img_tile, 0, TILE_SIZE - tile_h, 0, TILE_SIZE - tile_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                mask_tile = cv2.copyMakeBorder(mask_tile, 0, TILE_SIZE - tile_h, 0, TILE_SIZE - tile_w, cv2.BORDER_CONSTANT, value=[0])
                
            # Get YOLO labels
            yolo_labels = get_contours_yolo_format(mask_tile, TILE_SIZE)
            
            # Only save tile if it has an object, or we could save background tiles too.
            # To avoid saving thousands of empty sea tiles, we'll save all positive tiles
            # and a small fraction (e.g., 10%) of negative background tiles.
            is_background = len(yolo_labels) == 0
            if is_background and np.random.rand() > 0.1:
                continue
                
            tile_name = f"{base_name}_{y1}_{x1}"
            
            # Save Image
            cv2.imwrite(str(out_img_dir / f"{tile_name}.png"), img_tile)
            
            # Save Label (even if empty, for background)
            with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels) + "\n")

def main():
    print("Extracting and processing dataset...")
    
    if os.path.exists(OUT_DIR):
        print(f"Cleaning {OUT_DIR}...")
        shutil.rmtree(OUT_DIR)
        
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # Get all image paths
        all_files = z.namelist()
        
        # In AI4Shipwrecks, we have train and test directories
        for split in ['train', 'test']:
            print(f"Processing {split} split...")
            
            # Find image files
            img_paths = [f for f in all_files if f.startswith(f'AI4Shipwrecks/{split}/images/') and f.endswith('.png')]
            
            for i, img_path in enumerate(img_paths):
                # Corresponding label path
                base_name = os.path.basename(img_path).replace('.png', '')
                lbl_path = img_path.replace('/images/', '/labels/')
                
                # Check if label exists
                if lbl_path not in all_files:
                    continue
                    
                # Read Image
                img_data = z.read(img_path)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Read Label
                lbl_data = z.read(lbl_path)
                mask = cv2.imdecode(np.frombuffer(lbl_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                
                process_image(img, mask, split, base_name)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(img_paths)} images in {split}...")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()

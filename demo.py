import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import SonarPreprocessor

def save_individual_results(original, upscaled, median_filtered, despeckled, enhanced, base_name=""):
    """
    Saves each stage of the preprocessing pipeline as a separate image.
    """
    out_dir = "assets"
    prefix = base_name + "_" if base_name else ""
    
    cv2.imwrite(os.path.join(out_dir, f"{prefix}1_original.jpg"), original)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}2_median_filtered.jpg"), median_filtered)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}3_lee_despeckled.jpg"), despeckled)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}4_clahe_enhanced.jpg"), enhanced)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}5_upscaled.jpg"), upscaled)
    
    print(f"Saved individual step images to {out_dir}/ with prefix '{prefix}'")

def main():
    import sys
    
    # Default paths
    model_path = "assets/ESPCN_x4.pb"
    image_path = "assets/sample_sonar.jpg"
    
    # Override image path if provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}. Please check the path.")
        return
        
    print("Initializing Sonar Preprocessor...")
    # Initialize processor with the super-resolution model
    processor = SonarPreprocessor(sr_model_path=model_path if os.path.exists(model_path) else None)
    
    print(f"Processing image: {image_path}...")
    try:
        upscaled, original, median, despeckled, enhanced = processor.process(image_path)
        print("Processing complete!")
        print(f"Original shape: {original.shape}, Upscaled shape: {upscaled.shape}")
        
        # Determine output path based on input image
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"assets/{name}_comparison.jpg"
        
        # Save results individually
        save_individual_results(original, upscaled, median, despeckled, enhanced, base_name=name)
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()

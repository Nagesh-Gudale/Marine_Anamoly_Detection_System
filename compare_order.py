import cv2
import time
import os
from preprocessing import SonarPreprocessor

def run_comparison():
    img_path = "assets/noisy_shipwreck.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    processor = SonarPreprocessor()
    
    print("--- Method A: Denoise FIRST, then Upscale (Current) ---")
    start_time = time.time()
    
    # Denoise
    median = processor.apply_median_filter(img)
    despeckled = processor.apply_lee_filter(median)
    enhanced = processor.apply_clahe(despeckled)
    
    # Upscale
    upscaled_A = processor.apply_super_resolution(enhanced, method='lanczos')
    
    time_A = time.time() - start_time
    print(f"Time taken: {time_A:.2f} seconds")
    cv2.imwrite("assets/order_A_denoise_then_upscale.jpg", upscaled_A)
    
    
    print("\n--- Method B: Upscale FIRST, then Denoise ---")
    start_time = time.time()
    
    # Upscale first
    upscaled_initial = processor.apply_super_resolution(img, method='lanczos')
    
    # Since the image is 4x larger, the noise blobs are 4x larger. 
    # A standard 3x3 median or 5x5 Lee filter will be almost useless. 
    # But we'll run standard sizes first to demonstrate.
    median_B = processor.apply_median_filter(upscaled_initial)
    despeckled_B = processor.apply_lee_filter(median_B)
    enhanced_B = processor.apply_clahe(despeckled_B)
    
    time_B = time.time() - start_time
    print(f"Time taken: {time_B:.2f} seconds")
    cv2.imwrite("assets/order_B_upscale_then_denoise.jpg", enhanced_B)
    
    print("\nComparison complete. Check assets/order_A_denoise_then_upscale.jpg and assets/order_B_upscale_then_denoise.jpg")

if __name__ == "__main__":
    run_comparison()

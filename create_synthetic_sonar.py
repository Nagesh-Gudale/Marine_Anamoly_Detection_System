import cv2
import numpy as np
import os

def generate_synthetic_sonar(output_path="assets/sample_sonar.jpg"):
    # Create a dark, textured background (sea floor)
    bg = np.random.normal(50, 10, (400, 600)).astype(np.float32)
    
    # Add a bright object (debris/anomaly)
    cv2.rectangle(bg, (250, 150), (350, 200), 200, -1)
    
    # Add an acoustic shadow behind the object (dark region)
    cv2.rectangle(bg, (350, 150), (450, 200), 10, -1)
    
    # Add another small object
    cv2.circle(bg, (100, 300), 15, 180, -1)
    cv2.circle(bg, (115, 300), 15, 10, -1) # shadow
    
    # Add Speckle Noise (multiplicative)
    speckle = np.random.normal(1, 0.3, bg.shape)
    noisy_bg = bg * speckle
    
    # Add Salt and Pepper noise
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(noisy_bg)
    
    # Salt mode
    num_salt = np.ceil(amount * out.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in out.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount* out.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in out.shape]
    out[tuple(coords)] = 0
    
    # Clip and save
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Save as 3-channel image for consistency
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out_bgr)
    print(f"Saved synthetic sonar image to {output_path}")

if __name__ == "__main__":
    generate_synthetic_sonar()

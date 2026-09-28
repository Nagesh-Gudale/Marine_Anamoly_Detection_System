import cv2
import numpy as np
from scipy.ndimage import uniform_filter

class SonarPreprocessor:
    def __init__(self, sr_model_path=None):
        """
        Initializes the Sonar Preprocessor.
        :param sr_model_path: Path to the pre-trained super-resolution model (.pb file)
        """
        self.sr = None
        if sr_model_path:
            self._init_super_resolution(sr_model_path)

    def _init_super_resolution(self, model_path):
        """Initializes OpenCV's dnn_superres module."""
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            
            # Extract model name and scale from filename, e.g., ESPCN_x4.pb
            model_name = "espcn"
            scale = 4
            if "FSRCNN" in model_path.upper():
                model_name = "fsrcnn"
            if "x2" in model_path: scale = 2
            elif "x3" in model_path: scale = 3
            elif "x4" in model_path: scale = 4
                
            self.sr.setModel(model_name, scale)
            print(f"Loaded Super Resolution model: {model_name} (Scale: x{scale})")
        except Exception as e:
            print(f"Warning: Failed to load super resolution model. Upscaling will fall back to Bicubic. Error: {e}")
            self.sr = None

    def apply_median_filter(self, image, kernel_size=3):
        """
        Applies median filter to remove salt-and-pepper noise.
        """
        return cv2.medianBlur(image, kernel_size)

    def apply_lee_filter(self, image, window_size=5):
        """
        Applies Lee filter for speckle noise reduction (despeckling).
        Lee filter is a standard algorithm from SAR/Sonar research.
        """
        # Convert to float for mathematical operations
        img_float = np.float32(image)
        
        # Calculate local mean
        local_mean = uniform_filter(img_float, window_size)
        
        # Calculate local variance: E[X^2] - E[X]^2
        local_sqr_mean = uniform_filter(img_float**2, window_size)
        local_var = local_sqr_mean - local_mean**2
        
        # Estimate global noise variance (often estimated from a homogeneous area, 
        # here we use the mean of local variances as a simple heuristic)
        noise_variance = np.mean(local_var)
        
        # Calculate the Lee filter weights
        # Avoid division by zero
        weight = local_var / (local_var + noise_variance + 1e-8)
        
        # Apply the filter: mean + W * (image - mean)
        filtered_img = local_mean + weight * (img_float - local_mean)
        
        # Clip values to valid range and convert back to original dtype
        filtered_img = np.clip(filtered_img, 0, 255).astype(image.dtype)
        return filtered_img

    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        This enhances acoustic shadows which are crucial for anomaly detection.
        """
        # CLAHE works on grayscale, if it's BGR, convert to LAB and apply to L channel
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)

    def apply_super_resolution(self, image, method='lanczos'):
        """
        Upscales the image. By default uses Lanczos4 interpolation which is 
        highly effective and doesn't hallucinate artifacts on sonar textures.
        """
        if method == 'dnn' and self.sr is not None:
            return self.sr.upsample(image)
        else:
            # Fallback to standard Lanczos4 or Bicubic upscaling
            h, w = image.shape[:2]
            return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)

    def process(self, image_path, use_dnn_upscaling=False):
        """
        Runs the full preprocessing pipeline on an image.
        """
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        # 2. Salt and pepper removal
        median_filtered = self.apply_median_filter(img)

        # 3. Despeckling (Lee Filter)
        # Apply to each channel if color, though sonar is typically grayscale
        if len(median_filtered.shape) == 3:
            b, g, r = cv2.split(median_filtered)
            b_lee = self.apply_lee_filter(b)
            g_lee = self.apply_lee_filter(g)
            r_lee = self.apply_lee_filter(r)
            despeckled = cv2.merge((b_lee, g_lee, r_lee))
        else:
            despeckled = self.apply_lee_filter(median_filtered)

        # 4. Contrast Enhancement
        enhanced = self.apply_clahe(despeckled)

        # 5. Super-Resolution / Upscaling
        method = 'dnn' if use_dnn_upscaling else 'lanczos'
        upscaled = self.apply_super_resolution(enhanced, method=method)

        return upscaled, img, median_filtered, despeckled, enhanced

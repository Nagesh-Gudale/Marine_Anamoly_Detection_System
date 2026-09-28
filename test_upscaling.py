import cv2
import os

def test_upscale():
    img_path = "assets/noisy_shipwreck_4_clahe_enhanced.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print("Image not found")
        return
        
    h, w = img.shape[:2]
    
    # 1. Bicubic
    bicubic = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("assets/test_bicubic.jpg", bicubic)
    
    # 2. Lanczos4
    lanczos = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("assets/test_lanczos.jpg", lanczos)
    
    # 3. ESPCN
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("assets/ESPCN_x4.pb")
        sr.setModel("espcn", 4)
        espcn = sr.upsample(img)
        cv2.imwrite("assets/test_espcn.jpg", espcn)
    except Exception as e:
        print(f"ESPCN failed: {e}")

    print("Upscaling tests complete. Check test_bicubic.jpg, test_lanczos.jpg, and test_espcn.jpg")

if __name__ == "__main__":
    test_upscale()

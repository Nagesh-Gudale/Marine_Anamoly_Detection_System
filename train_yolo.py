from ultralytics import YOLO

def main():
    print("Initializing YOLOv8-seg model...")
    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    print("Starting training on AI4Shipwrecks tiled dataset...")
    # Train the model
    # Note: 'device=0' uses the first GPU if available. 
    # If no GPU is available, it will automatically fallback to CPU unless explicitly forced.
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        name='shipwrecks_seg_tiled',
        batch=16, # Adjust if out of memory
        patience=10
    )
    
    print("Training complete! Results saved to runs/segment/shipwrecks_seg_tiled")

if __name__ == '__main__':
    main()

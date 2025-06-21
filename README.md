# HS-Pose Vehicle Pose Estimation

Implementation of HS-Pose architecture adapted for vehicle pose estimation using KITTI dataset.

## Quick Start

```bash
# Clone repository
git clone https://github.com/[YourUsername]/HS-Pose-Vehicle-Estimation.git
cd HS-Pose-Vehicle-Estimation

# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py --mode demo

# Train model
python main.py --mode train --epochs 50

# Evaluate model
python main.py --mode eval --model_path ./models/best_model.pth

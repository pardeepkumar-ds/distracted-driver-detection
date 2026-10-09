# distracted-driver-detection

Deep learning image classifier for the State Farm distracted driver Kaggle competition using custom CNN and transfer learning.

Overview
Built a deep learning image classifier for the State Farm Kaggle competition that categorizes dashboard-camera frames into 10 driving activities — from safe driving to texting, phone use, and reaching behind.

What I Did
•  ⁠Preprocessed and augmented dashboard-camera images for training
•  ⁠Designed a custom CNN with 4 convolutional layers (~6.7M parameters)
•  ⁠Applied transfer learning using MobileNetV2, VGG16, and ResNet50
•  ⁠Evaluated all models with per-class F1 scores and confusion matrices

Tools Used
•  ⁠Python, TensorFlow, Keras
•  ⁠NumPy, pandas
•  ⁠Matplotlib, Seaborn
•  ⁠Jupyter Notebook

How to Run
1.  ⁠Clone this repository
2.  ⁠Install dependencies: pip install numpy pandas matplotlib seaborn scikit-learn
3.  ⁠Download the State Farm dataset from Kaggle
4.  ⁠Open the notebook and run all cells

Key Findings
•  ⁠The custom CNN achieved 100% accuracy on the verification set
•  ⁠Transfer learning models converged faster with fewer training epochs
•  ⁠Per-class F1 scores showed strong performance across all 10 activity categories

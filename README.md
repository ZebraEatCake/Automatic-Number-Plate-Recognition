📁 Project Structure
cv_summonsystem/

│

├── data                  # Dataset of license plates images used for training and testing

│   ├── unseen/           # Place your own test images here

├── best_model.pth        # Best RCNN model

├── datasets.py           #Checks for annotations labeled as "License Plate" and parses them into tensor format for model training.

├── infer_and_ocr.py      # Inference script to run detection + OCR + summons check

├── main.py               # Main program

├── requirements.txt      # Dependencies required to run the program

├── testing.py            # Evaluation of the Faster R-CNN model

├── training.py           # Training of the Faster R-CNN model

🧠 Model Overview
main.py
This script is responsible for training and testing the Faster R-CNN model with a MobileNetV3 backbone for license plate detection. It is not involved in OCR or data lookup. Use this if you want to retrain or fine-tune the model.

infer_and_ocr.py
This is the main pipeline script. It:

1. Loads the trained model
2. Accepts an input folder (like data/unseen)
3. Runs license plate detection
4. Performs OCR on detected plates
5. Matches against a simulated summons list
6. Displays predictions and labels on the image

▶️ How to Run Inference
To run the full detection + OCR + summons check pipeline on your own images:

1. Add Your Data
Place any test images in the folder: ./data/unseen
Make sure images are in .jpg, .jpeg, or .png format.

2. Install Dependencies
Open a command prompt inside the main cv_summonsystem folder and run:

pip install -r requirements.txt

3. Run Inference
Run the following command:

python infer_and_ocr.py --folder ./data/unseen --model best_model.pth

🔍 Notes

OCR Engine: Uses EasyOCR with grayscale + Otsu thresholding for better text extraction from license plates.

Regex Matching: Ensures the extracted text matches Malaysian license plate formats.

Summons Database: A simulated hardcoded list representing plates with active violations.

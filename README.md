**Face Recognition using VGG16 Convolutional Neural Network**

This project is aimed at recognizing faces in images using a VGG16 convolutional neural network (CNN) architecture. The model is trained on a dataset containing images of faces belonging to different individuals. Once trained, the model can predict the identity of a person in a given image.

### Project Structure

- **train/**: Directory containing the training images organized into subdirectories based on class labels (individual identities).
- **validate/**: Directory containing validation images for evaluating the model's performance.
- **face_recognition_vgg16.h5**: Trained model saved in HDF5 format.

### Dependencies

- Python 3.x
- TensorFlow 2.x
- OpenCV (cv2)
- Matplotlib
- Numpy
- Google Colab (for running the code in a Colab environment)

### Instructions

1. **Dataset Preparation**: Organize your face images into a directory structure where each subdirectory corresponds to a different individual, and each subdirectory contains images of that individual.
2. **Training**: Run the provided Python script in a suitable environment (e.g., Google Colab) to train the VGG16 model on the prepared dataset. Make sure to adjust the paths and parameters as necessary.
3. **Testing**: After training, the model can be used to recognize faces in new images. Update the `image_path` variable in the script with the path to the image you want to test. Then, run the script to detect and recognize faces in the image.

### How to Use

1. Clone this repository to your local machine.
2. Prepare your face dataset and organize it as described above.
3. Upload your dataset to Google Drive or use any other suitable storage.
4. Open the provided Python script (`face_recognition_vgg16.py`) and make necessary adjustments such as paths, parameters, and hyperparameters.
5. Run the script in a Python environment with the required dependencies installed.
6. After training, use the trained model to recognize faces in new images by updating the `image_path` variable and running the script.

### Notes

- The project utilizes transfer learning with the pre-trained VGG16 model for feature extraction.
- Data augmentation techniques are employed to enhance model generalization.
- The Haar Cascade classifier is used for face detection in images.
- The model achieves face recognition by predicting the identity of the detected face.

### Author

This project was developed by Suyash Salvi.

Feel free to reach out for any questions, suggestions, or improvements!

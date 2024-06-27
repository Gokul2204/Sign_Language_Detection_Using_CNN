# Sign_Language_Detection_Using_CNN

Implementation of a convolutional neural network (CNN) using TensorFlow and Keras for recognizing American Sign Language (ASL) gestures. Trained on the Sign Language MNIST dataset, this project aims to facilitate communication for the deaf or hard of hearing through real-time interpretation of hand gestures.

1.Overview
This project implements a convolutional neural network (CNN) using TensorFlow and Keras for recognizing letters from American Sign Language (ASL) gestures. The model is trained on the Sign Language MNIST dataset, consisting of grayscale images of hand gestures representing 24 classes (letters A-Z, excluding J and Z which involve motion).
2.Requirements
2.1 Required Libraries
* TensorFlow 2.x
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
2.2 Additional Libraries Used
* warnings: Suppress specific warnings.
* LabelBinarizer from sklearn.preprocessing: For label encoding.
* ImageDataGenerator from tensorflow.keras.preprocessing.image: Used for data augmentation.
* Regularizers from tensorflow.keras: Regularization techniques for model building.


3.Project Structure
* Sign_Language_Model_Batch_64_Epoch_10_V03.ipynb: Jupyter Notebook containing complete code for data preprocessing, model building, training, evaluation, and prediction.
* Sign_Language_Model_Batch_64_Epoch_10_V03.h5: Trained model saved in H5 file format.


4.Usage
4.1 Running the Jupyter Notebook
* Clone the repository:
git clone https://github.com/your/repository.git
cd repository
* Install dependencies:
pip install -r requirements.txt
* Open and execute the Jupyter Notebook:
jupyter notebook sign_language_detection.ipynb

Execute each cell in the notebook sequentially to preprocess data, train the model, evaluate performance metrics, and save the model.

4.2 Loading the Pre-trained Model
* To load the pre-trained model and make predictions:
from tensorflow.keras.models import load_model 
# Load the model 
model = load_model('Sign_Language_Model_Batch_64_Epoch_10_V03.h5')

4.3 Customization
* Model Architecture: Modify the CNN architecture in the notebook (sign_language_detection.ipynb) to experiment with different configurations.
* Hyperparameter Tuning: Adjust hyperparameters directly in the notebook for optimal model performance.
* Data Augmentation: Explore different augmentation techniques by modifying the ImageDataGenerator parameters.

4.4 License
        This project is licensed under the MIT License - see the LICENSE file for details.
        
4.5 Dataset Source
* Train → Attached
* Test → Attached

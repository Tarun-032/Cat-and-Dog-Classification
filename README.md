# Cat and Dog Image Classifier

This project is a simple image classifier built using a Support Vector Machine (SVM) to distinguish between cat and dog images. The classification is based on the purchase history of customers in a retail store.

## Features

- **Image Classification:**
  - Utilizes a trained SVM model to classify images into two categories: Cat or Dog.
  
- **Purchase History Clustering:**
  - Implements K-means clustering algorithm to group customers based on their purchase history.

## Technologies Used

- **Programming Language:**
  - Python
  
- **Libraries:**
  - scikit-learn
  - NumPy
  - Flask (for web interface)
  
- **Machine Learning Models:**
  - Support Vector Machine (SVM)
  - K-means Clustering

## Usage

### Image Classification:

1. **Run the Flask App:**
   - Navigate to the `app` directory.
   - Execute `python app.py` to start the Flask web application.

2. **Upload Image:**
   - Visit `http://localhost:5000/` in your web browser.
   - Upload an image using the provided form.

3. **Get Classification Result:**
   - Click the "Predict" button to get the classification result.

### Purchase History Clustering:

1. **Run the Clustering Script:**
   - Open the Jupyter notebook `svmpreprocess.ipynb` to run the K-means clustering algorithm on customer data.

2. **Visualize Clusters:**
   - The notebook includes visualizations such as pair plots and scatter plots to observe customer clusters.

## Future Enhancements

- **Improvement of Image Classifier:**
  - Fine-tune the SVM model for better accuracy.
  - Explore other advanced image classification models (e.g., CNN).

- **Enhancement of Clustering Algorithm:**
  - Experiment with different clustering algorithms.
  - Improve visualization and interpretation of customer clusters.

## Contributions

Contributions are welcome! If you have any suggestions or improvements, feel free to fork the repository, open issues, and submit pull requests.

## Acknowledgments

This project was developed as part of learning and exploration in machine learning and image processing.


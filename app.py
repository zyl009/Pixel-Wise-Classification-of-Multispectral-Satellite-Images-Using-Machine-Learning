# This project is licensed under the MIT License
# See the LICENSE file in the project root for more information.
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import rasterio
from PIL import Image
import io
import base64
from io import BytesIO
from multiprocessing import Pool
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Set the maximum upload file size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Optional for cache control
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60  # Set a longer session lifetime 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        # Save the image to memory
        image_data = BytesIO(image.read())
        # Reading TIFF images using rasterio
        with rasterio.open(image_data) as src:
            bands = src.read()

            img_data = np.moveaxis(bands, 0, -1)  # Move the channel axis to the end

            # Contrast stretch
            p2, p98 = np.percentile(img_data, (2, 98))
            img_data = np.clip(img_data, p2, p98)
            img_data = (img_data - p2) / (p98 - p2) * 255
            img_data = img_data.astype(np.uint8)

            # Convert to PNG and return image as base64
            img = Image.fromarray(img_data)
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()

        return jsonify({'image_data': img_base64})

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    samples = data['samples']
    image_base64 = data['image_data']

    # Decode base64 image to numpy array
    image_data = base64.b64decode(image_base64)
    image_io = io.BytesIO(image_data)
    with Image.open(image_io) as img:
        img_array = np.array(img)

    # Adjust image dimensions
    bands = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    height, width = bands.shape[1:]

    # Extract sample data
    X, y = [], []
    labels = []
    for sample in samples:
        polygon = Polygon([(pt['x'], pt['y']) for pt in sample['vertices']])
        label = sample['label']
        if label not in labels:
            labels.append(label)
        label_idx = labels.index(label)

        for y_coord in range(height):
            for x_coord in range(width):
                if polygon.contains(Point(x_coord, y_coord)):
                    pixel_value = bands[:, y_coord, x_coord]
                    if not np.all(pixel_value == 0):  # Filter black pixels
                        X.append(pixel_value)
                        y.append(label_idx)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on training data
    X_test = scaler.transform(X_test)        # Transform test data

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, y_train)

    # Flatten all pixels for classification
    pixels = bands.reshape(bands.shape[0], -1).T
    mask = np.all(pixels == 0, axis=1)  # Identify black pixels
    valid_pixels = pixels[~mask]
    # Scale all pixels
    valid_pixels_scaled = scaler.transform(valid_pixels)

    # Classify valid pixels
    classified_pixels = knn.predict(valid_pixels_scaled)
    classified_image = np.zeros(pixels.shape[0], dtype=np.int32)
    classified_image[~mask] = classified_pixels + 1  # Labels start from 1
    classified_image = classified_image.reshape(height, width)

    # Test predictions
    y_pred = knn.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(labels)))
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_sum[cm_sum == 0] = 1  # Prevent division by zero
    cm_normalized = cm.astype('float') / cm_sum
    classification_report_text = classification_report(y_test, y_pred, target_names=labels)

    # Create confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=".2f")
    # Add axis labels for clarity
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.title('Normalized Confusion Matrix')

    # Save confusion matrix as base64 image
    cm_img_io = io.BytesIO()
    plt.savefig(cm_img_io, format='PNG')
    cm_img_io.seek(0)
    cm_img_base64 = base64.b64encode(cm_img_io.getvalue()).decode()
    plt.close()

    # Return supervised classification result as base64
    num_classes = len(labels)
    colormap = matplotlib.colormaps['tab10']
    colors = np.vstack(([0, 0, 0, 1], colormap(np.linspace(0, 1, num_classes))))
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(classified_image, cmap=cmap, vmin=0, vmax=num_classes)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, num_classes + 1))
    cbar.ax.set_yticklabels(['Background'] + labels)
    plt.title("Supervised Classified Image")

    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode()
    plt.close()

    # Return classification report, accuracy, and confusion matrix as JSON
    return jsonify({
        'classified_image': img_base64,
        'classification_report': classification_report_text,
        'confusion_matrix_image': cm_img_base64
    })




@app.route('/unsupervised_classify', methods=['POST'])
def unsupervised_classify():
    data = request.json
    num_classes = int(data['num_classes'])  
    image_base64 = data['image_data']

    # Decode base64 image into numpy array
    image_data = base64.b64decode(image_base64)
    image_io = BytesIO(image_data)
    with Image.open(image_io) as img:
        img_array = np.array(img)

    bands = img_array.transpose(2, 0, 1)  # Adjust the dimensions to resemble the original GeoTIFF data
    pixels = bands.reshape(bands.shape[0], -1).T

    mask = np.all(pixels == 0, axis=1)

    # KMeans classification only for non-black areas
    valid_pixels = pixels[~mask]
    scaler = StandardScaler()
    valid_pixels_scaled = scaler.fit_transform(valid_pixels)
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    kmeans.fit(valid_pixels_scaled)


    classified_image = np.zeros(pixels.shape[0], dtype=np.int32)
    classified_image[~mask] = kmeans.labels_ + 1  # Label class from 1 instead of 0
    classified_image = classified_image.reshape(bands.shape[1], bands.shape[2])

    colormap = matplotlib.colormaps['tab10']
    colors = np.vstack(([0, 0, 0, 1], colormap(np.linspace(0, 1, num_classes))))
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(classified_image, cmap=cmap, vmin=0, vmax=num_classes)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, num_classes + 1))
    cbar.ax.set_yticklabels([f'Class {i}' for i in range(num_classes + 1)])
    plt.title("Unsupervised Classified Image")

    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode()
    plt.close()

    return jsonify({'classified_image': img_base64})

if __name__ == '__main__':
    port = os.getenv('PORT', 5000)
    app.run(host='0.0.0.0', port=port)

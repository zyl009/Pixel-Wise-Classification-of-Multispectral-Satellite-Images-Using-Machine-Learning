# Pixel-Wise Classification of Multispectral Satellite Images Using Machine Learning

## Project Description
This project, titled "Pixel-Wise Classification of Multispectral Satellite Images Using Machine Learning", presents a cloud-based application for pixel-wise classification of multispectral satellite images using machine learning. It supports both supervised (KNN with Euclidean distance) and unsupervised (K-means) classification, enabling users to upload GeoTIFF images, select training samples, and classify land cover types. The application features a user-friendly interface designed to simplified the process of image classification.

## Features

- **GeoTIFF Image Upload**: Easily upload multispectral GeoTIFF images for analysis.
- **Training Sample Selection**: Interactively select training samples for supervised classification.
- **Supervised Classification**: Utilize K-Nearest Neighbors algorithm with Euclidean distance for accurate land cover classification.
- **Unsupervised Classification**: Apply K-means clustering to identify natural groupings in the data.
- **Visualization**: View and analyze classification results directly within the application.

## Installation

The application has been published via Microsoft Azure. You can click the [link](https://satellite-image-classification-cpguc5dgbghmcyfd.germanywestcentral-01.azurewebsites.net/) to access the application directly. However, because it uses a free server, the performance of the web version is not very good. For a better experience, it is recommended to use the following method.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zyl009/Pixel-Wise-Classification-of-Multispectral-Satellite-Images-Using-Machine-Learning.git
   cd Pixel-Wise-Classification-of-Multispectral-Satellite-Images-Using-Machine-Learning
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the Application**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your web browser and navigate to `http://localhost:5000`.

3. **Upload and Classify Images**:
   - Upload your multispectral GeoTIFF image.
   - Choose the desired classification method (KNN or K-means).
   - For supervised classification, select training samples on the image. For unsupervised classification, input the number of classes.
   - View and analyze the resulting classified image.

See the [report](report.pdf) and [slides](projrct_slides.pptx) file for more details.

## Dependencies
This project utilizes the following open-source libraries:

| Library                | Description                                    | License Type      |
|------------------------|------------------------------------------------|-------------------|
| **Flask**              | Micro web framework for Python                | BSD License       |
| **NumPy**              | Library for numerical computations            | BSD License       |
| **Rasterio**           | Access and process raster data (e.g., GeoTIFF)| BSD License       |
| **Pillow (PIL)**       | Image processing library                      | HPND License      |
| **Matplotlib**         | Plotting and visualization library            | PSF/BSD License   |
| **Shapely**            | Geometric objects and operations              | BSD License       |
| **Scikit-learn**       | Machine learning library                      | BSD License       |
| **Base64** (built-in)  | Encoding and decoding base64 strings          | Python Standard   |
| **io** (built-in)      | Core input/output functionality               | Python Standard   |
- **Pillow**: Copyright © 2010-2024 by Alex Clark and contributors. Licensed under the HPND License.

Each library is subject to its respective license.Refer to their documentation for more details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author
Liang Zhongyou – [GitHub Profile](https://github.com/zyl009)
---



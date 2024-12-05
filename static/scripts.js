/* 
This project is licensed under the MIT License
See the LICENSE file in the project root for more information.
*/
let imagePath = '';
let map;
let drawnItems = new L.FeatureGroup();
let unsupervisedMap;
const samples = [];
let currentLabel = null;  
const labelMap = {};  // store the mapping relationship between labels and layers

document.getElementById('imageInput').addEventListener('change', function() {
    const file = this.files[0];
    // Check file size, limit is 10MB
    const maxSize = 10 * 1024 * 1024;  
    if (file.size > maxSize) {
        alert("Uploaded image is too large. Maximum size is 10 MB.");
        return;  
    }
    const reader = new FileReader();
    reader.onload = function(event) {
        const formData = new FormData();
        formData.append('image', file);
        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
          .then(data => {
            const imageBase64 = data.image_data;
            imagePath = `data:image/png;base64,${imageBase64}`;

              if (map) {
                  map.remove();
              }

              map = L.map('map', {
                  crs: L.CRS.Simple,
                  minZoom: -5,
                  zoomControl: true
              });

              const bounds = [[0, 0], [600, 800]]; // Adjust the bounds based on the image dimensions
              L.imageOverlay(imagePath, bounds).addTo(map);
              map.fitBounds(bounds);

              map.addLayer(drawnItems);
              const drawControl = new L.Control.Draw({
                  edit: {
                      featureGroup: drawnItems,
                      remove: true // Enable delete feature
                  },
                  draw: {
                      polygon: true,  // Enable polygon drawing
                      polyline: false,
                      circle: false,
                      rectangle: false,
                      marker: false,
                      circlemarker: false
                  }
              });
              map.addControl(drawControl);

              map.on(L.Draw.Event.CREATED, function(event) {
                  const layer = event.layer;
                  drawnItems.addLayer(layer);

                  if (currentLabel) {
                      // Collect vertices and assign the current label
                      const vertices = layer.getLatLngs()[0].map(point => ({ x: Math.floor(point.lng), y: Math.floor(point.lat) }));
                      samples.push({ vertices: vertices, label: currentLabel });

                      // Associating a layer with a label
                      if (!labelMap[currentLabel]) {
                          labelMap[currentLabel] = [];
                      }
                      labelMap[currentLabel].push(layer);
                  } else {
                      alert("Please enter a class label before drawing polygons.");
                      drawnItems.removeLayer(layer);
                  }
              });

              map.on(L.Draw.Event.DELETED, function(event) {
                  const layers = event.layers;
                  layers.eachLayer(function(layer) {
                      const layerVertices = layer.getLatLngs()[0].map(point => ({ x: Math.floor(point.lng), y: Math.floor(point.lat) }));

                      // Remove the corresponding sample from the samples array
                      const index = samples.findIndex(sample => JSON.stringify(sample.vertices) === JSON.stringify(layerVertices));
                      if (index !== -1) {
                          // Find the corresponding tag
                          const label = samples[index].label;
                          // Delete the mapping between layers and labels
                          labelMap[label] = labelMap[label].filter(l => l !== layer);
                          // If there is no layer under the label, remove it from labelMap
                          if (labelMap[label].length === 0) {
                              delete labelMap[label];
                              // Remove a tag from a page
                              document.getElementById(`label-${label}`).remove();
                          }
                          samples.splice(index, 1);
                      }
                  });
              });
          });
    }
    reader.readAsDataURL(file);
});

// Handle label input
document.getElementById('saveLabelButton').addEventListener('click', function() {
    currentLabel = document.getElementById('labelInput').value;
    if (currentLabel) {
        if (!labelMap[currentLabel]) {
            // Display labels on the page
            const labelContainer = document.createElement('div');
            labelContainer.id = `label-${currentLabel}`;
            labelContainer.innerHTML = `
                <span>${currentLabel}</span>
                <button onclick="deleteLabel('${currentLabel}')">Delete</button>
            `;
            document.getElementById('labelsList').appendChild(labelContainer);
            alert(`Current label is set to: ${currentLabel}. You can now draw polygons.`);
        } else {
            alert(`The label ${currentLabel} is already in use. You can continue drawing polygons with this label or delete it.`);
        }
    } else {
        alert("Please enter a class label.");
    }
});

// Delete a label and its associated polygon
function deleteLabel(label) {
    if (labelMap[label]) {
        // Delete all layers associated with this tag
        labelMap[label].forEach(layer => {
            drawnItems.removeLayer(layer);
            // Delete the corresponding polygon in samples
            const layerVertices = layer.getLatLngs()[0].map(point => ({ x: Math.floor(point.lng), y: Math.floor(point.lat) }));
            const index = samples.findIndex(sample => JSON.stringify(sample.vertices) === JSON.stringify(layerVertices));
            if (index !== -1) {
                samples.splice(index, 1);
            }
        });
        // Remove a tag from a mapping
        delete labelMap[label];
        // Remove a tag from a page
        document.getElementById(`label-${label}`).remove();
        alert(`Label "${label}" and its associated polygons have been deleted.`);
    } else {
        alert(`Label "${label}" not found.`);
    }
}

document.getElementById('classifyButton').addEventListener('click', function() {
    // Show classification progress prompt
    const loadingMessage = document.createElement('a');
    loadingMessage.innerHTML = 'Image classification is in progress, please wait!';
    document.getElementById('loadingMessage').appendChild(loadingMessage);

    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            samples: samples,
            image_data: imagePath.split(',')[1]  // Only pass the base64 data part
        })
    }).then(response => response.json())
      .then(data => {
        const classifiedImageBase64 = data.classified_image;
        const classifiedImagePath = `data:image/png;base64,${classifiedImageBase64}`;

        // Clear category progress prompt
        if (loadingMessage) {
            loadingMessage.remove();
        }

        // Display the classification result image
        const classifiedMap = L.map('classifiedMap', {
            crs: L.CRS.Simple,
            minZoom: -5,
            zoomControl: true
        });

        const bounds = [[0, 0], [600, 1000]]; // Adjust bounds to match image size
        L.imageOverlay(classifiedImagePath, bounds).addTo(classifiedMap);
        classifiedMap.fitBounds(bounds);
        // Create Download Link
        const downloadLink = document.createElement('a');
        downloadLink.href = classifiedImagePath;
        downloadLink.download = 'supervised_classified_image.png';
        downloadLink.innerHTML = "Download Supervised Classified Image";
        document.getElementById('downloadArea1').appendChild(downloadLink);

        
        // Create a title "Classification Report"
        const reportTitle = document.createElement('h3');
        reportTitle.innerHTML = 'Supervised Classification Report';
        document.getElementById('classificationReport').appendChild(reportTitle);
        // Show Classification Report
        const reportDiv = document.createElement('pre'); // Use <pre> to format the report
        reportDiv.innerHTML = data.classification_report;
        reportDiv.style.fontSize = '16px'; 
        reportDiv.style.lineHeight = '1.5'; 
        document.getElementById('classificationReport').appendChild(reportDiv);

        // Display confusion matrix image
        const cmImageBase64 = data.confusion_matrix_image;
        const cmImagePath = `data:image/png;base64,${cmImageBase64}`;

        const cmImageElement = document.createElement('img');
        cmImageElement.src = cmImagePath;
        cmImageElement.alt = 'Confusion Matrix';
        cmImageElement.style.width = '40%'; // Resize the confusion matrix image
        cmImageElement.style.marginTop = '20px'; 
        document.getElementById('Confusion Matrix').appendChild(cmImageElement);
        // Create a link to download the classification report
        const downloadReportLink = document.createElement('a');
        downloadReportLink.innerHTML = "Download Classification Report";
        downloadReportLink.style.display = 'block';
        const reportBlob = new Blob([`Classification Report:\n\n${data.classification_report}`], { type: 'text/plain' });
        const reportUrl = URL.createObjectURL(reportBlob);
        downloadReportLink.href = reportUrl;
        downloadReportLink.download = 'classification_report.txt';
        document.getElementById('downloadArea3').appendChild(downloadReportLink);

        
      })
      .catch(error => {
          console.error('Error during classification:', error);
          if (loadingMessage) {
              loadingMessage.remove();
          }
      });
});


document.getElementById('unsupervisedClassifyButton').addEventListener('click', function() {
    const numClasses = document.getElementById('numClassesInput').value;
    if (numClasses) {
        fetch('/unsupervised_classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_classes: numClasses,
                image_data: imagePath.split(',')[1]  
            })
        }).then(response => response.json())
          .then(data => {
            const classifiedImageBase64 = data.classified_image;
            const classifiedImagePath = `data:image/png;base64,${classifiedImageBase64}`;


              if (unsupervisedMap) {
                  unsupervisedMap.remove();
              }

              // Display the classified image
              unsupervisedMap = L.map('unsupervisedMap', {
                  crs: L.CRS.Simple,
                  minZoom: -5,
                  zoomControl: true
              });

              const bounds = [[0, 0], [600, 1000]]; // Adjust bounds to match image size
              L.imageOverlay(classifiedImagePath, bounds).addTo(unsupervisedMap);
              unsupervisedMap.fitBounds(bounds);
              // create downloadlink
              const downloadLink = document.createElement('a');
              downloadLink.href = classifiedImagePath;
              downloadLink.download = classifiedImagePath.split('/').pop();
              downloadLink.innerHTML = "Download Unsupervised Classified Image";
              document.getElementById('downloadArea2').appendChild(downloadLink);
          })
          .catch(error => {
              console.error('Error during unsupervised classification:', error);
          });
    } else {
        alert("Please enter the number of classes.");
    }
});

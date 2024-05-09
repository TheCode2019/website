function sendImage() {
    const imageInput = document.getElementById('imageInput');
    const resultsDiv = document.getElementById('ocrResults');

    if (imageInput.files.length === 0) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', imageInput.files[0]);

    resultsDiv.innerHTML = 'Processing...';

    // Update the URL to point to your Google Cloud Run app
    fetch('https://test-wpek6upsvq-nw.a.run.app/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultsDiv.innerHTML = 'Error: ' + data.error;
        } else {
            resultsDiv.innerHTML = 'OCR Results: ' + data.text;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultsDiv.innerHTML = 'Failed to retrieve OCR results.';
    });
}

function toggleMenu() {
    var menu = document.getElementById('dropdownMenu');
    if (menu.style.display === 'block') {
        menu.style.display = 'none';
    } else {
        menu.style.display = 'block';
    }
}



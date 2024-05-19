function getApiBaseUrl() {
    const hostname = window.location.hostname;

    // Define URLs based on the hostname
    if (hostname === "localhost" || hostname === "127.0.0.1") {
        return "http://localhost:5000"; // Local development 
    } else if (hostname === "test-wpek6upsvq-nw.a.run.app") {
        return "https://test-wpek6upsvq-nw.a.run.app/"; // Production backend URL (i will break it one day)
    } else if (hostname === "neoquid.web.app") {
        return "https://https://neoquid.web.app/"; // this is the one
    }
    // Default to production if no match
    return "https://https://neoquid.web.app/"; 
}

function sendImage() {
    const imageInput = document.getElementById('imageInput');
    const resultsDiv = document.getElementById('ocrResults');

    if (imageInput.files.length === 0) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', imageInput.files[0]);
    const apiUrl = getApiBaseUrl();

    resultsDiv.innerHTML = 'Processing...';

    // Append the specific endpoint to the base URL
    fetch(`${apiUrl}/upload`, {
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

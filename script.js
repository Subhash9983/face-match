/**
 * Video Face Matching System - Frontend Logic (Integrated)
 */

document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = "http://localhost:8000/api";
    
    // Elements - Photos
    const photoUpload = document.getElementById('photo-upload');
    const photoPreviewGrid = document.getElementById('photo-preview-grid');
    
    // Elements - Video
    const videoUpload = document.getElementById('video-upload');
    const videoPreviewContainer = document.getElementById('video-preview-container');
    const videoPlayer = document.getElementById('video-player');
    const faceOverlay = document.getElementById('face-overlay');
    
    // Elements - Actions
    const startBtn = document.getElementById('start-matching-btn');

    let matchResults = []; // To store data from backend
    let animationFrameId = null;

    /**
     * Helper: Show status message
     */
    const showStatus = (msg, isError = false) => {
        console.log(msg);
        if (isError) alert(msg);
    };

    /**
     * Handle Photo Uploads to Backend
     */
    photoUpload.addEventListener('change', async (event) => {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
            
            // Local preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const item = document.createElement('div');
                item.className = 'preview-item';
                const img = document.createElement('img');
                img.src = e.target.result;
                item.appendChild(img);
                photoPreviewGrid.appendChild(item);
            };
            reader.readAsDataURL(file);
        });

        try {
            const response = await fetch(`${API_BASE}/upload-photos`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            showStatus("Photos uploaded to server.");
        } catch (error) {
            showStatus("Error uploading photos.", true);
        }
    });

    /**
     * Handle Video Upload to Backend
     */
    videoUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Local Preview
        const videoURL = URL.createObjectURL(file);
        videoPlayer.src = videoURL;
        videoPreviewContainer.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            startBtn.disabled = true;
            startBtn.textContent = "Uploading Video...";
            const response = await fetch(`${API_BASE}/upload-video`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            showStatus("Video uploaded to server.");
        } catch (error) {
            showStatus("Error uploading video.", true);
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = "Start Face Matching";
        }
    });

    /**
     * Rendering logic: Draw boxes based on video timestamp
     */
    const renderBoxes = () => {
        if (videoPlayer.paused || videoPlayer.ended) {
            cancelAnimationFrame(animationFrameId);
            return;
        }

        const currentTimeMs = videoPlayer.currentTime * 1000;
        
        // Find the closest frame result from the backend data
        // backend results are stored with 'millisecond' key
        const frameData = matchResults.find(r => Math.abs(r.millisecond - currentTimeMs) < 100);

        faceOverlay.innerHTML = ''; // Clear previous boxes

        if (frameData) {
            frameData.faces.forEach(face => {
                const box = document.createElement('div');
                box.className = 'face-box';
                
                // Backend bbox is [x1, y1, x2, y2] in pixels
                // We need to convert to percentage for responsive overlay
                // Note: This assumes overlay matches video dimensions exactly
                const videoWidth = videoPlayer.videoWidth;
                const videoHeight = videoPlayer.videoHeight;

                const x1 = (face.bbox[0] / videoWidth) * 100;
                const y1 = (face.bbox[1] / videoHeight) * 100;
                const width = ((face.bbox[2] - face.bbox[0]) / videoWidth) * 100;
                const height = ((face.bbox[3] - face.bbox[1]) / videoHeight) * 100;

                box.style.left = `${x1}%`;
                box.style.top = `${y1}%`;
                box.style.width = `${width}%`;
                box.style.height = `${height}%`;

                const similarity = (face.similarity * 100).toFixed(1);
                const tag = document.createElement('div');
                tag.className = 'similarity-tag';
                
                // Use backend provided status and match flag
                if (face.is_matched) {
                    box.classList.add('matched');
                    tag.innerHTML = `<span>${face.status}</span>${face.name}<br>Similarity: ${similarity}%`;
                } else {
                    tag.innerHTML = `<span>${face.status}</span>Similarity: ${similarity}%`;
                }
                
                box.appendChild(tag);
                faceOverlay.appendChild(box);
            });
        }

        animationFrameId = requestAnimationFrame(renderBoxes);
    };

    /**
     * Handle Start Button Click
     */
    startBtn.addEventListener('click', async () => {
        if (photoPreviewGrid.children.length === 0 || videoPreviewContainer.classList.contains('hidden')) {
            alert("Please upload photos and a video first!");
            return;
        }

        try {
            startBtn.disabled = true;
            startBtn.textContent = "Processing (Server Side)...";
            
            const response = await fetch(`${API_BASE}/start-matching`, { method: 'POST' });
            const data = await response.json();

            if (data.status === "success") {
                matchResults = data.results;
                videoPlayer.play();
                startBtn.textContent = "Matching Results Playing";
                renderBoxes();
            } else {
                throw new Error(data.detail || "Processing failed");
            }
        } catch (error) {
            alert("Error: " + error.message);
            startBtn.disabled = false;
            startBtn.textContent = "Start Face Matching";
        }
    });

    videoPlayer.addEventListener('ended', () => {
        startBtn.disabled = false;
        startBtn.textContent = "Start Face Matching";
        faceOverlay.innerHTML = '';
        cancelAnimationFrame(animationFrameId);
    });
});

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
    const downloadResultsBtn = document.getElementById('download-results-btn');
    const downloadVideoBtn = document.getElementById('download-video-btn');

    let matchResultsData = null; // Store full results object
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
     * Rendering logic: Handled by processed video from server now.
     */

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
                matchResultsData = data.results;
                console.log("Attendance Summary:", data.results.summary);
                
                // Switch to processed video which has boxes/labels burned in
                const processedVideoUrl = `http://localhost:8000${data.results.video_url}?t=${Date.now()}`;
                videoPlayer.src = processedVideoUrl;
                videoPlayer.load();
                videoPlayer.play();

                startBtn.textContent = "Matching Results Playing";
                downloadResultsBtn.classList.remove('hidden');
                downloadVideoBtn.classList.remove('hidden');
                faceOverlay.innerHTML = ''; // Ensure overlay is empty
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

    /**
     * Download Results as JSON
     */
    downloadResultsBtn.addEventListener('click', () => {
        if (!matchResultsData) return;
        
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(matchResultsData, null, 4));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "face_match_results.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    });

    /**
     * Download Processed Video (Using dedicated download endpoint)
     */
    downloadVideoBtn.addEventListener('click', () => {
        // Use window.location.href - works correctly for cross-origin (port:3000 -> port:8000)
        // The backend sends Content-Disposition: attachment header which forces download
        window.location.href = `${API_BASE}/download-processed-video`;
        showStatus("Download initiated...");
    });
});

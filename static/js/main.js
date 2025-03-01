// Preview image upload
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.querySelector('input[type="file"]');
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewContainer = document.createElement('div');
                    previewContainer.className = 'mt-3';
                    previewContainer.innerHTML = `
                        <p>Preview:</p>
                        <img src="${e.target.result}" style="max-width: 100%; max-height: 300px;" class="img-thumbnail">
                    `;
                    
                    // Remove existing preview if any
                    const existingPreview = document.querySelector('.mt-3');
                    if (existingPreview) {
                        existingPreview.remove();
                    }
                    
                    // Add new preview
                    imageInput.parentNode.appendChild(previewContainer);
                }
                reader.readAsDataURL(file);
            }
        });
    }
});

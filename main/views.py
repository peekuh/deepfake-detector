from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegisterForm, ImageUploadForm
from .models import UploadedImage
from .utils import analyze_image

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

@login_required
def dashboard(request):
    user_images = UploadedImage.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'dashboard.html', {'images': user_images})

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.user = request.user
            image.save()
            
            # Analyze the uploaded image
            try:
                is_deepfake, confidence, heatmap_content, frequency_content, high_pass_content = analyze_image(image.image.path)
                
                # Update the image record with analysis results
                image.is_analyzed = True
                image.is_deepfake = is_deepfake
                image.confidence_score = confidence
                
                # Save the heatmap if generated
                if heatmap_content:
                    filename = f"heatmap_{image.id}.jpg"
                    image.heatmap_image.save(filename, heatmap_content)
                
                # Save the frequency analysis if generated
                if frequency_content:
                    filename = f"frequency_{image.id}.jpg"
                    image.frequency_image.save(filename, frequency_content)
                
                # Save the high-pass filter image if generated
                if high_pass_content:
                    filename = f"high_pass_{image.id}.jpg"
                    image.high_pass_image.save(filename, high_pass_content)
                
                image.save()
                
                messages.success(
                    request, 
                    f'Image analyzed successfully! {"This appears to be a deepfake" if is_deepfake else "This appears to be authentic"} (confidence: {confidence:.2%})'
                )
            except Exception as e:
                messages.warning(request, f'Image uploaded but analysis failed: {str(e)}')
                print(f"Error: {str(e)}")
            
            return redirect('dashboard')
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})


@login_required
def delete_image(request, image_id):
    """
    Deletes an image uploaded by the user.
    """
    image = get_object_or_404(UploadedImage, pk=image_id, user=request.user)  # Get the image or return a 404 if not found

    if request.method == 'POST':  # Only delete if the request is a POST request
        image.delete()
        messages.success(request, 'Image deleted successfully!')
        return redirect('dashboard')  # Redirect back to the dashboard

    # If it's a GET request, you might want to show a confirmation page
    # (optional, but good practice)
    return render(request, 'confirm_delete.html', {'image': image})  # Create a template named confirm_delete.html

@login_required
def image_analysis(request, image_id):
    """
    Display a detailed analysis of an image, including the Grad-CAM heatmap.
    """
    image = get_object_or_404(UploadedImage, pk=image_id, user=request.user)  # Ensure user owns the image
    
    # If the image hasn't been analyzed yet, analyze it now
    if not image.is_analyzed:
        try:
            is_deepfake, confidence, heatmap_content, frequency_content, high_pass_content = analyze_image(image.image.path)
            
            image.is_analyzed = True
            image.is_deepfake = is_deepfake
            image.confidence_score = confidence
            
            if heatmap_content:
                filename = f"heatmap_{image.id}.jpg"
                image.heatmap_image.save(filename, heatmap_content)
            
            if frequency_content:
                filename = f"frequency_{image.id}.jpg"
                image.frequency_image.save(filename, frequency_content)
            
            if high_pass_content:
                filename = f"high_pass_{image.id}.jpg"
                image.high_pass_image.save(filename, high_pass_content)
            
            image.save()
            
            messages.success(request, 'Image analysis completed.')
        except Exception as e:
            messages.warning(request, f'Analysis failed: {str(e)}')
            print(f"Error analyzing image: {str(e)}")
    
    # Calculate percentage for display
    confidence_percentage = None
    if image.confidence_score is not None:
        confidence_percentage = f"{image.confidence_score * 100:.2f}%"
    
    return render(request, 'image_analysis.html', {
        'image': image,
        'confidence_percentage': confidence_percentage
    })

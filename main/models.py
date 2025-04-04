from django.db import models
from django.contrib.auth.models import User


class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_analyzed = models.BooleanField(default=False)
    is_deepfake = models.BooleanField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)  # Store the confidence score
    heatmap_image = models.ImageField(upload_to='heatmaps/', null=True, blank=True)  # Add this field
    
    def __str__(self):
        return f"{self.user.username}'s image ({self.uploaded_at})"
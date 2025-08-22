import base64
from django.contrib import admin
from .models import ApprovedFace

# Define the ApprovedFaceAdmin class
class ApprovedFaceAdmin(admin.ModelAdmin):
    list_display = ('name', 'added_at', 'display_embedding')  # Show the name, timestamp, and embedding
    search_fields = ('name',)  # Allow searching by name

    # Optionally, add the embedding to the admin form (useful for debugging or viewing)
    readonly_fields = ('embedding',)  # Make 'embedding' field read-only (as it's binary data)

    # Display the embedding as a base64 string (for readability)
    def display_embedding(self, obj):
        # Convert the binary embedding to base64 string
        return base64.b64encode(obj.embedding).decode('utf-8')[:50]  # Display the first 50 characters of the base64 encoding

    display_embedding.short_description = 'Embedding Preview'  # Change the name of the column in the admin panel

# Register the ApprovedFace model and the ApprovedFaceAdmin class
admin.site.register(ApprovedFace, ApprovedFaceAdmin)

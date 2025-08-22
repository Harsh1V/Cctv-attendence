from django.db import models
import pickle  # For serializing and deserializing face embeddings
import torch
import numpy as np

# Model to store approved faces and their embeddings
class ApprovedFace(models.Model):
    # Name of the person (e.g., John Doe)
    name = models.CharField(max_length=255)
    
    pose = models.CharField(max_length=16, default="front")         # "front", "left", "right"
 
    # Store the serialized face embedding as binary data
    embedding = models.BinaryField()

    # Timestamp when the face was added
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def get_embedding(self):
        """
        Deserialize the embedding and return it as a numpy array (or tensor).
        """
        return pickle.loads(self.embedding)

    def save_embedding(self, embedding):
        """
        Serialize the embedding and save it to the database.
        """
        # Ensure the embedding is a NumPy array (if it's a PyTorch tensor, move to CPU and convert)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()  # Convert tensor to NumPy array if needed
        
        # Check if it's a valid NumPy array before serialization
        if isinstance(embedding, np.ndarray):
            self.embedding = pickle.dumps(embedding)  # Serialize the numpy array
            self.save()  # Save the instance with the serialized embedding
            print(f"✅ {self.name} embedding saved.")
        else:
            print(f"❌ Failed to save embedding for {self.name}: Not a valid NumPy array or tensor.")
            raise ValueError("Embedding must be a valid NumPy array or PyTorch tensor.")





        
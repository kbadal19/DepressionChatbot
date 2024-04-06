from django.db import models
from django.contrib.auth.models import User
import uuid

class BDI2Question(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question_text = models.CharField(max_length=255)

    def __str__(self):
        return self.question_text
    

class BDI2Response(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey(BDI2Question, on_delete=models.CASCADE)
    response_text = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    batch = models.IntegerField(default=1) 
    

    def __str__(self):
        return f"{self.user.username}'s response to '{self.question.question_text}'"

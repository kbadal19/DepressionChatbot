from rest_framework import serializers
from .models import PHQ9Question
from .models import PHQResponse


class PHQ9QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PHQ9Question
        fields = '__all__'

class PHQResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = PHQResponse
        fields = '__all__'

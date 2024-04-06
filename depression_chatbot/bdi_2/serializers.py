from .models import BDI2Question, BDI2Response
from rest_framework import serializers


class BDI2QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = BDI2Question
        fields = '__all__'

class  BDI2ResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = BDI2Response
        fields = '__all__'

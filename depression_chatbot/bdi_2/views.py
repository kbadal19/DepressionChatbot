from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from .models import BDI2Question, BDI2Response
from rest_framework import status
from .serializers import BDI2QuestionSerializer, BDI2ResponseSerializer
from django.db.models import Max

class BDI2QuestionList(APIView):
    def get(self, request):
        questions = BDI2Question.objects.all()
        serializer = BDI2QuestionSerializer(questions, many=True)
        return Response(serializer.data)
    
class BDI2ResponseCreate(APIView):
    def post(self, request):
        # Get the user from the request
        user = request.user

        # Check if there are any previous responses for the user
        previous_responses = BDI2Response.objects.filter(user=user)
        
        if previous_responses.exists():
            # If previous responses exist, get the maximum batch number and increment it by 1
            max_batch = previous_responses.aggregate(Max('batch'))['batch__max']
            batch_number = max_batch + 1
        else:
            # If no previous responses exist, set the batch number to 1
            batch_number = 1

        # Add the batch number to the request data
        request.data['batch'] = batch_number

        # Serialize and save the response
        serializer = BDI2ResponseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

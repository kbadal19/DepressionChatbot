from django.shortcuts import render


from rest_framework.views import APIView
from rest_framework.response import Response
from .models import PHQ9Question,PHQResponse
from rest_framework import status
from .serializers import PHQ9QuestionSerializer,PHQResponseSerializer
from django.db.models import Max

class PHQ9QuestionList(APIView):
    def get(self, request):
        questions = PHQ9Question.objects.all()
        serializer = PHQ9QuestionSerializer(questions, many=True)
        return Response(serializer.data)


class PHQResponseCreate(APIView):
    def post(self, request):
        # Get the user from the request
        user = request.data.get('user')

        # Check if there are any previous responses for the user
        previous_responses = PHQResponse.objects.filter(user=user)
        
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
        serializer = PHQResponseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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


from django.shortcuts import get_object_or_404
from django.db.models import Max
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PHQResponse
from django.contrib.auth.models import User

class PHQResponseCreate(APIView):
    def post(self, request):
        # Get the user ID from the request data
        user_id = request.data.get('user')

        # Retrieve the user instance
        user = get_object_or_404(User, pk=user_id)

        # Get the maximum batch number for the user's previous responses
        max_batch = PHQResponse.objects.filter(user=user).aggregate(Max('batch'))['batch__max']

        # Calculate the batch number
        batch_number = max_batch + 1 if max_batch is not None else 1

        # Iterate over each response in the payload
        for response_data in request.data.get('responses', []):
            # Get question ID and response text from the payload
            question_id = response_data.get('question_id')
            response_text = response_data.get('response_text')

            # Retrieve the question instance
            question = get_object_or_404(PHQ9Question, pk=question_id)

            # Create the PHQResponse instance
            PHQResponse.objects.create(
                user=user,
                question=question,
                response_text=response_text,
                batch=batch_number
            )

        return Response({"message": "Responses created successfully"}, status=status.HTTP_201_CREATED)

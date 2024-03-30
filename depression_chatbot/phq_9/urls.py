from . import views
from django.urls import path
from .views import PHQ9QuestionList, PHQResponseCreate

urlpatterns = [
    path('api/questions/', PHQ9QuestionList.as_view(), name='question-list'),
    path('api/responses/', PHQResponseCreate.as_view(), name='response-create'),
]

from . import views
from django.urls import path
from .views import PHQ9QuestionList, PHQResponseCreate, PHQScore

urlpatterns = [
    path('api/questions/', PHQ9QuestionList.as_view(), name='question-list'),
    path('api/responses/', PHQResponseCreate.as_view(), name='response-create'),
    path('api/score/<int:user_id>/', PHQScore.as_view(), name='score'),
]

from . import views
from django.urls import path
from .views import BDI2QuestionList, BDI2ResponseCreate

urlpatterns = [
    path('api/questions/', BDI2QuestionList.as_view(), name='question-list'),
    path('api/responses/', BDI2ResponseCreate.as_view(), name='response-create'),
]

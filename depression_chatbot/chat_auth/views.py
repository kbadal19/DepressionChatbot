from django.shortcuts import render
from rest_framework import generics, permissions
from rest_framework.response import Response
from knox.models import AuthToken
from .serializers import UserSerializer, RegisterSerializer
from django.contrib.auth import login
from rest_framework import permissions
from rest_framework.authtoken.serializers import AuthTokenSerializer
from knox.views import LoginView as KnoxLoginView
from rest_framework import status
class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
       
        return Response({
        "user": UserSerializer(user, context=self.get_serializer_context()).data,
        "token": AuthToken.objects.create(user)[1]
        })
      
        

class LoginAPI(KnoxLoginView):
    permission_classes = (permissions.AllowAny,)
    
    def post(self, request, format=None):
        serializer = AuthTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        first_name = user.first_name.capitalize()
        last_name = user.last_name.capitalize()
        last_login=user.last_login
        full_name = first_name +  ' ' + last_name
        login(request, user)
        token = AuthToken.objects.filter(user=user).first()

        # Customize the response data
        response_data = {
            # 'expiry': token.expiry,
            'token': token.digest,
            'user_id': user.id, # Include the user ID in the response
            'full_name': full_name,
            'last_login':last_login,
        }

        return Response(response_data, status=status.HTTP_200_OK)


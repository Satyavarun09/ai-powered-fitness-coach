from django.urls import path
from . import views

urlpatterns = [
    # Home
    path('', views.index, name='index'),

    # User auth
    path('UserLogin/', views.UserLogin, name='UserLogin'),
    path('UserLoginAction', views.UserLoginAction, name='UserLoginAction'),
    path('Register/', views.Register, name='Register'),
    path('UserScreen/', views.UserScreen, name='UserScreen'),
    path('Logout/', views.Logout, name='Logout'),

    # Admin auth
    path('AdminLogin/', views.AdminLogin, name='AdminLogin'),
    path('AdminLoginAction', views.AdminLoginAction, name='AdminLoginAction'),
    path('AdminDashboard/', views.AdminDashboard, name='AdminDashboard'),

    # Fitness features (protected)
    path('TrainDetection', views.TrainDetection, name='TrainDetection'),
    path('Predict', views.Predict, name='Predict'),
    path('PredictAction', views.PredictAction, name='PredictAction'),
    path('PredictVideo', views.PredictVideo, name='PredictVideo'),
    path('PredictVideoAction', views.PredictVideoAction, name='PredictVideoAction'),
    path('Graph', views.Graph, name='Graph'),
]

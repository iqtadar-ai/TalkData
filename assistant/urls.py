from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('matrix/', views.test_matrix, name='test_matrix'),
    path('undo/', views.undo_action, name='undo'),
    path('redo/', views.redo_action, name='redo'),
    ]  
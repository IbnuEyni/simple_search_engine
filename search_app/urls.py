from django.urls import path
from . import views

urlpatterns = [
    path('', views.query_page, name='query_page'),
    path('process_query/', views.process_query, name='process_query'),
]
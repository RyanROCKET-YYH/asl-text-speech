# Use the official Python image as the base image
FROM python:3

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# # Install Django
# RUN pip install Django==3.2.4

# Install DB
RUN pip install psycopg2


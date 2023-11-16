# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim

# Set working directory
WORKDIR /usr/src/app

# Copy the entire project directory into the container
COPY . .

# Install any dependencies
RUN pip install pytest numpy

# Run your application
CMD [ "pytest", "test_sparse_recommender.py" ]

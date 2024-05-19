# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run softmax_mnist.py when the container launches
CMD ["python", "softmax_mnist.py"]

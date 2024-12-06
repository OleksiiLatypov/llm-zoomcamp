# Use the specified base image
FROM svizor/zoomcamp-model:3.11.5-slim

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock to install dependencies
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --system --deploy

# Copy your Flask script and model file (update the names as needed)
COPY ["predict.py", "model1.bin", "dv.bin", "./"]

# Expose the port the app runs on
EXPOSE 9690

# Command to run the application using Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9690", "predict:app"]
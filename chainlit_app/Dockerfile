# Use an official Python runtime as a parent image
FROM python:3.11-bullseye

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install chainlit[sqlalchemy]
RUN pip install psycopg2-binary asyncpg

# Make port 8000 available to the world outside this container
EXPOSE 8010

# Run main.py when the container launches
# Dockerfile
CMD ["python", "-m", "chainlit", "run", "app.py", "--port", "8010", "--host", "0.0.0.0"]
# CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "8010", "--root-path", "/compliance-chat"]
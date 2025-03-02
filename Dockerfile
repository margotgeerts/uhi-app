# Use Miniconda as a base image
FROM continuumio/miniconda3:latest

# Set environment name
ENV ENV_NAME uhi

# Set working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app/

# Copy environment file
COPY env.yml /tmp/env.yml

# Install Conda environment and clean up
RUN conda env create -f /tmp/env.yml && conda clean --all -y

# Activate environment for shell commands
SHELL ["conda", "run", "-n", "uhi", "/bin/bash", "-c"]

# Expose Streamlit default port
EXPOSE 8501

# Set default command to run Streamlit
CMD ["conda", "run", "-n", "uhi", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    vim \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create dev user
RUN useradd -m dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

# Switch to dev user
USER dev

# Set working directory
WORKDIR /usr/src/app

# Copy requirements ONLY (not entire codebase - that comes from volume)
COPY --chown=dev:dev requirements.txt /usr/src/app/

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Install development tools
RUN pip install --user --no-cache-dir \
    watchdog[watchmedo] \
    ipython \
    ipdb

# Set Python path
ENV PATH="/home/dev/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Don't use ENTRYPOINT - let docker-compose handle commands
# This allows flexibility for different services (django, celery, etc.)
CMD ["bash"]

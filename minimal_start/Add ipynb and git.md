`Don't recommended, chances of out of time. install dependencies inside the running container`

## 🐍 Dockerfile (with Jupyter + Git support)

This `Dockerfile` sets up a Python 3.10 environment with:

- Git
- Build tools
- Jupyter Notebook support
- A named Jupyter kernel for use in VS Code

### 📄 `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /wbhk

# Install system packages: git, compiler for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential 

RUN pip install --no-cache-dir \
    ipykernel \
    notebook

RUN python -m ipykernel install --user --name=wbhk-kernel --display-name "Python 3.10 (wbhk)"

CMD ["bash"]

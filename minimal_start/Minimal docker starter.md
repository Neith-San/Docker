# 🐳 Minimal Docker Dev Setup & Image Sharing

This guide sets up a minimal Python development environment using Docker. It supports persistent development, saving your environment, and sharing it with others.

---

## 🧱 1. Initial Files

### 📄 `Dockerfile`

```dockerfile
FROM python:3.10-slim
WORKDIR /project
CMD ["bash"]
```

### 📄 `docker-compose.yml`

```yaml
services:
  dev:
    build: .
    image: wbhk-dev
    container_name: wbhk-dev
    volumes:
      - ./project:/project
    working_dir: /project
    stdin_open: true
    tty: true
```

---

## 🚀 2. Start Dev Environment

```bash
docker compose build
docker compose run --name wbhk-dev dev
```

- You’ll get an interactive bash shell inside the container.
- The host's `./project` is mounted into `/project` inside the container.

If you're building a reusable dev environment, you should prefer:

```bash
docker compose up -d
```
Then attach via:

```css
Ctrl+Shift+P → Remote-Containers: Attach to Running Container
```
---

## 🧠 3. End of Development

### ✅ (Optional) Embed code from host into container

```bash
docker cp ./project wbhk-dev:/project
```

### ✅ Commit container → image

```bash
docker commit wbhk-dev wbhk-dev-final
```

### ✅ Save image to `.tar`

```bash
docker save -o wbhk-dev.tar wbhk-dev-final
```

---

## 💾 4. Freeze Dependencies

```bash
pip freeze > requirements.txt
```

> Optional — but useful for tracking installed packages or rebuilding later.

---

## 🔁 5. Resume Work Later
Starting the container again
```bash
docker start -ai wbhk-dev
```

---

## 📦 6. Share with a Friend
Share the Tar file with the friend
### Load the image:

```bash
docker load -i wbhk-dev.tar
```

---

### ✅ Option A — Run directly

```bash
docker run -it --name wbhk-dev wbhk-dev-final
```

---

### ✅ Option B — Use `docker-compose.yml`

```yaml
services:
  dev:
    image: wbhk-dev-final
    container_name: wbhk-dev
    working_dir: /project
    stdin_open: true
    tty: true
```

Then:

```bash
docker compose run dev
```

---

---

## 🧰 7. Extract Code from Image to Host (Optional)

If you saved your container state into an image (e.g., `wbhk-dev-final`), your friend can extract the code from the container like this:

1. Run a container from the image:
   ```bash
   docker run -it --name temp-dev wbhk-dev-final

### ✅ Benefits

- ✅ All code + Python packages are inside the image  
- ✅ No volumes or rebuild needed  
- ✅ Fast, clean, and portable  

---

## 🧠 VS Code Integration

1. Run the container.
2. In VS Code:  
   - Press `Ctrl+Shift+P` → `Remote-Containers: Attach to Running Container`
   - Select `wbhk-dev`
   - If needed: `File → Open Folder → /project` to access your code

---

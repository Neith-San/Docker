https://www.youtube.com/watch?v=PkynrL1aU9o&list=PLOLrQ9Pn6cayGytG1fgUPEsUp3Onol8V7&index=3

<img width="1325" height="688" alt="image" src="https://github.com/user-attachments/assets/5ec9cb3f-0ae5-453f-87b1-641d8295177d" />

That entrypoint.sh is what turns a Docker container into a self-healing Django service instead of just “a Python process”.

Without it, Docker would only run Django.
With it, Docker prepares the system before Django starts.

Let’s decode your file:

#!/bin/ash


Alpine Linux shell. This is the OS inside your image.

echo "Apply database migrations"
python manage.py migrate


This does something extremely important:

It upgrades the database schema every time the container starts.

Why this matters:

You rebuild the image

You add a Django model

The volume still has an old database

Without this → Django crashes

With this → database auto-upgrades

This is how cloud Django works.

exec "$@"


This is the magic line.

It means:

“Now run whatever command Docker wanted to run.”

For example in your Dockerfile:

CMD ["gunicorn", "mysite.wsgi:application"]


Docker really runs:

/entrypoint.sh gunicorn mysite.wsgi:application


So your script becomes:

Run migrations

Then replace itself with Gunicorn

Why exec matters:

Signals (Ctrl+C, docker stop) go to Django correctly

Gunicorn becomes PID 1

No zombie processes

Clean shutdown

Without exec, Docker networking, logs, and restarts break.

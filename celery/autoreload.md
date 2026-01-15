Hot-reload problem (and solution)

By default Celery does not reload when you edit code.

So after editing a task, the worker keeps using old code.

Fix (dev only)

Use Celery’s autoreload:

worker:
  command: celery -A mysite worker -l info --pool=solo --autoreload


Now:

You edit tasks.py

Celery reloads automatically

No restart needed

This is huge for development.

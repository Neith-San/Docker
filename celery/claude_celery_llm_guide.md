# Celery Shared Tasks & LLM Guide

## Core Concepts

### @shared_task vs @app.task
```python
# Option 1: @shared_task (Recommended for reusable apps)
from celery import shared_task

@shared_task
def my_task():
    return "Hello"

# Option 2: @app.task (When you need app-specific config)
from munkhsub_v3.celery import app

@app.task
def my_task():
    return "Hello"
```

**Use `@shared_task`** - it works across different Celery apps and is more flexible.

### .delay() vs .apply_async()

```python
# .delay() - Simple, quick syntax
result = my_task.delay(arg1, arg2)

# .apply_async() - Advanced options
result = my_task.apply_async(
    args=[arg1, arg2],
    kwargs={'key': 'value'},
    countdown=10,  # Run after 10 seconds
    expires=300,   # Expire after 5 minutes
    retry=True,
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)
```

**Use `.delay()`** for simple cases, **`.apply_async()`** when you need control.

---

## Example 1: Single LLM Call (Basic)

### tasks.py
```python
from celery import shared_task
import torch
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Load model once per worker (not per task)
_generator = None

def get_llm():
    global _generator
    if _generator is None:
        device = 0 if torch.cuda.is_available() else -1
        _generator = pipeline(
            'text-generation',
            model='gpt2',
            device=device
        )
        logger.info(f"Model loaded on device: {device}")
    return _generator

@shared_task(bind=True, max_retries=3)
def generate_text(self, prompt, max_length=100):
    """
    Generate text using LLM
    
    Args:
        prompt (str): Input prompt
        max_length (int): Max tokens to generate
        
    Returns:
        dict: Result with generated text
    """
    try:
        logger.info(f"Task {self.request.id}: Generating text")
        
        llm = get_llm()
        result = llm(prompt, max_length=max_length, num_return_sequences=1)
        
        return {
            'success': True,
            'text': result[0]['generated_text'],
            'task_id': self.request.id
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise self.retry(exc=e, countdown=60)
```

### views.py
```python
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .tasks import generate_text
from celery.result import AsyncResult
import json

@require_http_methods(["POST"])
def submit_task(request):
    """Submit a text generation task"""
    data = json.loads(request.body)
    prompt = data.get('prompt', 'Hello world')
    
    # Queue the task (non-blocking)
    result = generate_text.delay(prompt, max_length=100)
    
    return JsonResponse({
        'task_id': result.id,
        'status': 'queued'
    })

@require_http_methods(["GET"])
def get_result(request, task_id):
    """Check task status and get result"""
    result = AsyncResult(task_id)
    
    if result.ready():
        # Task completed
        return JsonResponse({
            'status': 'completed',
            'result': result.result
        })
    elif result.failed():
        # Task failed
        return JsonResponse({
            'status': 'failed',
            'error': str(result.info)
        })
    else:
        # Task still running
        return JsonResponse({
            'status': 'pending',
            'progress': result.info if result.state == 'PROGRESS' else None
        })
```

### urls.py
```python
from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.submit_task, name='submit_task'),
    path('result/<str:task_id>/', views.get_result, name='get_result'),
]
```

### Usage
```bash
# Submit task
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time"}'

# Response: {"task_id": "abc123", "status": "queued"}

# Check result
curl http://localhost:8000/result/abc123/

# Response: {"status": "completed", "result": {...}}
```

---

## Example 2: Multiple Sequential LLM Calls (Chain)

```python
from celery import shared_task, chain

@shared_task
def generate_story_outline(topic):
    """Step 1: Generate story outline"""
    llm = get_llm()
    prompt = f"Create a story outline about: {topic}"
    result = llm(prompt, max_length=150)
    return result[0]['generated_text']

@shared_task
def expand_story(outline):
    """Step 2: Expand the outline"""
    llm = get_llm()
    prompt = f"Expand this outline into a full story: {outline}"
    result = llm(prompt, max_length=500)
    return result[0]['generated_text']

@shared_task
def add_dialogue(story):
    """Step 3: Add dialogue"""
    llm = get_llm()
    prompt = f"Add dialogue to this story: {story}"
    result = llm(prompt, max_length=600)
    return result[0]['generated_text']

# Chain tasks together (each waits for previous)
def create_full_story(topic):
    """Run tasks in sequence"""
    workflow = chain(
        generate_story_outline.s(topic),
        expand_story.s(),
        add_dialogue.s()
    )
    result = workflow.apply_async()
    return result.id

# In views.py
@require_http_methods(["POST"])
def create_story(request):
    data = json.loads(request.body)
    topic = data.get('topic', 'space adventure')
    
    task_id = create_full_story(topic)
    
    return JsonResponse({
        'task_id': task_id,
        'message': 'Story generation started (3 steps)'
    })
```

---

## Example 3: Multiple Parallel LLM Calls (Group)

```python
from celery import shared_task, group

@shared_task
def generate_with_temperature(prompt, temperature):
    """Generate text with specific temperature"""
    llm = get_llm()
    result = llm(
        prompt, 
        max_length=100,
        temperature=temperature,
        do_sample=True
    )
    return {
        'temperature': temperature,
        'text': result[0]['generated_text']
    }

def generate_variations(prompt, num_variations=5):
    """Generate multiple variations in parallel"""
    # Create parallel tasks
    job = group(
        generate_with_temperature.s(prompt, temp) 
        for temp in [0.5, 0.7, 0.9, 1.1, 1.3]
    )
    result = job.apply_async()
    return result.id

# In views.py
@require_http_methods(["POST"])
def get_variations(request):
    data = json.loads(request.body)
    prompt = data.get('prompt')
    
    task_id = generate_variations(prompt)
    
    return JsonResponse({
        'task_id': task_id,
        'message': '5 variations generating in parallel'
    })

@require_http_methods(["GET"])
def get_group_result(request, task_id):
    """Get results from group of tasks"""
    from celery.result import GroupResult
    
    result = GroupResult.restore(task_id)
    
    if result.ready():
        variations = result.get()  # List of all results
        return JsonResponse({
            'status': 'completed',
            'variations': variations
        })
    else:
        completed = sum(1 for r in result.results if r.ready())
        return JsonResponse({
            'status': 'pending',
            'progress': f'{completed}/{len(result.results)} completed'
        })
```

---

## Example 4: Streaming LLM with Progress Updates

```python
from celery import shared_task

@shared_task(bind=True)
def generate_long_text(self, prompt, num_chunks=5):
    """Generate text in chunks with progress updates"""
    llm = get_llm()
    full_text = prompt
    
    for i in range(num_chunks):
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': num_chunks,
                'status': f'Generating chunk {i+1}/{num_chunks}'
            }
        )
        
        # Generate chunk
        result = llm(full_text, max_length=len(full_text) + 100)
        full_text = result[0]['generated_text']
    
    return {
        'success': True,
        'text': full_text,
        'chunks_generated': num_chunks
    }

# In views.py - JavaScript polling example
@require_http_methods(["GET"])
def check_progress(request, task_id):
    """Check task progress"""
    result = AsyncResult(task_id)
    
    if result.state == 'PROGRESS':
        return JsonResponse({
            'status': 'in_progress',
            'progress': result.info  # The meta dict we sent
        })
    elif result.ready():
        return JsonResponse({
            'status': 'completed',
            'result': result.result
        })
    else:
        return JsonResponse({
            'status': result.state.lower()
        })
```

---

## Example 5: Different LLM Models in Different Tasks

```python
from transformers import pipeline

# Different models for different tasks
_summarizer = None
_classifier = None
_generator = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    return _summarizer

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("sentiment-analysis", device=0)
    return _classifier

def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline("text-generation", model="gpt2", device=0)
    return _generator

@shared_task
def summarize_text(text):
    """Summarize long text"""
    summarizer = get_summarizer()
    result = summarizer(text, max_length=130, min_length=30)
    return result[0]['summary_text']

@shared_task
def analyze_sentiment(text):
    """Analyze sentiment"""
    classifier = get_classifier()
    result = classifier(text)
    return result[0]

@shared_task
def generate_response(prompt):
    """Generate text"""
    generator = get_generator()
    result = generator(prompt, max_length=100)
    return result[0]['generated_text']

# Combine different models
def process_document(text):
    """Process document with multiple models"""
    from celery import chord
    
    # Run tasks in parallel, then combine results
    callback = combine_results.s()
    header = [
        summarize_text.s(text),
        analyze_sentiment.s(text),
        generate_response.s(f"Key points from: {text[:100]}")
    ]
    
    result = chord(header)(callback)
    return result.id

@shared_task
def combine_results(results):
    """Combine results from multiple models"""
    summary, sentiment, key_points = results
    return {
        'summary': summary,
        'sentiment': sentiment,
        'key_points': key_points
    }
```

---

## Example 6: Retry Logic & Error Handling

```python
@shared_task(
    bind=True,
    max_retries=5,
    default_retry_delay=60  # Wait 60 seconds between retries
)
def robust_generation(self, prompt):
    """Generate text with robust error handling"""
    try:
        llm = get_llm()
        result = llm(prompt, max_length=100)
        return result[0]['generated_text']
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU OOM error")
        # Clear cache and retry
        torch.cuda.empty_cache()
        raise self.retry(exc=e, countdown=120)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Exponential backoff: 60s, 120s, 240s...
        raise self.retry(
            exc=e, 
            countdown=60 * (2 ** self.request.retries)
        )

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3},
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes
    retry_jitter=True  # Add randomness to prevent thundering herd
)
def auto_retry_generation(self, prompt):
    """Automatic retry with backoff"""
    llm = get_llm()
    result = llm(prompt, max_length=100)
    return result[0]['generated_text']
```

---

## Example 7: Task Prioritization

```python
@shared_task
def high_priority_generation(prompt):
    """High priority task"""
    llm = get_llm()
    result = llm(prompt, max_length=50)
    return result[0]['generated_text']

@shared_task
def low_priority_generation(prompt):
    """Low priority task"""
    llm = get_llm()
    result = llm(prompt, max_length=200)
    return result[0]['generated_text']

# In views.py
def submit_with_priority(request):
    data = json.loads(request.body)
    prompt = data.get('prompt')
    is_urgent = data.get('urgent', False)
    
    if is_urgent:
        # High priority - runs first
        result = high_priority_generation.apply_async(
            args=[prompt],
            priority=9  # 0-9, higher = more priority
        )
    else:
        # Low priority
        result = low_priority_generation.apply_async(
            args=[prompt],
            priority=3
        )
    
    return JsonResponse({'task_id': result.id})
```

---

## Complete Example: Multi-Step LLM Workflow

```python
# tasks.py
from celery import shared_task, chain, group, chord
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def analyze_input(self, user_input):
    """Step 1: Analyze user input"""
    self.update_state(state='PROGRESS', meta={'step': 'analyzing'})
    
    classifier = get_classifier()
    sentiment = classifier(user_input)[0]
    
    return {
        'input': user_input,
        'sentiment': sentiment
    }

@shared_task(bind=True)
def generate_response_variants(self, analysis_result):
    """Step 2: Generate 3 response variants in parallel"""
    self.update_state(state='PROGRESS', meta={'step': 'generating variants'})
    
    prompt = analysis_result['input']
    sentiment = analysis_result['sentiment']['label']
    
    # Adjust tone based on sentiment
    if sentiment == 'NEGATIVE':
        tone_prompts = [
            f"Respond empathetically to: {prompt}",
            f"Provide helpful solution for: {prompt}",
            f"Acknowledge concern in: {prompt}"
        ]
    else:
        tone_prompts = [
            f"Respond positively to: {prompt}",
            f"Expand on: {prompt}",
            f"Add details to: {prompt}"
        ]
    
    # Generate variants in parallel
    variants_job = group(
        generate_text.s(p, max_length=150) 
        for p in tone_prompts
    )
    
    variants = variants_job.apply_async().get()
    
    return {
        'original': analysis_result,
        'variants': variants
    }

@shared_task(bind=True)
def rank_responses(self, generation_result):
    """Step 3: Rank responses by quality"""
    self.update_state(state='PROGRESS', meta={'step': 'ranking'})
    
    variants = generation_result['variants']
    
    # Simple ranking by length (you could use another model here)
    ranked = sorted(
        variants,
        key=lambda x: len(x['text']),
        reverse=True
    )
    
    return {
        'best_response': ranked[0],
        'alternatives': ranked[1:],
        'original_analysis': generation_result['original']
    }

# Complete workflow
def process_user_message(user_input):
    """
    Complete workflow:
    1. Analyze sentiment
    2. Generate 3 variants in parallel
    3. Rank and return best
    """
    workflow = chain(
        analyze_input.s(user_input),
        generate_response_variants.s(),
        rank_responses.s()
    )
    
    result = workflow.apply_async()
    return result.id
```

---

## Testing Tasks in Django Shell

```python
# Enter Django shell
docker compose exec django python manage.py shell

# Test single task
>>> from llm_tasks.tasks import generate_text
>>> result = generate_text.delay("Hello world")
>>> result.id
'abc-123-def'
>>> result.ready()  # False if still running
>>> result.get(timeout=30)  # Wait up to 30 seconds
{'success': True, 'text': '...'}

# Test chain
>>> from llm_tasks.tasks import process_user_message
>>> task_id = process_user_message("I'm frustrated with this!")
>>> from celery.result import AsyncResult
>>> result = AsyncResult(task_id)
>>> result.state
'PROGRESS'
>>> result.info
{'step': 'generating variants'}
>>> result.get(timeout=60)  # Get final result
```

---

## Key Takeaways

1. **`@shared_task`**: Use for reusable tasks
2. **`.delay()`**: Simple async execution
3. **`.apply_async()`**: Advanced control (priority, countdown, etc.)
4. **`bind=True`**: Access task instance (for retry, progress updates)
5. **`chain()`**: Sequential execution (A → B → C)
6. **`group()`**: Parallel execution (A, B, C all at once)
7. **`chord()`**: Parallel then combine (A, B, C → D)
8. **Progress updates**: Use `self.update_state()`
9. **Load models once**: Use global variables with lazy loading
10. **Error handling**: Use retry logic with exponential backoff

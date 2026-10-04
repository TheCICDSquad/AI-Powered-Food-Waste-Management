FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY foodwaste/ foodwaste/
COPY api/ api/
COPY app/ app/
COPY data/food_waste_dataset.csv data/

# Train inside the image: the build fails if the model misses the quality gate,
# so a bad model can never be shipped.
RUN python -m foodwaste.train

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12
RUN pip install x
COPY . /app
CMD ["python", "app.py"]

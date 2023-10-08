FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
COPY requirements.txt setup.py /app/
RUN pip3 install -r /app/requirements.txt
COPY ./app /app
COPY ./models /models
COPY ./data/mean_sales_item_id_month.csv /data/mean_sales_item_id_month.csv
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]
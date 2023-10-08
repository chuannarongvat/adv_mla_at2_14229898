from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime
import numpy as np

app = FastAPI()

mean_sell_price_df = pd.read_csv('../data/mean_sell_price.csv')

predictive_model = load('../models/predictive/xgb_model.joblib')
item_id_encoder = load('../models/predictive/item_id_encoder.joblib')
dept_id_encoder = load('../models/predictive/dept_id_encoder.joblib')
store_id_encoder = load('../models/predictive/store_id_encoder.joblib')
state_id_encoder = load('../models/predictive/state_id_encoder.joblib')
cat_id_encoder = load('../models/predictive/cat_id_encoder.joblib')

forecasting_model = load('../models/forecasting/sarima_model.joblib')

def render_html(body_content: str) -> HTMLResponse:
    """
    Render the provided body content within a standard HTML template.
    """
    html_template = f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f4f4;
                }}
                form {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                }}
                input[type="text"], input[type="date"] {{
                    padding: 10px;
                    margin-bottom: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    width: 100%;
                }}
                input[type="submit"] {{
                    background-color: #007BFF;
                    color: #fff;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                input[type="submit"]:hover {{
                    background-color: #0056b3;
                }}
                h2 {{
                    color: #333;
                }}
            </style>
        </head>
        <body>
            {body_content}
            <br>
        </body>
    </html>
    """
    return HTMLResponse(content=html_template)

@app.get("/", response_class=HTMLResponse)
def read_root():
    body_content = """
    <h2>Advanced Machine Learning Assignment 2: Machine Learning as a Service</h2>
    <p>This project is a part of the <strong>"Advanced Machine Learnin Application - Spring 2023"</strong> course, focusing on Assignment 2, <strong>"Machine Learning as a Service".</strong></p>
    
    <p><strong>Objective:</strong>  Shops offer items from three distinct categories: hobbies, foods, and household. The primary objectives include:</p>
    <ol>
        <li>Developing a predictive model using a Machine Learning algorithm to accurately predict the sales revenue for a given item in a specific store on a given date.</li>
        <li>Designing a forecasting model utilizing a time-series analysis algorithm that forecasts the total sales revenue across all stores and items for the upcoming 7 days.</li>
    </ol>

    <h3>Endpoints:</h3>
    <ul>
        <li><strong>/sales/stores/items/ (GET)</strong>: Navigate to the prediction form to predict the sales revenue for a specific item, store, and date.</li>
        <li><strong>/sales/stores/items/ (POST)</strong>: Endpoint for the form submission to predict the sales revenue.</li>
        <li><strong>/forecast/ (GET)</strong>: Forecast the total sales revenue across all stores and items for the next 7 days. Output format TBD.</li>
    </ul>

    <h3>Performance Metrics:</h3>
    <p>The choice of performance metrics for the models will be determined based on the nature and distribution of the data, and the business objectives. They are yet to be decided.</p>

    <h3>Project Repository:</h3>
    <p>For more detailed information, data exploration, model building, and other related activities, refer to the <a href="https://github.com/chuannarongvat/adv_mla_at2">GitHub Repository</a>.</p>
    """
    return render_html(body_content)

@app.get("/sales/stores/items/", response_class=HTMLResponse)
async def get_sales_input_form():
    body_content = """
    <h2>Predict Sales Revenue</h2>
    <form action="/sales/stores/items/" method="post">
        Item ID: <br>
        <input type="text" name="item_id"><br><br>
        Store ID: <br>
        <input type="text" name="store_id"><br><br>
        Date: <br>
        <input type="date" name="date"><br><br>
        <input type="submit" value="Predict">
    </form>
    """
    return render_html(body_content)

@app.post("/sales/stores/items/")
async def predict_sales(item_id: str = Form(None), store_id: str = Form(None), date: str = Form(None)):
    # Check if any of the inputs is missing
    if not all([item_id, store_id, date]):
        content = """
        <h2>Predicted Sales Revenue</h2>
        <h1>Error</h1>
        <p>All input fields (Item ID, Store ID, and Date) are required.</p>
        
        <br>
        <br>
        <a href="/sales/stores/items/" style="display: inline-block; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
        """
        return render_html(content)
    
    content = ""
    # 1. Lookup the sell price.
    try:
        sell_price = mean_sell_price_df[mean_sell_price_df['item_id'] == item_id]['sell_price'].values[0]
    except IndexError:
        content = f"""
        <h2>Predicted Sales Revenue</h2>
        <h1>Error</h1>
        <p>No item_id {item_id} in the store.</p> 
        
        <p>Please select a different item_id.</p>
        
        <br>
        <br>
        <a href="/sales/stores/items/" style="display: inline-block; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
        """
        return render_html(content)
                
    store_id_list = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    if store_id not in store_id_list:
        content = f"""
        <h2>Error</h2>
        <p>Invalid Store Id: <strong>{store_id}.</strong> Must be one of <strong>{', '.join(store_id_list)}</strong></p>
        
        <br>
        <br>
        <a href="/sales/stores/items/" style="display: inline-block; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
        """
        return render_html(content)
    
    # 2. Convert the categorical columns using the respective encoders.
    item_id_encoded = item_id_encoder.transform([item_id])[0]
    dept_id = "_".join(item_id.split('_')[:2])
    dept_id_encoded = dept_id_encoder.transform([dept_id])[0]
    cat_id_encoded = cat_id_encoder.transform([item_id.split('_')[0]])[0]
    store_id_encoded = store_id_encoder.transform([store_id])[0]
    state_id_encoded = state_id_encoder.transform([store_id.split('_')[0]])[0]
    
    # 3. Handle date-related manipulations.
    date_dt = pd.to_datetime(date)
    year = date_dt.year
    month = date_dt.month
    week_number = np.ceil((date_dt - pd.Timestamp('2011-01-29')) / np.timedelta64(1, 'W')).astype(int) + 1
    day_of_week = date_dt.dayofweek
    season_sin = np.sin(2 * np.pi * month / 12)
    season_cos = np.cos(2 * np.pi * month / 12)
    
    # 4. Assemble the final DataFrame.
    user_input = pd.DataFrame({
        'item_id': [item_id_encoded],
        'dept_id': [dept_id_encoded],
        'cat_id': [cat_id_encoded],
        'store_id': [store_id_encoded],
        'state_id': [state_id_encoded],
        'sell_price': [sell_price],
        'year': [year],
        'month': [month],
        'week_number': [week_number],
        'day_of_week': [day_of_week],
        'season_sin': [season_sin],
        'season_cos': [season_cos]
    })
    
    sales_pred = predictive_model.predict(user_input)[0]
    sales_revenue_pred = sales_pred * sell_price
    content = f"""
        <h2>Predict Sales Revenue</h2>

        <p>The predicted sales revenue for item <strong>{item_id}</strong> in store <strong>{store_id}</strong> on date <strong>{date}</strong> is:</p>

        <div style="font-size: 24px; color: #007BFF; margin-top: 10px;">${sales_revenue_pred:0.2f}</div>

        <br>
        <br>
        <a href="/sales/stores/items/" style="display: inline-block; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
        """
    return render_html(content)

@app.get("/sales/national/", response_class=HTMLResponse)
async def get_forecast_or_form(start_date: str = None):
    if not start_date:
        body_content = """
        <h2>Forecast Weekly Sales</h2>
        <form action="/sales/national/" method="get">
            Start Date (to forecast from): <br>
            <input type="date" name="start_date"><br><br>
            <input type="submit" value="Forecast">
        </form>
        """
        return render_html(body_content)

    try:
        start_date_dt = pd.to_datetime(start_date)
    except ValueError:
        content = """
        <h2>Error</h2>
        <p>Invalid date format. Please use YYYY-MM-DD format.</p>
        <a href="/sales/national/" style="display: inline-block; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
        """
        return render_html(content)

    forecasts = forecasting_model.forecast(steps=7)

    forecast_dates = pd.date_range(start_date_dt, periods=7)
    
    table_rows = ""
    for date, forecast in zip(forecast_dates, forecasts):
        table_rows += f"<tr><td>{date.strftime('%Y-%m-%d')}</td><td>${forecast:.2f}</td></tr>"

    content = f"""
    <h2>Forecasts from {start_date}</h2>
    
    <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
        <thead>
            <tr>
                <th style="text-align: left; border-bottom: 2px solid #666; padding: 8px;">Date</th>
                <th style="text-align: left; border-bottom: 2px solid #666; padding: 8px;">Forecasted Sales ($)</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    
    <a href="/sales/national/" style="display: inline-block; margin-top: 20px; padding: 10px 15px; background-color: #007BFF; color: #fff; border-radius: 4px; text-decoration: none; cursor: pointer;">Go Back</a>
    """
    return render_html(content)




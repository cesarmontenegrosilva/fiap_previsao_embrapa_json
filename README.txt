# Flask Wine Sales Forecasting API

This project is a Flask-based API that collects, processes, and predicts wine sales data from the EMBRAPA Viticulture website using web scraping and the Prophet forecasting model.

## Features

- **Data Collection**: The API collects wine sales data from the EMBRAPA Viticulture website for different categories such as production, processing, commercialization, importation, and exportation.
- **Data Preparation**: The API processes the collected data, particularly focusing on the sales of "VINHO FINO DE MESA" (fine table wine), specifically the "Tinto" (red wine) subcategory.
- **Forecasting**: The API uses the Prophet model to predict future wine sales based on historical data.
- **Evaluation**: The model's performance is evaluated using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R2), and Mean Absolute Percentage Error (MAPE).

## Technologies Used

- **Python**: The programming language used to develop the application.
- **Flask**: A lightweight WSGI web application framework for Python.
- **BeautifulSoup**: A library used for web scraping to extract data from HTML and XML files.
- **Prophet**: A forecasting tool developed by Facebook used to predict time series data.
- **Pandas**: A data manipulation and analysis library.
- **Scikit-learn**: A machine learning library used here for model evaluation metrics.
- **Matplotlib**: A plotting library used to generate graphs for predictions (optional and currently commented out).

## Endpoints

### `/`

- **Method**: GET
- **Description**: This endpoint triggers the data collection, preparation, and forecasting process.
- **Query Parameters**:
  - `periodos` (optional): Number of periods (years) to forecast. Default is 5.
- **Response**: A JSON object containing the following:
  - `mae`: Mean Absolute Error of the forecast.
  - `rmse`: Root Mean Squared Error of the forecast.
  - `r2`: R-squared value of the forecast.
  - `mape`: Mean Absolute Percentage Error of the forecast.
  - `previsoes`: List of dictionaries containing the forecasted values (`ds` for date and `yhat` for predicted value).

## Setup and Installation

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cesarmontenegrosilva/fiap_previsao_embrapa_json
Navigate to the project directory:

bash
Copiar código
cd your-repo-name
Install the required packages:

bash
Copiar código
pip install -r requirements.txt
Run the Flask application:

bash
Copiar código
python app.py
The API will be available at http://127.0.0.1:5000/.

Usage
You can access the API via a web browser, Postman, or any HTTP client. For example:

bash
Copiar código
curl "http://127.0.0.1:5000/?periodos=10"
This will return a JSON response with the forecasted wine sales for the next 10 years, along with evaluation metrics.

Notes
The forecasting model uses historical data from 1970 to 2023.
The plotting functionality for visualizing the forecast is currently commented out but can be re-enabled by uncommenting the related code in the plotar_grafico function and the relevant parts of the / route.
License
This project is licensed under the MIT License. See the LICENSE file for details.

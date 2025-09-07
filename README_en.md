# A-Share DCA Trading Bot

This project is a **Quantitative Dollar-Cost Averaging (DCA) Trading Bot for the Chinese A-share market**.  
It supports data collection, model training, signal scoring, budget allocation, and weekly trading simulation.  
By combining machine learning models, financial/news sentiment analysis, and dynamic fund management, it helps you automatically build a weekly investment portfolio with a starting budget of ¥20,000.

---

## 🚀 Features
- **Data Collection & Preprocessing**: Fetch market and earnings data using CSV/akshare.  
- **Model Training & Prediction**: Supports classification/regression and saves trained models automatically.  
- **Signal Scoring**: Combines price trends, financial indicators, news sentiment, and model outputs.  
- **Budget Allocation**: Automatically selects Top N stocks (default: 5) and allocates funds proportionally.  
- **Weekly Trading Simulation**: Generates Excel reports (buy plan, expected return, fees).  
- **Proxy Support**: Configurable to access Chinese financial data from overseas environments.  

---

## 📂 Project Structure
```
dca_bot/
│── tools/               # Utility scripts
│   ├── check_signals.py
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── model.py
│   ├── eval.py
│   ├── train_model.py
│   ├── predict_model.py
│   ├── model_config.yaml
│── orchestrator.py       # Main orchestrator script
│── run_weekly.py         # Weekly task runner (or run orchestrator directly)
│── main.py
│── requirements.txt      # Dependencies
│── README.md
│── .gitignore
├── signals_config.yaml
├── budget.yaml
├── symbol.txt
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/a-share-dca-bot.git
cd a-share-dca-bot
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 3. Daily Run (Data Fetch)
The project relies on updated daily market and earnings data:
```bash
run_fetch.bat
python cninfo_earnings_to_csv.py
```

### 4. Train Models
Run once a month to update models (more stocks = slower training):
```bash
train_once.bat
```
- Outputs `model.pkl` for each stock.  
- To build a pooled model for all stocks, add `--pooled` when running.

### 5. Predict & Generate Scores
```bash
predict_once.bat
```
- Produces `model_score.json` required by the orchestrator.  

### 6. Weekly Orchestrator Run
```bash
python orchestrator.py --budget configs/budget.yaml --signals configs/signals_config.yaml --timeout 6 --workers 6
```

**Output:**  
Excel report under `logs/weekly_budget/Report_YYYYMMDD.xlsx` containing:  
- Selected Top 5 stocks  
- Buy quantity & cost  
- Expected 7-day sell price & return  
- Commission (default: EastMoney broker rate)  

---

## 🔧 Configuration

### 1. Stock Pool (`symbols.txt`)
```
600111.SH
600183.SH
600176.SH
601138.SH
```
👉 Update this list to change stock pool.  

### 2. Signal Configuration (`signals_config.yaml`)
```yaml
symbols:
  600111.SH: {}
  600183.SH: {}
  600176.SH: {}
  601138.SH: {}

news:
  lookback_days: 7
  rss:
    enabled: true
    timeout_sec: 10
    retries: 2
    backoff: 1.8
  html_scrape:
    enabled: true
    sources:
      - kind: bing
        url: "https://www.bing.com/news/search?q={q}+site%3Aeastmoney.com&setlang=zh-cn&FORM=HDRSC6"
      - kind: bing
        url: "https://www.bing.com/news/search?q={q}+site%3Asina.com.cn&setlang=zh-cn&FORM=HDRSC6"
```
👉 Modify `sources` to adjust news providers.  

### 3. Model Signals (`tools/model_signals.yaml`)
```yaml
weights:
  price_trend: 0.25
  earnings_trend: 0.25
  news_sentiment: 0.25
  model_score: 0.25
```
👉 Adjust weights to rebalance contributions.  

### 4. Budget (`budget.yaml`)
```yaml
budget: 20000
topn: 5
broker_fee: 0.0003   # Brokerage fee (0.03%)
```
👉 Modify budget, Top N, or fee settings here.  

---

## 🌐 Data Sources
- **Market Data:** `run_fetch.bat` → stored under `data/`  
- **Earnings Data:** `python cninfo_earnings_to_csv.py`  
- **Overseas Users:** Recommended EastMoney API + akshare  
- **Domestic Users:** Direct akshare  
- **Default Earnings Fetch:** `fetch_cninfo.py` (Cninfo)  
- **News Data:** RSS (Google News) + HTML scraping (Bing, EastMoney, Sina, Netease, Tencent Finance, etc.)  

# A-Share DCA Trading Bot

本项目是一个 **A股定投量化交易机器人**，支持数据获取、模型训练、信号打分、预算分配和每周交易模拟。它结合机器学习模型、新闻/财报情绪分析和动态资金管理，帮助你在 2 万元预算下自动构建每周投资组合。

---

## 🚀 功能特点
- **数据获取与预处理**：基于 CSV/akshare 抓取行情与财报数据。
- **模型训练与预测**：支持分类/回归，自动保存训练好的模型。
- **信号打分**：结合价格趋势、财报指标、新闻情绪和模型分数。
- **预算分配**：自动选取 TopN 股票（默认 5 支），按比例分配资金。
- **每周交易模拟**：生成 Excel 报告（买入计划、预估收益、手续费）。
- **支持代理**：可在美国/海外环境下配置代理访问国内财经数据。

---

## 📂 项目结构
dca_bot/
│── tools/ # 工具脚本
│ ├── check_signals.py
│ ├── data_loader.py
│ ├── features.py
│ ├── labels.py
│ ├── model.py
| |—— eval.py
│ ├── train_model.py
│ ├── predict_model.py
│ ├── model_config.yaml
│── orchestrator.py # 主控脚本
│── run_weekly.py # 每周任务脚本（这个你可以直接运行orchestrator也可以单独点他）
│── main.py
│── requirements.txt # 依赖清单
│── README.md
│── .gitignore
├── signals_config.yaml
├── budget.yaml
├── symbol.txt

---

## ⚙️ 安装与运行

### 1. 克隆仓库
```bash
git clone https://github.com/<你的用户名>/a-share-dca-bot.git
cd a-share-dca-bot
2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
3. 每日运行（数据抓取）

项目依赖每日行情和财报更新，你需要运行：

run_fetch.bat
python cninfo_earnings_to_csv.py
4. 训练模型
train_once.bat（这个一个月更新一次模型就行，如果你选股较多就要更新的慢点---毕竟做模型很耗时间）（这个会输出一个model.pkl文件给每只股票单独生成的，如果你想要所有股票做一个集合模型在运行train_once.bat的时候在后面加上 --pooled）

5. 预测并生成结果
predict_once.bat(这个会有一个model_score.json是后面orchestrator要的）

6. 每周运行一次 Orchestrator
python orchestrator.py --budget configs/budget.yaml --signals configs/signals_config.yaml --timeout 6 --workers 6


输出结果会保存在：

logs/weekly_budget/Report_YYYYMMDD.xlsx


生成的 Excel 报告包含：

选出的前 5 支股票

买入手数与成本

预计 7 日后卖出价与收益率

手续费计算（默认东方财富券商标准）

1. 股票池（symbols.txt）
600111.SH
600183.SH
600176.SH
601138.SH


👉 想换目标（symbols， signals——config， model——signals 都需要改对应不同的。其中symbol负责对接财报抓取列表，signal对应市场数据和权重比如check——signal和orchestrator的运行参数，而model——signals只参与模型训练，budget不涉及选股只有预算和比例修改）

2. 信号配置（signals_config.yaml）
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


👉 想调整新闻来源，就修改 sources。

3. 模型信号配置（tools/model_signals.yaml）
weights:
  price_trend: 0.25
  earnings_trend: 0.25
  news_sentiment: 0.25
  model_score: 0.25


👉 想调整模型权重比例，修改这里即可。

4. 预算（budget.yaml）
budget: 20000
topn: 5
broker_fee: 0.0003   # 单边千三手续费


👉 想换预算或买入股票数量，直接改这里。

🌐 数据源说明

行情数据：run——fetch.bat（写入 data/ 文件夹）

财报数据：python cninfo_earnings_to_csv.py

国外：推荐注册东方财富 API + akshare

国内：直接用 akshare

默认支持 fetch_cninfo.py 抓取巨潮
新闻数据：RSS（Google News）+ HTML 抓取（Bing + 东方财富/新浪/网易/腾讯财经等）

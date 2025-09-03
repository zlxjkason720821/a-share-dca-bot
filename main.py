import os, sys, argparse, time, csv
from datetime import datetime
from typing import Any, Dict, Optional, List

import yaml
from pydantic import BaseModel, Field

# ===== Interfaces =====
class OrderLeg(BaseModel):
    symbol: str
    qty: Optional[float] = None            # 固定股数
    notional: Optional[float] = None       # 固定金额
    order_type: str = "market"             # market / limit
    limit_up_pct: Optional[float] = None   # limit: 以最近价上浮/下浮百分比（买入用上浮，卖出用下浮）
    lot_size: Optional[int] = None         # A股100股一手等

class Plan(BaseModel):
    account: str
    currency: str = "USD"
    legs: List[OrderLeg]

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 支持从环境变量注入
    raw = os.path.expandvars(raw)
    return yaml.safe_load(raw)

def round_lot(qty: float, lot_size: Optional[int]) -> int:
    if lot_size and lot_size > 1:
        return int(qty // lot_size * lot_size)
    return int(qty)

def env_bool(name: str, default: bool=False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1","true","yes","y","on")

# ===== Broker base =====
class BrokerBase:
    def get_last_price(self, symbol: str) -> float:
        raise NotImplementedError
    def ensure_currency(self, currency: str):
        pass
    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float] = None) -> dict:
        raise NotImplementedError

# ===== Manual broker (for A-shares without open API) =====
class ManualBroker(BrokerBase):
    def __init__(self, output_dir="./logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_last_price(self, symbol: str) -> float:
        # 无行情的离线模式：让用户自行填入，或由你接入第三方行情
        return 0.0

    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float] = None) -> dict:
        # 记录为待执行清单
        path = os.path.join(self.output_dir, f"manual_orders_{datetime.now().strftime('%Y%m%d')}.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time","symbol","qty","notional","order_type","limit_price","note"])
            writer.writerow([datetime.now().isoformat(), symbol, qty or "", notional or "", order_type, limit_price or "", "请在东方财富/同花顺手工下单"])
        return {"status":"logged","symbol":symbol,"qty":qty,"notional":notional,"order_type":order_type,"limit_price":limit_price,"path":path}

# ===== Alpaca broker =====
def _try_import_alpaca():
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi
    except Exception as e:
        print("[warn] alpaca_trade_api 未安装或导入失败：", e)
        return None

class AlpacaBroker(BrokerBase):
    def __init__(self, key_id: str, secret_key: str, base_url: str):
        api_mod = _try_import_alpaca()
        if api_mod is None:
            raise RuntimeError("缺少 alpaca_trade_api 依赖。请先 pip install alpaca-trade-api")
        self.api = api_mod.REST(key_id, secret_key, base_url)
    def get_last_price(self, symbol: str) -> float:
        quote = self.api.get_latest_trade(symbol)
        return float(quote.price)
    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float] = None) -> dict:
        ptype = "market" if order_type=="market" else "limit"
        if qty:
            o = self.api.submit_order(symbol=symbol, qty=qty, side="buy", type=ptype, time_in_force="day", limit_price=limit_price)
        else:
            o = self.api.submit_order(symbol=symbol, notional=notional, side="buy", type=ptype, time_in_force="day", limit_price=limit_price)
        return {"status":"submitted","id":str(o.id),"symbol":symbol}

# ===== IBKR broker =====
def _try_import_ib():
    try:
        from ib_insync import IB, Stock, MarketOrder, LimitOrder
        return IB, Stock, MarketOrder, LimitOrder
    except Exception as e:
        print("[warn] ib_insync 未安装或导入失败：", e)
        return None, None, None, None

class IBKRBroker(BrokerBase):
    def __init__(self, host: str, port: int, client_id: int):
        IB, Stock, MarketOrder, LimitOrder = _try_import_ib()
        if IB is None:
            raise RuntimeError("缺少 ib_insync 依赖。请先 pip install ib-insync 并运行 TWS/IB Gateway")
        self.IB = IB; self.Stock = Stock; self.MarketOrder = MarketOrder; self.LimitOrder = LimitOrder
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
    def get_last_price(self, symbol: str) -> float:
        contract = self.Stock(symbol, 'SMART', 'USD')
        mkt = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(1.0)
        return float(mkt.last if mkt.last else (mkt.close or 0.0))
    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float] = None) -> dict:
        if qty is None:
            # 以 notional/last_price 近似换算股数
            last = self.get_last_price(symbol)
            qty = int((notional or 0) / max(last, 0.01))
        contract = self.Stock(symbol, 'SMART', 'USD')
        order = self.MarketOrder('BUY', qty) if order_type=="market" else self.LimitOrder('BUY', qty, limit_price)
        trade = self.ib.placeOrder(contract, order)
        return {"status":"submitted","symbol":symbol,"orderId":trade.order.orderId}

# ===== Tiger broker =====
def _try_import_tiger():
    try:
        from tigeropen.common.util.order_utils import MarketOrder, LimitOrder
        from tigeropen.trade.tiger_trade_client import TigerTradeClient
        from tigeropen.common.response import TigerResponse
        return MarketOrder, LimitOrder, TigerTradeClient
    except Exception as e:
        print("[warn] tigeropen 未安装或导入失败：", e)
        return None, None, None

class TigerBroker(BrokerBase):
    def __init__(self, account: str, tiger_id: str, private_key: str, sandbox: bool=True):
        MarketOrder, LimitOrder, TigerTradeClient = _try_import_tiger()
        if TigerTradeClient is None:
            raise RuntimeError("缺少 tigeropen 依赖。请 pip install tigeropen，并依据官方文档配置公私钥。")
        self.MarketOrder = MarketOrder; self.LimitOrder = LimitOrder
        self.client = TigerTradeClient(tiger_id=tiger_id, private_key_path=private_key, sandbox=sandbox)
        self.account = account
    def get_last_price(self, symbol: str) -> float:
        # 省略行情实现，实际可调用行情接口或第三方
        return 0.0
    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float] = None) -> dict:
        if qty is None:
            raise ValueError("Tiger 适配简化示例需提供 qty")
        order = self.MarketOrder(symbol, qty) if order_type=="market" else self.LimitOrder(symbol, qty, limit_price)
        resp = self.client.place_order(self.account, order)
        return {"status":"submitted","resp":str(resp)}

# ===== Futu broker =====
def _try_import_futu():
    try:
        from futu import OpenSecTradeContext, TrdEnv, TrdSide, OrderType, TrdMarket, RET_OK
        from futu import TrdMarket as FTTrdMarket
        return OpenSecTradeContext, TrdEnv, TrdSide, OrderType, TrdMarket, RET_OK
    except Exception as e:
        print("[warn] futu-api 未安装或导入失败：", e)
        return (None,)*6

class FutuBroker(BrokerBase):
    def __init__(self, host: str, port: int):
        OpenSecTradeContext, TrdEnv, TrdSide, OrderType, TrdMarket, RET_OK = _try_import_futu()
        if OpenSecTradeContext is None:
            raise RuntimeError("缺少 futu-api 依赖或未启动 OpenD。请安装 futu-api 并启动 FutuOpenD。")
        self.OpenSecTradeContext = OpenSecTradeContext
        self.TrdEnv = TrdEnv; self.TrdSide = TrdSide; self.OrderType = OrderType; self.TrdMarket = TrdMarket; self.RET_OK = RET_OK
        self.ctx = OpenSecTradeContext(host=host, port=port, security_firm=0)
    def get_last_price(self, symbol: str) -> float:
        # 省略行情（需另开行情上下文）；此处返回0以避免示例依赖
        return 0.0
    def place_order(self, symbol: str, qty: Optional[int], notional: Optional[float], order_type: str, limit_price: Optional[float]=None) -> dict:
        if qty is None:
            raise ValueError("Futu 适配简化示例需提供 qty")
        side = self.TrdSide.BUY
        otype = self.OrderType.MARKET if order_type=="market" else self.OrderType.NORMAL
        ret, data = self.ctx.place_order(price=limit_price or 0, qty=qty, code=symbol, trd_side=side, order_type=otype, trd_env=self.TrdEnv.SIMULATE)
        if ret != self.RET_OK:
            raise RuntimeError(f"Futu place_order error: {data}")
        return {"status":"submitted","order_id":str(data['order_id'].iloc[0])}

# ===== factory =====
def broker_factory(name: str, params: dict) -> BrokerBase:
    name = name.lower()
    if name == "alpaca":
        return AlpacaBroker(**params)
    if name == "ibkr":
        return IBKRBroker(**params)
    if name == "tiger":
        return TigerBroker(**params)
    if name == "futu":
        return FutuBroker(**params)
    if name == "manual":
        return ManualBroker(**params)
    raise ValueError(f"Unsupported broker: {name}")

def run_plan(plan_name: str, cfg: dict, dry_run: bool=False):
    accounts = cfg.get("accounts", {})
    plans = cfg.get("plans", {})
    if plan_name not in plans:
        raise KeyError(f"Plan '{plan_name}' not found in config.")
    p = Plan(**plans[plan_name])
    acc = accounts[p.account]
    broker = broker_factory(acc["broker"], acc.get("params", {}))

    results = []
    for leg in p.legs:
        leg = OrderLeg(**leg.model_dump())
        limit_price = None
        if leg.order_type == "limit":
            last = broker.get_last_price(leg.symbol)
            if last <= 0 and leg.limit_up_pct is not None:
                # 无行情时提示用户
                print(f"[warn] 无法获取 {leg.symbol} 最近价，limit_up_pct 将被忽略。")
            else:
                limit_price = round(last * (1 + (leg.limit_up_pct or 0.0)), 4)

        # notional -> qty 的简单近似（无行情或不支持时，保留 notional 由券商端处理）
        qty = leg.qty
        if qty is None and leg.notional is not None and hasattr(broker, "get_last_price"):
            last = broker.get_last_price(leg.symbol)
            if last and last > 0:
                qty = max(1, int(leg.notional / last))

        # A股整数手处理
        if leg.lot_size and qty:
            qty = max(leg.lot_size, round_lot(qty, leg.lot_size))

        if dry_run:
            print(f"[dry-run] {leg.symbol} qty={qty} notional={leg.notional} type={leg.order_type} limit_price={limit_price}")
            res = {"status":"dry-run","symbol":leg.symbol,"qty":qty,"notional":leg.notional,"limit_price":limit_price}
        else:
            res = broker.place_order(symbol=leg.symbol, qty=qty, notional=leg.notional, order_type=leg.order_type, limit_price=limit_price)
        results.append(res)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--list-plans", action="store_true")
    ap.add_argument("--run-plan", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.list_plans:
        print("可用计划：", ", ".join(cfg.get("plans", {}).keys()))
        return

    if args.run_plan:
        res = run_plan(args.run_plan, cfg, dry_run=args.dry_run)
        print(res)
        return

    print("使用方法：")
    print("  python main.py --list-plans")
    print("  python main.py --run-plan demo_us_weekly --dry-run")

if __name__ == "__main__":
    main()

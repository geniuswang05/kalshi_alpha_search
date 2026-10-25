"""Kalshi Trading API v2 客户端骨架。

====================================================================
你需要去 Kalshi 拿的东西（只有一样）：
  API Key —— 登录 kalshi.com → Settings → API Keys → Create New Key。
  会得到两个东西：
    1. Key ID（一串 UUID）
    2. RSA 私钥文件（.pem，只显示一次，下载保存好）
  把它们配置成环境变量：
    export KALSHI_KEY_ID="你的key id"
    export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi_private_key.pem"

  官方文档：https://docs.kalshi.com （API Reference → Trading API v2）
====================================================================

我们用到的 API 端点（base: https://external-api.kalshi.com/trade-api/v2，
2026-09 官方文档域名；旧域名 api.elections.kalshi.com 仍兼容）：

价格格式注意：此域名下价格字段是美元字符串（如 yes_bid_dollars: "0.5600"），
旧版整数美分字段（yes_bid: 56）已移除，解析用 market_implied_prob()。

  只读行情（GET，无需签名；orderbook 除外——它要签名）：
    GET /series                       列出系列（找 NBA 相关的 series_ticker）
    GET /events?series_ticker=...     某系列下的事件（= 一场具体比赛）
    GET /markets?event_ticker=...     事件下的合约（主队赢/客队赢）
    GET /markets/{ticker}             单个合约详情（yes_bid/yes_ask/last_price/volume）
    GET /markets/{ticker}/orderbook   订单簿深度（算滑点用，需签名）
    GET /markets/trades               公开成交记录
    GET /series/{series_ticker}/markets/{ticker}/candlesticks
                                      历史K线（period_interval: 1/60/1440 分钟）
                                      —— 回测拿 Kalshi 历史价格就靠它

  交易/账户（需签名。注意：V1 的 POST /portfolio/orders 已被 V2 取代，
  2026-05 后弃用，所以文档里搜不到）：
    GET    /portfolio/balance                  账户余额
    GET    /portfolio/positions                持仓
    POST   /portfolio/events/orders            下单（V2：side=bid/ask，价格美元字符串）
    DELETE /portfolio/events/orders/{order_id} 撤单（V2）
    GET    /portfolio/orders                   挂单/历史订单列表
    GET    /portfolio/fills                    成交记录（算真实 PnL 用）

认证方式（每个请求三个 header）：
    KALSHI-ACCESS-KEY:       Key ID
    KALSHI-ACCESS-TIMESTAMP: 毫秒时间戳
    KALSHI-ACCESS-SIGNATURE: 用 RSA 私钥对 "timestamp + method + path"
                             做 RSA-PSS SHA256 签名后 base64
"""

import base64
import datetime
import os

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DEMO_URL = "https://external-api.demo.kalshi.co/trade-api/v2"   # 模拟盘（假钱）；需单独申请 demo key


def market_implied_prob(market: dict) -> dict:
    """从 market dict 提取隐含概率（新版 API 价格是 '0.5600' 这种美元字符串）。

    返回 yes 侧的 bid/ask/mid，mid 可当作市场隐含概率的点估计；
    实际算 edge 时买入用 ask、卖出用 bid（考虑价差）。
    """
    bid = float(market["yes_bid_dollars"])
    ask = float(market["yes_ask_dollars"])
    return {"yes_bid": bid, "yes_ask": ask, "yes_mid": (bid + ask) / 2.0}


class KalshiClient:
    def __init__(self, key_id: str | None = None, private_key_path: str | None = None,
                 base_url: str = BASE_URL):
        self.base_url = base_url
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        self._private_key = None
        if self.private_key_path and os.path.exists(self.private_key_path):
            self._load_private_key()

    def _load_private_key(self):
        from cryptography.hazmat.primitives import serialization  # pip install cryptography

        with open(self.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        msg = f"{timestamp_ms}{method}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _headers(self, method: str, path: str) -> dict:
        if self._private_key is None:
            return {}  # 公开行情端点可以裸奔
        ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, f"/trade-api/v2{path}"),
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(self.base_url + path, params=params,
                            headers=self._headers("GET", path), timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ---------- 行情 ----------

    def get_series_list(self, category: str = "Sports") -> dict:
        return self._get("/series", params={"category": category})

    def get_events(self, series_ticker: str, status: str = "open") -> dict:
        return self._get("/events", params={"series_ticker": series_ticker,
                                            "status": status})

    def get_markets(self, event_ticker: str) -> dict:
        return self._get("/markets", params={"event_ticker": event_ticker})

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """需要签名认证（2026-09 起 orderbook 不再公开）。"""
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: str, limit: int = 100) -> dict:
        return self._get("/markets/trades", params={"ticker": ticker, "limit": limit})

    def get_candlesticks(self, series_ticker: str, ticker: str,
                         start_ts: int, end_ts: int, period_interval: int = 60) -> dict:
        """历史K线。period_interval 只接受 1/60/1440（分钟）。回测数据源。"""
        return self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts,
                    "period_interval": period_interval})

    # ---------- 账户/交易（全部需要签名；实盘前先在 demo 环境验证）----------

    def _request(self, method: str, path: str, json_body: dict | None = None) -> dict:
        resp = requests.request(method, self.base_url + path, json=json_body,
                                headers=self._headers(method, path), timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def get_balance(self) -> dict:
        return self._get("/portfolio/balance")

    def get_positions(self, limit: int = 100) -> dict:
        return self._get("/portfolio/positions", params={"limit": limit})

    def get_orders(self, status: str = "resting", ticker: str | None = None) -> dict:
        params = {"status": status}
        if ticker:
            params["ticker"] = ticker
        return self._get("/portfolio/orders", params=params)

    def get_fills(self, ticker: str | None = None, min_ts: int | None = None) -> dict:
        params = {}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        return self._get("/portfolio/fills", params=params or None)

    def create_order(self, ticker: str, side: str, count: float, price: float,
                     client_order_id: str | None = None,
                     time_in_force: str = "good_till_canceled",
                     post_only: bool = False) -> dict:
        """V2 下单（POST /portfolio/events/orders）。

        side: "bid"(买) / "ask"(卖) —— V2 是单簿模型，不再区分 yes/no 方向；
        count/price 必须是定点字符串，如 count="10.00", price="0.5600"。
        """
        body = {
            "ticker": ticker,
            "side": side,
            "count": f"{count:.2f}",
            "price": f"{price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id  # 幂等去重，强烈建议传
        if post_only:
            body["post_only"] = True                   # 会成交（吃单）的单直接拒绝，保证只做 maker
        return self._request("POST", "/portfolio/events/orders", json_body=body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")

    def cancel_all(self) -> dict:
        """撤销全部挂单（DELETE /portfolio/events/orders）——机器人退出/熔断时调用。"""
        return self._request("DELETE", "/portfolio/events/orders")


if __name__ == "__main__":
    # 无 key 冒烟测试：公开行情端点应该能直接访问
    client = KalshiClient()
    series = client.get_series_list()
    tickers = [s.get("ticker") for s in series.get("series", [])]
    nba = [t for t in tickers if t and "NBA" in t.upper()]
    print(f"共 {len(tickers)} 个 sports series，NBA 相关：{nba}")

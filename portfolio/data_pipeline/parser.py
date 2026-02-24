from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "symbol",
    "mic",
    "account_type",
    "quantity",
    "book_value",
    "average_price",
]


def _safe_get(d: dict, path: list[str]):
    current = d
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _extract_account_type(node: dict) -> str | None:
    accounts = node.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        return None
    account_id = accounts[0].get("id") if isinstance(accounts[0], dict) else None
    if not account_id or not isinstance(account_id, str):
        return None
    return account_id.split("-", 1)[0].upper()


def parse_portfolio_json(raw_json: dict) -> pd.DataFrame:
    edges = _safe_get(raw_json, ["data", "identity", "financials", "current", "positions", "edges"])
    if not isinstance(edges, list):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    records: list[dict] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node = edge.get("node")
        if not isinstance(node, dict):
            continue

        records.append(
            {
                "symbol": _safe_get(node, ["security", "stock", "symbol"]),
                "mic": _safe_get(node, ["security", "stock", "primaryMic"]),
                "account_type": _extract_account_type(node),
                "quantity": node.get("quantity"),
                "book_value": _safe_get(node, ["bookValue", "amount"]),
                "average_price": _safe_get(node, ["averagePrice", "amount"]),
            }
        )

    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    for col in ["quantity", "book_value", "average_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df

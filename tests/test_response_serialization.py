# 回归测试：_response / _json_safe 能安全序列化 pandas/numpy 缺失值。
# 背景：get_stock_info 返回含 pd.NaT / nan 的原始行时，FastMCP 序列化抛错
# （"float object cannot be interpreted as an integer"），导致工具调用 isError。
# 修复：所有工具都经 _response 返回，在 _response 内对 data 递归清理，
#       一处修复对全部 111 个工具生效。
import json

import numpy as np
import pandas as pd
import pytest

from pa_mcp.server import _json_safe, _response


def test_nat_and_nan_become_null():
    """NaT / NaN 缺失值应转成 JSON null，而不是让序列化崩溃。"""
    data = {
        "list_date": pd.NaT,
        "market_cap": float("nan"),
        "np_nan": np.nan,
        "np_nat": np.datetime64("NaT"),
        "name": "贵州茅台",
    }
    resp = _response(data=data)
    cleaned = resp["data"]
    assert cleaned["list_date"] is None
    assert cleaned["market_cap"] is None
    assert cleaned["np_nan"] is None
    assert cleaned["np_nat"] is None
    assert cleaned["name"] == "贵州茅台"
    # 必须能被标准 JSON 序列化
    json.dumps(resp, ensure_ascii=False)


def test_timestamp_to_iso_string():
    """Timestamp / date 应转为 ISO 字符串（客户端更好消费）。"""
    data = {"ts": pd.Timestamp("2026-09-06 10:30:00"), "d": "普通字符串"}
    cleaned = _response(data=data)["data"]
    assert cleaned["ts"] == "2026-09-06T10:30:00"
    assert cleaned["d"] == "普通字符串"
    json.dumps(_response(data=data), ensure_ascii=False)


def test_recursive_dict_and_list():
    """嵌套 dict / list 内的缺失值也要清理，正常值保持不变。"""
    data = {
        "rows": [
            {"date": pd.NaT, "close": 10.5},
            {"date": "2026-09-05", "close": 11.0},
        ],
        "meta": {"ok": True, "when": pd.Timestamp("2026-09-06")},
    }
    cleaned = _response(data=data)["data"]
    assert cleaned["rows"][0]["date"] is None
    assert cleaned["rows"][0]["close"] == 10.5
    assert cleaned["rows"][1]["date"] == "2026-09-05"
    assert cleaned["meta"]["when"] == "2026-09-06T00:00:00"
    json.dumps(cleaned, ensure_ascii=False)


def test_normal_data_unaffected():
    """干净数据不应被误改。"""
    data = {"success": True, "items": [1, 2, 3], "s": "x", "n": None}
    cleaned = _response(data=data)["data"]
    assert cleaned == {"success": True, "items": [1, 2, 3], "s": "x", "n": None}


def test_scalars_and_containers_do_not_false_positive():
    """字典/列表容器不能因 pd.isna 逐元素返回而被误判为空。"""
    assert _json_safe({"a": 1}) == {"a": 1}
    assert _json_safe([1, 2]) == [1, 2]
    assert _json_safe("text") == "text"
    assert _json_safe(3) == 3
    assert _json_safe(True) is True
    assert _json_safe(None) is None
    assert _json_safe(0) == 0
    assert _json_safe(0.0) == 0.0


def test_numpy_scalars_to_native():
    """numpy 原生标量转为 python 原生，避免序列化问题。"""
    assert _json_safe(np.float64(1.5)) == 1.5
    assert _json_safe(np.int64(7)) == 7
    assert _json_safe(np.float32(2.0)) == 2.0

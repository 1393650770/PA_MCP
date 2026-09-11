"""charts.figures 纯函数测试：新增的连板梯队图与预测验证图。

只测纯函数（不碰 DB / kaleido 导出），保证快速且稳定。
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from pa_mcp.charts.figures import (
    limit_up_ladder_figure,
    prediction_review_figure,
)


def _ladder_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
        "limit_up_count": [60, 45, 80],
        "first_board_count": [40, 30, 65],
        "board2_count": [12, 8, 10],
        "board3_count": [5, 4, 3],
        "board4p_count": [3, 3, 2],
    })


def _pred_df() -> pd.DataFrame:
    return pd.DataFrame({
        "predict_date": pd.to_datetime(["2026-09-01"] * 4),
        "symbol": ["000001", "600196", "000333", "601318"],
        "horizon": ["5d"] * 4,
        "direction": ["up", "up", "down", "sideways"],
        "probability": [0.7, 0.6, 0.55, 0.5],
        "status": ["hit", "miss", "hit", "ambiguous"],
        "actual_return_pct": [3.2, -1.1, 2.0, 0.4],
    })


class TestLimitUpLadderFigure:
    def test_returns_figure_with_four_bars(self):
        fig = limit_up_ladder_figure(_ladder_df())
        assert isinstance(fig, go.Figure)
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 4  # 首板/2板/3板/4板+
        names = {b.name for b in bars}
        assert {"首板", "2 板", "3 板", "4 板+"} == names

    def test_total_line_added_when_present(self):
        fig = limit_up_ladder_figure(_ladder_df())
        lines = [t for t in fig.data if t.type == "scatter"]
        assert len(lines) == 1  # 涨停总数参考线
        assert lines[0].name == "涨停总数"

    def test_date_axis_is_category(self):
        """'MM-DD' 字符串会被 plotly 按「月-年」误解析 → 必须 category。"""
        fig = limit_up_ladder_figure(_ladder_df())
        assert fig.layout.xaxis.type == "category"

    def test_stacked_values_sum_to_total(self):
        df = _ladder_df()
        fig = limit_up_ladder_figure(df)
        bars = [t for t in fig.data if t.type == "bar"]
        for i in range(len(df)):
            stacked = sum(int(b.y[i]) for b in bars)
            assert stacked == int(df["limit_up_count"][i])

    @pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"date": []})])
    def test_empty_returns_placeholder(self, df):
        fig = limit_up_ladder_figure(df)
        assert isinstance(fig, go.Figure)
        assert any(a.text == "暂无数据" for a in fig.layout.annotations)


class TestPredictionReviewFigure:
    def test_returns_two_panel_figure(self):
        fig = prediction_review_figure(_pred_df())
        assert isinstance(fig, go.Figure)
        bars = [t for t in fig.data if t.type == "bar"]
        assert bars  # 左面板有命中/未中/模糊柱
        scatters = [t for t in fig.data if t.type == "scatter"]
        assert scatters  # 右面板散点

    def test_counts_match_input(self):
        df = _pred_df()
        fig = prediction_review_figure(df)
        bars = [t for t in fig.data if t.type == "bar"]
        # hit=2, miss=1, ambiguous=1 → 各方向笔数总和 = 4
        total = sum(int(v) for b in bars for v in b.y)
        assert total == len(df)

    def test_directions_labeled_in_chinese(self):
        fig = prediction_review_figure(_pred_df())
        xcats = [t.x for t in fig.data if t.type == "bar" and t.x is not None]
        assert any("看涨" in list(x) for x in xcats)

    def test_empty_returns_placeholder(self):
        fig = prediction_review_figure(pd.DataFrame())
        assert any(a.text == "暂无数据" for a in fig.layout.annotations)

    def test_missing_optional_columns_still_renders(self):
        """只有 direction/status 也能出图（缺概率/实际收益时跳过散点）。"""
        fig = prediction_review_figure(
            pd.DataFrame({"direction": ["up"], "status": ["hit"]}))
        assert isinstance(fig, go.Figure)
        assert not [t for t in fig.data if t.type == "scatter"]

"""毒性预测和PNEC推导结果的Altair图表。"""

from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

from model import ModelConfig, predict_toxicity


CURRENT_COLOR = "#d62728"
MEASURED_COLOR = "#1f77b4"
PREDICTED_COLOR = "#7f7f7f"
LINE_COLOR = "#333333"


def prepare_model_plot_data(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """计算绘图所需的固定方程预测值，不在图中暴露建模集身份。"""
    plot_data = data.copy()
    plot_data["Fixed_Prediction"] = [
        predict_toxicity(
            config.key,
            {name: row[name] for name in config.descriptors},
        )
        for _, row in plot_data.iterrows()
    ]
    plot_data["Plot_Data_Type"] = plot_data["Status"].map(
        lambda status: "预测数据" if status == "Untested" else "实测数据"
    )
    plot_data["logP"] = plot_data["AlogP"]
    return plot_data


def toxicity_scatter_chart(
    plot_data: pd.DataFrame, endpoint: str, selected_cas: Optional[str] = None
) -> alt.Chart:
    """实测值—固定方程预测值散点图及y=x参考线。"""
    measured = plot_data.loc[plot_data["Status"] != "Untested"].copy()
    minimum = min(measured["Exp_pTox"].min(), measured["Fixed_Prediction"].min())
    maximum = max(measured["Exp_pTox"].max(), measured["Fixed_Prediction"].max())
    padding = max((maximum - minimum) * 0.06, 0.1)
    domain = [float(minimum - padding), float(maximum + padding)]
    line_data = pd.DataFrame({"x": domain, "y": domain})

    base = (
        alt.Chart(measured)
        .mark_circle(size=75, color=MEASURED_COLOR, opacity=0.75)
        .encode(
            x=alt.X(
                "Exp_pTox:Q",
                title=f"实测{endpoint}",
                scale=alt.Scale(domain=domain, nice=False),
            ),
            y=alt.Y(
                "Fixed_Prediction:Q",
                title=f"预测{endpoint}",
                scale=alt.Scale(domain=domain, nice=False),
            ),
            tooltip=[
                alt.Tooltip("CAS:N", title="CAS"),
                alt.Tooltip("Exp_pTox:Q", title=f"实测{endpoint}", format=".4f"),
                alt.Tooltip("Fixed_Prediction:Q", title=f"预测{endpoint}", format=".4f"),
            ],
        )
    )
    reference = (
        alt.Chart(line_data)
        .mark_line(color=LINE_COLOR, strokeDash=[5, 4])
        .encode(x="x:Q", y="y:Q")
    )
    chart = reference + base
    current = measured.loc[measured["CAS"].astype(str) == str(selected_cas)]
    if not current.empty:
        chart = chart + (
            alt.Chart(current)
            .mark_point(shape="diamond", size=180, color=CURRENT_COLOR, filled=True)
            .encode(x="Exp_pTox:Q", y="Fixed_Prediction:Q", tooltip=["CAS:N"])
        )
    return chart.properties(height=340, title="实测值—预测值对比（虚线为y=x）")


def williams_chart(
    plot_data: pd.DataFrame, h_star: float, selected_cas: Optional[str] = None
) -> alt.Chart:
    """仅使用有实测残差的记录绘制威廉姆斯图。"""
    measured = plot_data.loc[plot_data["Status"] != "Untested"].copy()
    base = (
        alt.Chart(measured)
        .mark_circle(size=75, color=MEASURED_COLOR, opacity=0.75)
        .encode(
            x=alt.X("SI_h:Q", title="杠杆值 h"),
            y=alt.Y("SI_Std_Resid:Q", title="标准化残差"),
            tooltip=[
                alt.Tooltip("CAS:N", title="CAS"),
                alt.Tooltip("SI_h:Q", title="h", format=".4f"),
                alt.Tooltip("SI_Std_Resid:Q", title="标准化残差", format=".4f"),
            ],
        )
    )
    vertical = (
        alt.Chart(pd.DataFrame({"h": [h_star]}))
        .mark_rule(color=LINE_COLOR, strokeDash=[5, 4])
        .encode(x="h:Q")
    )
    horizontal = (
        alt.Chart(pd.DataFrame({"residual": [-3.0, 0.0, 3.0]}))
        .mark_rule(color=LINE_COLOR, strokeDash=[5, 4])
        .encode(y="residual:Q")
    )
    chart = vertical + horizontal + base
    current = measured.loc[measured["CAS"].astype(str) == str(selected_cas)]
    if not current.empty:
        chart = chart + (
            alt.Chart(current)
            .mark_point(shape="diamond", size=180, color=CURRENT_COLOR, filled=True)
            .encode(x="SI_h:Q", y="SI_Std_Resid:Q", tooltip=["CAS:N"])
        )
    return chart.properties(height=340, title=f"威廉姆斯图（h* = {h_star:.4f}）")


def chemical_space_chart(
    plot_data: pd.DataFrame, selected_cas: Optional[str] = None
) -> alt.Chart:
    """以MW和logP展示全部化合物及当前化合物。"""
    base = (
        alt.Chart(plot_data)
        .mark_circle(size=60, opacity=0.65)
        .encode(
            x=alt.X("MW:Q", title="MW（g/mol）"),
            y=alt.Y("logP:Q", title="logP"),
            color=alt.Color(
                "Plot_Data_Type:N",
                title="数据类型",
                scale=alt.Scale(
                    domain=["实测数据", "预测数据"],
                    range=[MEASURED_COLOR, PREDICTED_COLOR],
                ),
            ),
            tooltip=[
                alt.Tooltip("CAS:N", title="CAS"),
                alt.Tooltip("MW:Q", title="MW", format=".3f"),
                alt.Tooltip("logP:Q", title="logP", format=".4f"),
                alt.Tooltip("Plot_Data_Type:N", title="数据类型"),
            ],
        )
    )
    chart = base
    current = plot_data.loc[plot_data["CAS"].astype(str) == str(selected_cas)]
    if not current.empty:
        chart = chart + (
            alt.Chart(current)
            .mark_point(shape="diamond", size=200, color=CURRENT_COLOR, filled=True)
            .encode(
                x="MW:Q",
                y="logP:Q",
                tooltip=["CAS:N", "MW:Q", "logP:Q"],
            )
        )
    return chart.properties(height=340, title="MW–logP化学空间（红色菱形为当前化合物）")


def pnec_taxa_chart(taxa_details: List[Dict[str, Any]]) -> alt.Chart:
    """比较四类群pMCV，并突出PNEC推导使用的最敏感类群。"""
    data = pd.DataFrame(taxa_details)
    data["Plot_Species"] = data["Species"].astype(str).str.split("（").str[0]
    data["Used_Label"] = data["Used_for_PNEC"].map({True: "用于PNEC", False: "其他类群"})
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("pMCV:Q", title="pMCV（越高表示越敏感）"),
            y=alt.Y("Plot_Species:N", title="类群", sort="-x"),
            color=alt.Color(
                "Data_Type:N",
                title="数据类型",
                scale=alt.Scale(
                    domain=["实测数据", "预测数据"],
                    range=[MEASURED_COLOR, PREDICTED_COLOR],
                ),
            ),
            opacity=alt.condition(
                alt.FieldOneOfPredicate(field="Used_for_PNEC", oneOf=[True]),
                alt.value(1.0),
                alt.value(0.4),
            ),
            tooltip=[
                alt.Tooltip("Plot_Species:N", title="类群"),
                alt.Tooltip("pMCV:Q", title="pMCV", format=".4f"),
                alt.Tooltip("MCV_mg_L:Q", title="MCV（mg/L）", format=".4g"),
                alt.Tooltip("Data_Type:N", title="数据类型"),
                alt.Tooltip("Prediction_Quality:N", title="预测质量"),
                alt.Tooltip("Used_Label:N", title="用途"),
            ],
        )
    )
    labels = (
        alt.Chart(data)
        .mark_text(align="left", dx=5, color=LINE_COLOR)
        .encode(
            x="pMCV:Q",
            y=alt.Y("Plot_Species:N", sort="-x"),
            text=alt.Text("pMCV:Q", format=".3f"),
        )
    )
    return (bars + labels).properties(
        height=300,
        title="四个类群慢性毒性对比（不透明条形为PNEC推导依据）",
    )

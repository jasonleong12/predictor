"""PFAS水生毒性预测及PNEC估算工具。"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from model import EFFECT_LEVEL, MODEL_CONFIGS, find_compound_record, load_dataset, predict
from plots import (
    chemical_space_chart,
    pnec_taxa_chart,
    prepare_model_plot_data,
    toxicity_scatter_chart,
    williams_chart,
)
from pnec import ASSESSMENT_FACTOR, derive_pnec, load_pnec_dataset


st.set_page_config(page_title="PFAS水生毒性预测及PNEC估算工具", layout="wide")

QUALITY_CN = {
    "Good": "良好",
    "Moderate": "中等",
    "Poor": "较差",
    "Bad/Unreliable": "较差/不可靠",
}
AD_CN = {"In": "应用域内", "Outside AD": "应用域外"}
SPECIES_ORDER = ("zebrafish", "daphnia_magna", "water_flea", "midge", "algae")
TAXON_NOTES = {
    "water_flea_chronic": (
        "水溞慢性数据涵盖：大型溞（*Daphnia magna*）、模糊网纹溞（*Ceriodaphnia dubia*）、"
        "多刺裸腹溞（*Moina macrocopa*）、隆线溞（*Daphnia carinata*）和"
        "湖生蚤状溞（*Daphnia pulicaria*）。"
    ),
    "midge_chronic": (
        "摇蚊慢性数据涵盖：岸溪摇蚊（*Chironomus riparius*）、羽摇蚊（*Chironomus plumosus*）"
        "和伸展双叶摇蚊（*Chironomus tentans*）。"
    ),
    "algae_chronic": (
        "绿藻慢性数据涵盖：栅藻属（*Scenedesmus*）、小球藻属（*Chlorella*）、"
        "衣藻属（*Chlamydomonas*）、尖胞藻属（*Raphidocelis*）和月牙藻属（*Selenastrum*）。"
    ),
}
SPECIES_IMAGES = {
    "zebrafish": ("zebrafish.jpg", "斑马鱼示意图"),
    "daphnia_magna": ("water_flea.jpg", "大型溞（*Daphnia magna*）示意图"),
    "water_flea": ("water_flea.jpg", "水溞类群示意图（图为大型溞 *Daphnia magna*）"),
    "midge": ("midge.jpg", "摇蚊类群示意图"),
    "algae": ("algae.jpg", "绿藻类群示意图"),
}
SCORE_CN = {3: "良好", 2: "中等", 1: "较差"}


def score_with_label(value):
    score = int(value)
    return f"{score}（{SCORE_CN[score]}）"


def format_number(value, digits=8):
    if value is None:
        return "—"
    return f"{value:.{digits}g}"


@st.cache_data
def get_dataset(model_key):
    return load_dataset(model_key)


@st.cache_data
def get_pnec_dataset():
    return load_pnec_dataset()


@st.cache_data
def get_plot_data(model_key):
    return prepare_model_plot_data(load_dataset(model_key), MODEL_CONFIGS[model_key])


def render_reference():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**参考文献**")
    st.sidebar.markdown(
        "Liang, W., Li, J., Niu, L., Wang, X., Zhao, X., 2026. "
        "A species-specific QSAR-based framework for PFAS hazard identification "
        "and ecological risk mapping. *Environmental Research*. 305, 125057. "
        "[https://doi.org/10.1016/j.envres.2026.125057]"
        "(https://doi.org/10.1016/j.envres.2026.125057)"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("结果仅用于科研与危害筛查，不能替代实测毒性数据或法定环境限值。")


def render_toxicity_module():
    species_labels = {
        group: next(
            config.species_label
            for config in MODEL_CONFIGS.values()
            if config.species_group == group
        )
        for group in SPECIES_ORDER
    }

    with st.sidebar:
        st.header("模型选择")
        selected_species = st.selectbox(
            "物种", [species_labels[group] for group in SPECIES_ORDER]
        )
        species_group = next(
            group for group, label in species_labels.items() if label == selected_species
        )
        toxicity_options = [
            config.toxicity_type
            for config in MODEL_CONFIGS.values()
            if config.species_group == species_group
        ]
        selected_toxicity = st.selectbox(
            "毒性类型", toxicity_options, key=f"toxicity_{species_group}"
        )
        model_key = next(
            key
            for key, config in MODEL_CONFIGS.items()
            if config.species_group == species_group
            and config.toxicity_type == selected_toxicity
        )
        config = MODEL_CONFIGS[model_key]
        st.selectbox("毒性终点", [config.endpoint])
        if species_group in SPECIES_IMAGES:
            image_name, image_caption = SPECIES_IMAGES[species_group]
            image_path = Path(__file__).resolve().parent / "assets" / "species" / image_name
            image_left, image_center, image_right = st.columns([1, 4, 1])
            with image_center:
                st.image(str(image_path), use_container_width=True)
            st.caption(image_caption)

    data = get_dataset(model_key)
    cas_options = sorted(data["CAS"].astype(str).tolist())

    st.subheader("1. 化合物与自变量输入")
    st.caption(
        f"当前模型：{config.species_label}｜{config.toxicity_type}｜{config.endpoint}"
    )
    if model_key in TAXON_NOTES:
        st.caption(TAXON_NOTES[model_key])
    selected_option = st.selectbox(
        "选择化合物CAS",
        cas_options,
        key=f"cas_selector_{model_key}",
    )
    record = find_compound_record(model_key, selected_option, data)
    defaults = {name: float(record[name]) for name in (*config.descriptors, "MW", "AlogP")}

    widget_suffix = f"{model_key}_{selected_option}"
    cas = selected_option
    with st.form(f"prediction_form_{model_key}"):
        input_columns = st.columns(len(config.descriptors) + 2)
        descriptor_values = {}
        for column, name in zip(input_columns, config.descriptors):
            with column:
                descriptor_values[name] = st.number_input(
                    name,
                    value=defaults[name],
                    format="%.12g",
                    key=f"{name}_{widget_suffix}",
                )
        with input_columns[-2]:
            mw = st.number_input(
                "MW（g/mol）",
                value=defaults["MW"],
                min_value=0.0,
                format="%.8g",
                key=f"MW_{widget_suffix}",
                help="用于将摩尔浓度换算为mg/L。",
            )
        with input_columns[-1]:
            alogp = st.number_input(
                "logP",
                value=defaults["AlogP"],
                format="%.8g",
                key=f"AlogP_{widget_suffix}",
                help="logP仅作为化合物信息展示，不参与固定方程计算。",
            )
        submitted = st.form_submit_button("计算毒性并评价结果", type="primary")

    with st.expander("查看固定模型方程与使用说明"):
        st.code(config.equation, language=None)
        st.markdown(
            "- 有实测记录的化合物：明确展示实测毒性，模型预测仅用于实测—预测对照，不进行PRI评级。\n"
            "- 待预测化合物且描述符未修改：由固定方程计算毒性，并显示应用域及PRI质量评价。\n"
            "- 修改描述符：重新计算毒性和杠杆值应用域，PRI显示为未评估。"
        )

    if not submitted:
        return

    try:
        result = predict(model_key, cas, descriptor_values, mw, alogp)
    except ValueError as exc:
        st.error(str(exc))
        return

    endpoint = result["Endpoint"]
    is_measured = result["Data_Type"] == "实测数据"
    st.subheader("2. 毒性数据与模型预测")
    st.caption(
        f"化合物信息：CAS = {result['CAS']}｜MW = {format_number(result['MW'])} g/mol｜"
        f"logP = {format_number(result['AlogP'])}"
    )
    st.info(f"毒性效应说明：输出结果基于{EFFECT_LEVEL}。")
    if is_measured:
        st.success("该化合物有实测毒性数据，以下首先展示实测结果；模型预测仅作对照。")
        st.markdown("**实测毒性**")
        exp1, exp2, exp3, exp4 = st.columns(4)
        exp1.metric(f"实测{endpoint}", f"{result['Exp_pTox']:.6f}")
        exp2.metric("实测毒性（mmol/L）", format_number(result["Exp_Tox_mmol_L"]))
        exp3.metric("实测毒性（µmol/L）", format_number(result["Exp_Tox_umol_L"]))
        exp4.metric("实测毒性（mg/L）", format_number(result["Exp_Tox_mg_L"]))
        st.markdown("**模型预测（用于实测—预测对照）**")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric(f"预测{endpoint}", f"{result['pTox']:.6f}")
    metric2.metric("预测毒性（mmol/L）", format_number(result["Tox_mmol_L"]))
    metric3.metric("预测毒性（µmol/L）", format_number(result["Tox_umol_L"]))
    metric4.metric("预测毒性（mg/L）", format_number(result["Tox_mg_L"]))
    if is_measured:
        st.caption(
            f"预测{endpoint}－实测{endpoint} = {result['Prediction_Minus_Exp']:+.6f}"
        )

    st.subheader("3. 应用域与质量评价")
    quality_col, ad_col, h_col = st.columns(3)
    quality = result["Prediction_Quality"]
    quality_text = QUALITY_CN.get(quality, "未评估")
    if is_measured:
        quality_col.metric("数据性质", "实测数据")
    else:
        quality_col.metric("预测质量", quality_text)
    ad_col.metric("应用域判定", AD_CN.get(result["AD_status"], result["AD_status"]))
    h_col.metric("杠杆值h", f"{result['h']:.5f}", help=f"h* = {config.h_star:.4f}")

    if result["AD_status"] == "Outside AD":
        st.warning("该化合物位于应用域外或存在响应异常，模型预测不确定性较高，请谨慎使用。")

    if is_measured:
        st.info("该记录为实测数据，实测值优先于模型预测值，不适用于PRI质量评级。")
        st.write(f"标准化残差：{result['SI_Std_Resid']:.4f}")
    elif result["Reference_Match"]:
        if quality in ("Poor", "Bad/Unreliable"):
            st.error("综合预测质量为较差或不可靠，不建议将本结果作为定量决策依据。")

        st.markdown("**PRI分项与相似性信息**")
        pri_table = pd.DataFrame(
            [
                ["MAE-LOO评分（权重0.5）", score_with_label(result["Score_MAE_LOO"])],
                ["毒性响应接近程度评分（权重0.3）", score_with_label(result["Score_Proximity_Train_YObs_Mean"])],
                ["基于相似性的应用域评分（权重0.2）", score_with_label(result["Score_Similarity_AD"])],
                ["综合评分", score_with_label(result["Composite_Score"])],
                ["欧氏距离", result["Euclidean_Distance"]],
                ["广义杰卡德系数", result["Generalized_Jaccard_Coeff"]],
                ["毒性响应分布位置", result["Yobs_Test_Distribution"]],
            ],
            columns=["评价项目", "结果"],
        )
        pri_table["结果"] = pri_table["结果"].astype(str)
        st.dataframe(pri_table, hide_index=True, use_container_width=True)
    else:
        st.warning("当前描述符与原始记录不一致，PRI综合质量未评估。")

    st.subheader("4. 结果可视化")
    plot_data = get_plot_data(model_key)
    highlight_cas = result["CAS"] if result["Reference_Match"] else None
    scatter_tab, williams_tab, space_tab = st.tabs(
        ["实测—预测散点图", "威廉姆斯图", "MW–logP化学空间"]
    )
    with scatter_tab:
        st.altair_chart(
            toxicity_scatter_chart(plot_data, endpoint, highlight_cas),
            use_container_width=True,
        )
        if not is_measured:
            st.caption("当前化合物没有实测毒性值，因此不在实测—预测散点图中单独标出。")
    with williams_tab:
        st.altair_chart(
            williams_chart(plot_data, config.h_star, highlight_cas),
            use_container_width=True,
        )
        if not is_measured:
            st.caption("当前化合物没有实测残差，因此威廉姆斯图不显示其残差点。")
    with space_tab:
        st.altair_chart(
            chemical_space_chart(plot_data, highlight_cas),
            use_container_width=True,
        )

    st.subheader("5. 导出结果")
    export_columns = [
        "Model_Key", "CAS", "Species", "Toxicity_Type", "Endpoint", "Effect_Level",
        "Data_Type", *config.descriptors, "MW", "AlogP",
        "Exp_pTox", "Exp_Tox_mmol_L", "Exp_Tox_umol_L", "Exp_Tox_mg_L",
        "pTox", "Tox_mmol_L", "Tox_umol_L", "Tox_mg_L", "h", "h_star",
        "AD_status", "Prediction_Quality",
    ]
    export_data = pd.DataFrame([{column: result.get(column) for column in export_columns}])
    export_data = export_data.rename(
        columns={"AlogP": "logP", "Prediction_Quality": "预测质量"}
    )
    safe_cas = re.sub(r"[^A-Za-z0-9_.-]+", "_", result["CAS"])
    st.download_button(
        "下载当前结果（CSV）",
        data=export_data.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{model_key}_{safe_cas}.csv",
        mime="text/csv",
    )


def render_pnec_module():
    st.sidebar.header("PNEC设置")
    st.sidebar.selectbox("评估因子（AF）", [ASSESSMENT_FACTOR])
    data = get_pnec_dataset()
    cas_options = sorted(data["CAS"].astype(str).tolist())

    st.subheader("1. 化合物输入")
    selected_option = st.selectbox("选择化合物CAS", cas_options)
    with st.form("pnec_form"):
        cas = selected_option
        submitted = st.form_submit_button("推导PNEC", type="primary")

    with st.expander("查看PNEC推导流程"):
        st.markdown(
            "1. 获取斑马鱼、水溞、摇蚊和绿藻的慢性毒性pMCV；有实测值时采用实测值，否则采用QSAR预测值。\n"
            "2. 将各类群的pMCV换算为MCV（mg/L）。pMCV越高，对应MCV越低，表示该类群越敏感。\n"
            "3. 选择pMCV最高的类群作为最敏感类群，并采用其MCV。\n"
            f"4. 按评估因子法计算：**PNEC = 最敏感类群MCV ÷ AF**，本模块固定AF = {ASSESSMENT_FACTOR}。\n\n"
            "AF = 10对应包含鱼类、溞类和藻类等至少3个物种长期毒性数据的推导情形。"
            "推导结果用于初步生态风险筛查，不直接作为特定区域的水生态基准或安全限值。"
        )

    if not submitted:
        return

    try:
        result = derive_pnec(cas)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("2. PNEC推导结果")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PNEC（mg/L）", format_number(result["PNEC_mg_L"]))
    col2.metric("PNEC（µg/L）", format_number(result["PNEC_ug_L"]))
    col3.metric("最敏感类群MCV（mg/L）", format_number(result["MCV_mg_L"]))
    col4.metric("评估因子（AF）", result["AF"])
    st.caption(
        f"化合物信息：CAS = {result['CAS']}｜"
        f"logP = {format_number(result['Sensitive_AlogP'])}"
    )

    st.markdown("**最敏感类群及其数据质量**")
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("最敏感类群", result["Sensitive_Species"])
    info2.metric("用于推导的pMCV", f"{result['Sensitive_pMCV']:.6f}")
    info3.metric("数据类型", result["Sensitive_Data_Type"])
    quality = result["Sensitive_Prediction_Quality"]
    quality_text = QUALITY_CN.get(quality, "不适用（实测）" if quality is None else "未评估")
    info4.metric("预测质量", quality_text)
    st.write(
        f"数据构成：{result['PNEC_Data_Composition']}；"
        f"应用域：{AD_CN.get(result['Sensitive_AD_status'], '不适用（实测）')}。"
    )
    if result["PNEC_Quality_Warning"]:
        st.warning(result["PNEC_Quality_Warning"])
    st.markdown("**四个类群的慢性毒性与质量记录**")
    taxa_rows = []
    for detail in result["Taxa_Details"]:
        taxa_rows.append(
            {
                "类群": detail["Species"],
                "pMCV": detail["pMCV"],
                "MCV（mg/L）": detail["MCV_mg_L"],
                "数据类型": detail["Data_Type"],
                "应用域": AD_CN.get(detail["AD_status"], "—"),
                "预测质量": QUALITY_CN.get(detail["Prediction_Quality"], "—"),
                "用于PNEC": "是" if detail["Used_for_PNEC"] else "否",
            }
        )
    st.dataframe(pd.DataFrame(taxa_rows), hide_index=True, use_container_width=True)
    st.altair_chart(pnec_taxa_chart(result["Taxa_Details"]), use_container_width=True)
    st.code(
        f"PNEC = {result['MCV_mg_L']:.8g} mg/L ÷ {result['AF']} "
        f"= {result['PNEC_mg_L']:.8g} mg/L",
        language=None,
    )

    st.subheader("3. 导出结果")
    export_columns = [
        "CAS", "AF", "Sensitive_Species", "Sensitive_pMCV", "Sensitive_Data_Type",
        "Sensitive_AD_status", "Sensitive_Prediction_Quality",
        "PNEC_Data_Composition", "PNEC_Quality_Warning",
        "Sensitive_AlogP", "MCV_mg_L", "PNEC_mg_L", "PNEC_ug_L", "Derivation",
    ]
    export_data = pd.DataFrame([{column: result.get(column) for column in export_columns}])
    export_data = export_data.rename(
        columns={"Sensitive_AlogP": "logP", "Sensitive_Prediction_Quality": "预测质量"}
    )
    safe_cas = re.sub(r"[^A-Za-z0-9_.-]+", "_", result["CAS"])
    st.download_button(
        "下载PNEC结果（CSV）",
        data=export_data.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"pnec_{safe_cas}.csv",
        mime="text/csv",
    )


logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
logo_left, logo_center, logo_right = st.columns([1, 3, 1])
with logo_center:
    st.image(str(logo_path), use_container_width=True)

st.markdown(
    '<h1 style="text-align: center;">PFAS水生毒性预测及PNEC估算工具</h1>',
    unsafe_allow_html=True,
)

module = st.sidebar.selectbox("功能模块", ["毒性预测", "PNEC估算"])
if module == "毒性预测":
    render_toxicity_module()
else:
    render_pnec_module()
render_reference()

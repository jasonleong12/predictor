"""基于四个水生类群慢性毒性和评估因子法的PNEC推导。"""

import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from model import MODEL_CONFIGS, find_compound_record, load_dataset


PNEC_DATA_PATH = Path(__file__).resolve().parent / "data" / "pnec.csv"
CHRONIC_MODEL_KEYS = (
    "zebrafish_chronic",
    "water_flea_chronic",
    "midge_chronic",
    "algae_chronic",
)
ASSESSMENT_FACTOR = 10


def load_pnec_dataset() -> pd.DataFrame:
    """读取Table S11中的188种化合物PNEC及敏感性排序。"""
    data = pd.read_csv(PNEC_DATA_PATH, encoding="utf-8-sig")
    if len(data) != 188 or data["CAS"].nunique() != 188:
        raise ValueError("Table S11应包含188种不重复化合物。")
    return data


def find_pnec_record(cas: Any, dataset: pd.DataFrame = None):
    if cas is None or not str(cas).strip():
        return None
    data = load_pnec_dataset() if dataset is None else dataset
    target = str(cas).strip().casefold()
    matches = data.loc[data["CAS"].astype(str).str.strip().str.casefold() == target]
    if matches.empty:
        return None
    return matches.iloc[0]


def classify_pnec_use(data_type: str, quality: Any, ad_status: Any) -> str:
    """根据最敏感类群的数据质量确定PNEC的建议使用等级。"""
    if data_type == "实测数据":
        return "实测数据推导"
    if ad_status == "Outside AD" or quality in ("Poor", "Bad/Unreliable"):
        return "不建议定量使用"
    if quality == "Moderate" or quality is None or ad_status is None:
        return "低可靠性筛查参考值"
    return "正常输出（筛查用途）"


def derive_pnec(cas: Any) -> Dict[str, Any]:
    """以慢性源表pMCV选择最敏感类群，并按其MCV/10推导PNEC。"""
    record = find_pnec_record(cas)
    if record is None:
        raise ValueError("未找到该CAS。PNEC模块目前仅支持Table S11中的188种化合物。")

    taxa_details = []
    for model_key in CHRONIC_MODEL_KEYS:
        config = MODEL_CONFIGS[model_key]
        toxicity_record = find_compound_record(model_key, cas)
        if toxicity_record is None:
            raise ValueError(f"{config.species_label}慢性模型中缺少化合物{cas}的记录。")

        is_measured = str(toxicity_record["Status"]) != "Untested"
        p_mcv = float(
            toxicity_record["Exp_pTox"] if is_measured else toxicity_record["SI_Pred_pTox"]
        )
        mw = float(toxicity_record["MW"])
        alogp = float(toxicity_record["AlogP"])
        mcv_mg_l = (10.0 ** (-p_mcv)) * mw
        taxa_details.append(
            {
                "Model_Key": model_key,
                "Species": config.species_label,
                "Sheet": config.sheet,
                "pMCV": p_mcv,
                "MW": mw,
                "AlogP": alogp,
                "MCV_mg_L": mcv_mg_l,
                "Data_Type": "实测数据" if is_measured else "预测数据",
                "AD_status": None if is_measured else str(toxicity_record["AD_status"]),
                "Prediction_Quality": (
                    None if is_measured else str(toxicity_record["Prediction_Quality"])
                ),
                "SI_h": float(toxicity_record["SI_h"]),
            }
        )

    sensitive = max(taxa_details, key=lambda item: item["pMCV"])
    sensitive_model_difference = abs(sensitive["pMCV"] - float(record["Rank1_pMCV"]))

    # 以慢性源表pMCV及其对应MW重新计算MCV，不再采用Table S11的MCV或PNEC。
    mcv_mg_l = sensitive["MCV_mg_L"]
    pnec_mg_l = mcv_mg_l / ASSESSMENT_FACTOR

    for detail in taxa_details:
        detail["Used_for_PNEC"] = detail["Model_Key"] == sensitive["Model_Key"]

    return {
        "CAS": str(record["CAS"]),
        "AF": ASSESSMENT_FACTOR,
        "MCV_mg_L": mcv_mg_l,
        "MCV_ug_L": mcv_mg_l * 1000.0,
        "PNEC_mg_L": pnec_mg_l,
        "PNEC_ug_L": pnec_mg_l * 1000.0,
        "Sensitive_Model_Key": sensitive["Model_Key"],
        "Sensitive_Species": sensitive["Species"],
        "Sensitive_Sheet": sensitive["Sheet"],
        "Sensitive_AlogP": sensitive["AlogP"],
        "Sensitive_pMCV": sensitive["pMCV"],
        "Sensitive_Model_pMCV": sensitive["pMCV"],
        "Sensitive_Model_Difference": sensitive_model_difference,
        "Sensitive_Data_Type": sensitive["Data_Type"],
        "Sensitive_AD_status": sensitive["AD_status"],
        "Sensitive_Prediction_Quality": sensitive["Prediction_Quality"],
        "PNEC_Use_Level": classify_pnec_use(
            sensitive["Data_Type"],
            sensitive["Prediction_Quality"],
            sensitive["AD_status"],
        ),
        "TableS11_Rank1_pMCV": float(record["Rank1_pMCV"]),
        "TableS11_MCV_mg_L": float(record["MCV_mg_L"]),
        "TableS11_PNEC_mg_L": float(record["PNEC_mg_L"]),
        "Taxa_Details": taxa_details,
        "Derivation": "PNEC = 慢性源表最敏感类群MCV / AF = MCV / 10",
    }

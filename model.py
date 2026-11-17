"""六种PFAS水生毒性固定QSAR方程、应用域及结果质量评价。"""

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
EFFECT_LEVEL = "个体水平表型效应（例如生长、形态、繁殖、死亡、行为和发育等）"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    sheet: str
    species_group: str
    species_label: str
    species_latin: str
    toxicity_type: str
    endpoint: str
    descriptors: Tuple[str, ...]
    coefficients: Mapping[str, float]
    intercept: float
    h_star: float
    equation: str
    expected_total: int
    expected_untested: int

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"{self.key}.csv"


def _coefficients(**values: float) -> Mapping[str, float]:
    return MappingProxyType(values)


_MODEL_CONFIGS = {
    "zebrafish_acute": ModelConfig(
        key="zebrafish_acute", sheet="Table S3", species_group="zebrafish",
        species_label="斑马鱼（Danio rerio）", species_latin="Danio rerio",
        toxicity_type="急性", endpoint="pMAV",
        descriptors=("ATSC1m", "GATS7i", "SpMin3_Bh(s)"),
        coefficients=_coefficients(ATSC1m=-0.0047, GATS7i=1.7672, **{"SpMin3_Bh(s)": 1.7729}),
        intercept=-1.2611, h_star=0.4,
        equation="pMAV = -0.0047 × ATSC1m + 1.7672 × GATS7i + 1.7729 × SpMin3_Bh(s) - 1.2611",
        expected_total=196, expected_untested=158,
    ),
    "daphnia_acute": ModelConfig(
        key="daphnia_acute", sheet="Table S4", species_group="daphnia_magna",
        species_label="大型溞（Daphnia magna）", species_latin="Daphnia magna",
        toxicity_type="急性", endpoint="pMAV",
        descriptors=("VR1_Dzs", "P_VSA_MR_4", "P_VSA_s_3"),
        coefficients=_coefficients(VR1_Dzs=0.0031, P_VSA_MR_4=0.0691, P_VSA_s_3=0.1879),
        intercept=-1.5112, h_star=0.48,
        equation="pMAV = 0.0031 × VR1_Dzs + 0.0691 × P_VSA_MR_4 + 0.1879 × P_VSA_s_3 - 1.5112",
        expected_total=193, expected_untested=160,
    ),
    "zebrafish_chronic": ModelConfig(
        key="zebrafish_chronic", sheet="Table S5", species_group="zebrafish",
        species_label="斑马鱼（Danio rerio）", species_latin="Danio rerio",
        toxicity_type="慢性", endpoint="pMCV",
        descriptors=("MATS7p", "hmin", "MLFER_BO"),
        coefficients=_coefficients(MATS7p=5.497, hmin=-8.9986, MLFER_BO=0.8872),
        intercept=5.8687, h_star=0.6667,
        equation="pMCV = 5.497 × MATS7p - 8.9986 × hmin + 0.8872 × MLFER_BO + 5.8687",
        expected_total=192, expected_untested=169,
    ),
    "water_flea_chronic": ModelConfig(
        key="water_flea_chronic", sheet="Table S6", species_group="water_flea",
        species_label="水溞", species_latin="Water flea",
        toxicity_type="慢性", endpoint="pMCV",
        descriptors=("MPC7", "P_VSA_m_2"),
        coefficients=_coefficients(MPC7=0.0488, P_VSA_m_2=0.0454),
        intercept=-0.7522, h_star=0.5,
        equation="pMCV = 0.0488 × MPC7 + 0.0454 × P_VSA_m_2 - 0.7522",
        expected_total=199, expected_untested=176,
    ),
    "midge_chronic": ModelConfig(
        key="midge_chronic", sheet="Table S7", species_group="midge",
        species_label="摇蚊", species_latin="Midge",
        toxicity_type="慢性", endpoint="pMCV",
        descriptors=("VE3_Dzi", "AVP-0", "SM03_AEA(dm)"),
        coefficients=_coefficients(VE3_Dzi=-0.3096, **{"AVP-0": 54.8244, "SM03_AEA(dm)": 0.5197}),
        intercept=-24.207, h_star=0.6,
        equation="pMCV = -0.3096 × VE3_Dzi + 54.8244 × AVP-0 + 0.5197 × SM03_AEA(dm) - 24.207",
        expected_total=194, expected_untested=168,
    ),
    "algae_chronic": ModelConfig(
        key="algae_chronic", sheet="Table S8", species_group="algae",
        species_label="绿藻", species_latin="Green algae",
        toxicity_type="慢性", endpoint="pMCV",
        descriptors=("AATSC0i", "SpMin3_Bhs", "GATS2m"),
        coefficients=_coefficients(AATSC0i=0.9305, SpMin3_Bhs=2.4397, GATS2m=-2.7571),
        intercept=-5.7124, h_star=0.5217,
        equation="pMCV = 0.9305 × AATSC0i + 2.4397 × SpMin3_Bhs - 2.7571 × GATS2m - 5.7124",
        expected_total=195, expected_untested=166,
    ),
}
MODEL_CONFIGS = MappingProxyType(_MODEL_CONFIGS)


def get_config(model_key: str) -> ModelConfig:
    if model_key not in MODEL_CONFIGS:
        raise ValueError("所选模型不存在。")
    return MODEL_CONFIGS[model_key]


def load_dataset(model_key: str) -> pd.DataFrame:
    """读取指定模型对应的SI建模数据和待预测数据。"""
    config = get_config(model_key)
    data = pd.read_csv(config.data_path, encoding="utf-8-sig")
    untested_count = int((data["Status"] == "Untested").sum())
    if len(data) != config.expected_total or untested_count != config.expected_untested:
        raise ValueError(f"{config.sheet}的数据记录数与模型配置不一致。")
    return data


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}必须是有限数值。")
    return number


def _descriptor_values(config: ModelConfig, descriptors: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(_finite_number(descriptors.get(name), name) for name in config.descriptors)


def predict_toxicity(model_key: str, descriptors: Mapping[str, Any]) -> float:
    """直接代入论文表3-1固定方程，不训练或调整模型。"""
    config = get_config(model_key)
    values = _descriptor_values(config, descriptors)
    result = config.intercept + sum(
        config.coefficients[name] * value for name, value in zip(config.descriptors, values)
    )
    if not math.isfinite(result):
        raise ValueError("输入描述符导致的预测结果超出可计算范围。")
    return float(result)


def convert_toxicity(p_tox: Any, mw: Optional[Any] = None) -> Dict[str, Optional[float]]:
    """将负对数毒性换算为mmol/L、µmol/L和mg/L。"""
    p_value = _finite_number(p_tox, "毒性负对数值")
    try:
        tox_mmol_l = 10.0 ** (-p_value)
    except OverflowError as exc:
        raise ValueError("毒性负对数值绝对值过大，无法换算浓度。") from exc
    if not math.isfinite(tox_mmol_l):
        raise ValueError("毒性负对数值绝对值过大，无法换算浓度。")

    tox_mg_l = None
    if mw is not None:
        mw_value = _finite_number(mw, "MW")
        if mw_value <= 0:
            raise ValueError("MW必须大于0。")
        tox_mg_l = tox_mmol_l * mw_value
    return {
        "Tox_mmol_L": tox_mmol_l,
        "Tox_umol_L": tox_mmol_l * 1000.0,
        "Tox_mg_L": tox_mg_l,
    }


def calculate_leverage(
    model_key: str, descriptors: Mapping[str, Any], dataset: Optional[pd.DataFrame] = None
) -> float:
    """用Training样本的结构矩阵计算杠杆值，不估计模型系数。"""
    config = get_config(model_key)
    values = _descriptor_values(config, descriptors)
    data = load_dataset(model_key) if dataset is None else dataset
    training = data.loc[data["Status"] == "Training"]
    if len(training) <= len(config.descriptors):
        raise ValueError(f"{config.sheet}的有效结构样本不足，无法计算杠杆值。")

    descriptor_matrix = training.loc[:, config.descriptors].to_numpy(dtype=float)
    x_train = np.column_stack((np.ones(len(training)), descriptor_matrix))
    x_new = np.array([1.0, *values], dtype=float)
    xtx_inverse = np.linalg.inv(x_train.T @ x_train)
    leverage = float(x_new @ xtx_inverse @ x_new.T)
    if not math.isfinite(leverage):
        raise ValueError("无法根据当前输入计算杠杆值。")
    return leverage


def find_compound_record(
    model_key: str, cas: Any, dataset: Optional[pd.DataFrame] = None
) -> Optional[pd.Series]:
    """在指定模型的SI数据中按CAS查找化合物。"""
    if cas is None or not str(cas).strip():
        return None
    data = load_dataset(model_key) if dataset is None else dataset
    target = str(cas).strip().casefold()
    matches = data.loc[data["CAS"].astype(str).str.strip().str.casefold() == target]
    if matches.empty:
        return None
    return matches.iloc[0]


def descriptors_match_record(
    config: ModelConfig, record: pd.Series, descriptors: Mapping[str, Any]
) -> bool:
    """仅在描述符与SI一致时沿用该记录的实测值或PRI结果。"""
    values = _descriptor_values(config, descriptors)
    return all(
        math.isclose(value, float(record[name]), rel_tol=1e-10, abs_tol=1e-10)
        for name, value in zip(config.descriptors, values)
    )


def predict(
    model_key: str,
    cas: Any,
    descriptors: Mapping[str, Any],
    mw: Optional[Any] = None,
    alogp: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算毒性，并返回实测参考值、SI质量结果或自定义应用域结果。"""
    config = get_config(model_key)
    if cas is None or not str(cas).strip():
        raise ValueError("CAS不能为空；无CAS化合物可使用SI中的NOCAS编号。")

    values = _descriptor_values(config, descriptors)
    clean_descriptors = dict(zip(config.descriptors, values))
    p_tox = predict_toxicity(model_key, clean_descriptors)
    result = {
        "Model_Key": model_key,
        "CAS": str(cas).strip(),
        "Species": config.species_label,
        "Toxicity_Type": config.toxicity_type,
        "Endpoint": config.endpoint,
        "Effect_Level": EFFECT_LEVEL,
        "MW": None if mw is None else _finite_number(mw, "MW"),
        "AlogP": None if alogp is None else _finite_number(alogp, "AlogP"),
        "pTox": p_tox,
        "h_star": config.h_star,
        "Data_Type": "预测数据",
        **clean_descriptors,
    }
    result.update(convert_toxicity(p_tox, mw))

    record = find_compound_record(model_key, cas)
    if record is not None and descriptors_match_record(config, record, clean_descriptors):
        status = str(record["Status"])
        result.update(
            {
                "h": float(record["SI_h"]),
                "Reference_Match": True,
                "SI_Pred_pTox": float(record["SI_Pred_pTox"]),
            }
        )
        if status == "Untested":
            result.update(
                {
                    "AD_status": str(record["AD_status"]),
                    "Prediction_Quality": str(record["Prediction_Quality"]),
                    "Quality_Source": f"SI {config.sheet}（CAS及描述符匹配）",
                    "Score_MAE_LOO": int(record["Score_MAE_LOO"]),
                    "Score_Proximity_Train_YObs_Mean": int(
                        record["Score_Proximity_Train_YObs_Mean"]
                    ),
                    "Score_Similarity_AD": int(record["Score_Similarity_AD"]),
                    "Composite_Score": int(record["Composite_Score"]),
                    "Euclidean_Distance": float(record["Euclidean_Distance"]),
                    "Generalized_Jaccard_Coeff": float(record["Generalized_Jaccard_Coeff"]),
                    "Yobs_Test_Distribution": str(record["Yobs_Test_Distribution"]),
                }
            )
        else:
            exp_ptox = float(record["Exp_pTox"])
            exp_concentrations = convert_toxicity(exp_ptox, mw)
            std_residual = float(record["SI_Std_Resid"])
            result.update(
                {
                    "Data_Type": "实测数据",
                    "AD_status": (
                        "In"
                        if result["h"] < config.h_star and abs(std_residual) <= 3.0
                        else "Outside AD"
                    ),
                    "Prediction_Quality": None,
                    "Quality_Source": f"SI {config.sheet}实测数据；不适用PRI评级",
                    "Exp_pTox": exp_ptox,
                    "Exp_Tox_mmol_L": exp_concentrations["Tox_mmol_L"],
                    "Exp_Tox_umol_L": exp_concentrations["Tox_umol_L"],
                    "Exp_Tox_mg_L": exp_concentrations["Tox_mg_L"],
                    "SI_Std_Resid": std_residual,
                    "Prediction_Minus_Exp": p_tox - exp_ptox,
                }
            )
    else:
        leverage = calculate_leverage(model_key, clean_descriptors)
        result.update(
            {
                "h": leverage,
                "AD_status": "In" if leverage < config.h_star else "Outside AD",
                "Prediction_Quality": None,
                "Quality_Source": "实时杠杆值计算；PRI未评估",
                "Reference_Match": False,
                "SI_Pred_pTox": None,
            }
        )
    return result

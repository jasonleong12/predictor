"""Table S11 PNEC推导及四类群质量关联测试。"""

import unittest

import pnec


class PNECDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pnec.load_pnec_dataset()

    def test_dataset_and_all_188_derivations(self):
        self.assertEqual(len(self.data), 188)
        self.assertEqual(self.data["CAS"].nunique(), 188)
        for _, row in self.data.iterrows():
            with self.subTest(cas=row["CAS"]):
                result = pnec.derive_pnec(row["CAS"])
                self.assertEqual(result["AF"], 10)
                self.assertEqual(len(result["Taxa_Details"]), 4)
                used = [d for d in result["Taxa_Details"] if d["Used_for_PNEC"]]
                self.assertEqual(len(used), 1)
                sensitive = used[0]
                self.assertEqual(sensitive["Species"], result["Sensitive_Species"])
                self.assertEqual(result["Sensitive_pMCV"], sensitive["pMCV"])
                self.assertEqual(result["MCV_mg_L"], sensitive["MCV_mg_L"])
                self.assertEqual(
                    result["Sensitive_pMCV"], max(d["pMCV"] for d in result["Taxa_Details"])
                )
                tolerance = max(1e-15, abs(result["PNEC_mg_L"]) * 1e-12)
                self.assertAlmostEqual(
                    result["PNEC_mg_L"], result["MCV_mg_L"] / 10, delta=tolerance
                )
                self.assertAlmostEqual(
                    result["PNEC_ug_L"], result["PNEC_mg_L"] * 1000, places=12
                )

    def test_predicted_sensitive_taxon_quality(self):
        result = pnec.derive_pnec("10116-92-4")
        self.assertEqual(result["Sensitive_Species"], "斑马鱼")
        self.assertEqual(result["Sensitive_Data_Type"], "预测数据")
        self.assertEqual(result["Sensitive_Prediction_Quality"], "Good")
        self.assertEqual(result["Sensitive_AD_status"], "In")
        self.assertAlmostEqual(result["Sensitive_pMCV"], 3.89283183666974, places=10)
        self.assertAlmostEqual(result["PNEC_mg_L"], 0.00695229072218063, places=12)

    def test_measured_sensitive_taxon_is_explicit(self):
        result = pnec.derive_pnec("1763-23-1")
        self.assertEqual(result["Sensitive_Species"], "摇蚊")
        self.assertEqual(result["Sensitive_Data_Type"], "实测数据")
        self.assertIsNone(result["Sensitive_Prediction_Quality"])
        self.assertIsNone(result["Sensitive_AD_status"])
        self.assertAlmostEqual(result["Sensitive_pMCV"], 5.1982, places=6)
        self.assertAlmostEqual(result["PNEC_mg_L"], 0.0003167493332973899, places=15)

    def test_source_differences_are_retained(self):
        results = [pnec.derive_pnec(cas) for cas in self.data["CAS"]]
        self.assertEqual(sum(r["Sensitive_Model_Difference"] > 1e-6 for r in results), 8)
        self.assertEqual(sum(r["Sensitive_Model_Difference"] > 0.05 for r in results), 5)
        result = pnec.derive_pnec("14477-72-6")
        self.assertEqual(result["Sensitive_Species"], "绿藻")
        self.assertGreater(result["Sensitive_Model_Difference"], 1.0)

    def test_pnec_data_composition_and_quality_warning(self):
        predicted = pnec.derive_pnec("10116-92-4")
        self.assertEqual(predicted["PNEC_Data_Composition"], "模型预测推导")
        self.assertIsNone(predicted["PNEC_Quality_Warning"])

        measured = pnec.derive_pnec("1763-23-1")
        self.assertEqual(measured["PNEC_Data_Composition"], "实测数据推导")
        self.assertIsNone(measured["PNEC_Quality_Warning"])

        mixed = pnec.derive_pnec("151772-59-7")
        self.assertEqual(mixed["PNEC_Data_Composition"], "实测与预测数据联合推导")

        self.assertIsNone(pnec.pnec_quality_warning("Moderate", "In"))
        self.assertIsNone(pnec.pnec_quality_warning("Good", "In"))
        warning = "最敏感类群的预测超出应用域或预测质量较差，所得PNEC的不确定性较高，应谨慎使用。"
        self.assertEqual(pnec.pnec_quality_warning("Good", "Outside AD"), warning)
        self.assertEqual(pnec.pnec_quality_warning("Bad/Unreliable", "In"), warning)

    def test_unknown_cas(self):
        with self.assertRaises(ValueError):
            pnec.derive_pnec("NOT-IN-S11")
        with self.assertRaises(ValueError):
            pnec.derive_pnec("")


if __name__ == "__main__":
    unittest.main()

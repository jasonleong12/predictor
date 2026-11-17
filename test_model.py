"""六种固定QSAR方程、应用域、实测值和SI质量记录的回归测试。"""

import unittest

import model


EXPECTED_COUNTS = {
    "zebrafish_acute": (196, 30, 8, 158),
    "daphnia_acute": (193, 25, 8, 160),
    "zebrafish_chronic": (192, 18, 5, 169),
    "water_flea_chronic": (199, 18, 5, 176),
    "midge_chronic": (194, 20, 6, 168),
    "algae_chronic": (195, 23, 6, 166),
}


class SixQSARModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {key: model.load_dataset(key) for key in model.MODEL_CONFIGS}

    def descriptors(self, config, row):
        return {name: row[name] for name in config.descriptors}

    def test_all_dataset_counts_and_quality_completeness(self):
        self.assertEqual(set(EXPECTED_COUNTS), set(model.MODEL_CONFIGS))
        for key, counts in EXPECTED_COUNTS.items():
            with self.subTest(model=key):
                data = self.datasets[key]
                actual = (
                    len(data),
                    int((data["Status"] == "Training").sum()),
                    int((data["Status"] == "Test").sum()),
                    int((data["Status"] == "Untested").sum()),
                )
                self.assertEqual(actual, counts)
                self.assertEqual(data["CAS"].nunique(), len(data))
                quality = data.loc[data["Status"] == "Untested", "Prediction_Quality"]
                self.assertFalse(quality.isna().any())
                self.assertTrue(
                    set(quality).issubset({"Good", "Moderate", "Poor", "Bad/Unreliable"})
                )

    def test_fixed_coefficients_are_read_only(self):
        for key, config in model.MODEL_CONFIGS.items():
            with self.subTest(model=key):
                self.assertEqual(len(config.coefficients), len(config.descriptors))
                with self.assertRaises(TypeError):
                    config.coefficients[config.descriptors[0]] = 0.0

    def test_fixed_equations_reproduce_untested_si_predictions(self):
        allowed_exceptions = {("daphnia_acute", "10116-92-4")}
        for key, config in model.MODEL_CONFIGS.items():
            untested = self.datasets[key].loc[self.datasets[key]["Status"] == "Untested"]
            for _, row in untested.iterrows():
                with self.subTest(model=key, cas=row["CAS"]):
                    if (key, row["CAS"]) in allowed_exceptions:
                        continue
                    predicted = model.predict_toxicity(key, self.descriptors(config, row))
                    self.assertAlmostEqual(predicted, row["SI_Pred_pTox"], places=8)

    def test_documented_daphnia_exception_is_not_hidden(self):
        config = model.MODEL_CONFIGS["daphnia_acute"]
        row = self.datasets["daphnia_acute"].loc[
            self.datasets["daphnia_acute"]["CAS"] == "10116-92-4"
        ].iloc[0]
        predicted = model.predict_toxicity("daphnia_acute", self.descriptors(config, row))
        self.assertAlmostEqual(predicted, -1.323138669, places=8)
        self.assertAlmostEqual(row["SI_Pred_pTox"], 1.68469483309, places=8)

    def test_all_measured_compounds_return_experimental_data(self):
        for key, config in model.MODEL_CONFIGS.items():
            measured = self.datasets[key].loc[self.datasets[key]["Status"] != "Untested"]
            for _, row in measured.iterrows():
                with self.subTest(model=key, cas=row["CAS"]):
                    result = model.predict(
                        key, row["CAS"], self.descriptors(config, row), row["MW"]
                    )
                    self.assertTrue(result["Reference_Match"])
                    self.assertEqual(result["Data_Type"], "实测数据")
                    self.assertEqual(result["Exp_pTox"], row["Exp_pTox"])
                    self.assertIsNone(result["Prediction_Quality"])
                    expected_ad = (
                        "In"
                        if row["SI_h"] < config.h_star and abs(row["SI_Std_Resid"]) <= 3.0
                        else "Outside AD"
                    )
                    self.assertEqual(result["AD_status"], expected_ad)
                    converted = model.convert_toxicity(row["Exp_pTox"], row["MW"])
                    self.assertAlmostEqual(
                        result["Exp_Tox_mg_L"], converted["Tox_mg_L"], places=10
                    )

    def test_training_leverages_match_si(self):
        for key, config in model.MODEL_CONFIGS.items():
            training = self.datasets[key].loc[self.datasets[key]["Status"] == "Training"]
            for _, row in training.iterrows():
                with self.subTest(model=key, cas=row["CAS"]):
                    leverage = model.calculate_leverage(
                        key, self.descriptors(config, row), self.datasets[key]
                    )
                    self.assertAlmostEqual(leverage, row["SI_h"], places=4)

    def test_known_untested_cas_uses_si_quality(self):
        for key, config in model.MODEL_CONFIGS.items():
            row = self.datasets[key].loc[self.datasets[key]["Status"] == "Untested"].iloc[0]
            with self.subTest(model=key):
                result = model.predict(
                    key, row["CAS"], self.descriptors(config, row), row["MW"]
                )
                self.assertTrue(result["Reference_Match"])
                self.assertEqual(result["Prediction_Quality"], row["Prediction_Quality"])
                self.assertEqual(result["AD_status"], row["AD_status"])
                self.assertEqual(result["h"], row["SI_h"])
                self.assertEqual(result["Endpoint"], config.endpoint)

    def test_modified_descriptors_do_not_reuse_measurements_or_pri(self):
        for key, config in model.MODEL_CONFIGS.items():
            row = self.datasets[key].iloc[0]
            descriptors = self.descriptors(config, row)
            descriptors[config.descriptors[0]] += 1.0
            with self.subTest(model=key):
                result = model.predict(key, row["CAS"], descriptors, row["MW"])
                self.assertFalse(result["Reference_Match"])
                self.assertIsNone(result["Prediction_Quality"])
                self.assertNotIn("Exp_pTox", result)
                self.assertNotEqual(result["h"], row["SI_h"])

    def test_custom_input_has_real_time_ad_but_no_pri(self):
        for key, config in model.MODEL_CONFIGS.items():
            row = self.datasets[key].loc[self.datasets[key]["Status"] == "Training"].iloc[0]
            with self.subTest(model=key):
                result = model.predict(
                    key, "CUSTOM-001", self.descriptors(config, row), row["MW"]
                )
                self.assertFalse(result["Reference_Match"])
                self.assertIsNone(result["Prediction_Quality"])
                self.assertAlmostEqual(result["h"], row["SI_h"], places=4)
                self.assertEqual(result["AD_status"], "In")

    def test_unit_conversion(self):
        converted = model.convert_toxicity(2.0, 100.0)
        self.assertAlmostEqual(converted["Tox_mmol_L"], 0.01)
        self.assertAlmostEqual(converted["Tox_umol_L"], 10.0)
        self.assertAlmostEqual(converted["Tox_mg_L"], 1.0)
        self.assertIsNone(model.convert_toxicity(2.0)["Tox_mg_L"])

    def test_invalid_inputs(self):
        config = model.MODEL_CONFIGS["zebrafish_acute"]
        for bad_value in (None, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=bad_value):
                with self.assertRaises(ValueError):
                    model.predict_toxicity(
                        "zebrafish_acute",
                        {"ATSC1m": bad_value, "GATS7i": 1.0, "SpMin3_Bh(s)": 0.0},
                    )
        for bad_mw in (0, -1, float("nan")):
            with self.subTest(mw=bad_mw):
                with self.assertRaises(ValueError):
                    model.convert_toxicity(1.0, bad_mw)
        with self.assertRaises(ValueError):
            model.predict("missing_model", "CAS", {})
        with self.assertRaises(ValueError):
            model.predict(config.key, "", {name: 0.0 for name in config.descriptors})


if __name__ == "__main__":
    unittest.main()

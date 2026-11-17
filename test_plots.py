"""毒性及PNEC图表的数据准备和Altair序列化测试。"""

import unittest

import model
import plots
import pnec


class PlotGenerationTests(unittest.TestCase):
    def test_all_model_charts_can_be_generated(self):
        for key, config in model.MODEL_CONFIGS.items():
            with self.subTest(model=key):
                data = model.load_dataset(key)
                plot_data = plots.prepare_model_plot_data(data, config)
                self.assertEqual(len(plot_data), len(data))
                self.assertEqual(set(plot_data["Plot_Data_Type"]), {"实测数据", "预测数据"})
                first = plot_data.iloc[0]
                expected = model.predict_toxicity(
                    key, {name: first[name] for name in config.descriptors}
                )
                self.assertAlmostEqual(first["Fixed_Prediction"], expected, places=10)
                measured_cas = data.loc[data["Status"] != "Untested", "CAS"].iloc[0]
                charts = [
                    plots.toxicity_scatter_chart(plot_data, config.endpoint, measured_cas),
                    plots.williams_chart(plot_data, config.h_star, measured_cas),
                    plots.chemical_space_chart(plot_data, measured_cas),
                ]
                for chart in charts:
                    chart_dict = chart.to_dict()
                    self.assertIn("height", chart_dict)

    def test_untested_cas_has_no_measured_residual_highlight(self):
        config = model.MODEL_CONFIGS["zebrafish_acute"]
        data = plots.prepare_model_plot_data(model.load_dataset(config.key), config)
        untested_cas = data.loc[data["Status"] == "Untested", "CAS"].iloc[0]
        plots.toxicity_scatter_chart(data, config.endpoint, untested_cas).to_dict()
        plots.williams_chart(data, config.h_star, untested_cas).to_dict()
        plots.chemical_space_chart(data, untested_cas).to_dict()

    def test_pnec_taxa_chart(self):
        result = pnec.derive_pnec("10116-92-4")
        chart_dict = plots.pnec_taxa_chart(result["Taxa_Details"]).to_dict()
        self.assertEqual(chart_dict["height"], 300)


if __name__ == "__main__":
    unittest.main()

import unittest
from ReusableComponents.CustomModules.OpenAI.admin_handler import ModelPricing

class TestModelPricing(unittest.TestCase):

    def test_filter_by_cost(self):
        pricing = {"input": 2.50, "cached_input": 1.25, "output": 10.00}
        
        # Test case where cost is within range
        self.assertTrue(ModelPricing.filter_by_cost(pricing, "input", 3.00, 2.00))
        
        # Test case where cost is above max_cost
        self.assertFalse(ModelPricing.filter_by_cost(pricing, "input", 2.00, 1.00))
        
        # Test case where cost is below min_cost
        self.assertFalse(ModelPricing.filter_by_cost(pricing, "input", 3.00, 3.00))
        
        # Test case where cost_type is not in pricing
        self.assertFalse(ModelPricing.filter_by_cost(pricing, "training", 3.00, 1.00))

    def test_get_pricing(self):
        # Test case for specific model and version
        pricing = ModelPricing.get_pricing(model_type="gpt-4o", model_name="gpt-4o-2024-08-06")
        self.assertEqual(pricing["name"], "gpt-4o-2024-08-06")
        self.assertEqual(pricing["pricing"]["input"], 2.50)
        
        # Test case for filtering by cost
        pricing_list = ModelPricing.get_pricing(cost_type="input", max_cost=0.20)
        self.assertTrue(any(p["name"] == "gpt-4o-mini-2024-07-18" for p in pricing_list))
        
        # Test case for no matching data
        with self.assertRaises(ValueError):
            ModelPricing.get_pricing(model_type="non-existent-model")

    def test_calculate_cost(self):
        # Test case for calculating cost
        cost = ModelPricing.calculate_cost("gpt-4o", "gpt-4o-2024-08-06", 1_000_000, 500_000, 2_000_000)
        self.assertAlmostEqual(cost["input"], 2.50)
        self.assertAlmostEqual(cost["cached_input"], 0.625)
        self.assertAlmostEqual(cost["output"], 20.00)
        self.assertAlmostEqual(cost["total"], 23.125)
        
        # Test case for non-existent model
        with self.assertRaises(ValueError):
            ModelPricing.calculate_cost("non-existent-model", "version", 1_000_000)

if __name__ == "__main__":
    unittest.main()
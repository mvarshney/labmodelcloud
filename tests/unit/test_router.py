from unittest import TestCase
from data_plane.gateway.services.router import WeightedRouter

class TestWeightedRouter(TestCase):
    def setUp(self):
        self.router = WeightedRouter()

    def test_initial_weights(self):
        # Test that the initial weights are set correctly
        self.router.set_weights({'model_a': 0.5, 'model_b': 0.5})
        self.assertEqual(self.router.weights, {'model_a': 0.5, 'model_b': 0.5})

    def test_select_model(self):
        # Test model selection based on weights
        self.router.set_weights({'model_a': 0.7, 'model_b': 0.3})
        selections = {'model_a': 0, 'model_b': 0}
        for _ in range(1000):
            selected_model = self.router.select_model()
            selections[selected_model] += 1
        self.assertGreater(selections['model_a'], selections['model_b'])

    def test_update_weights(self):
        # Test updating weights
        self.router.set_weights({'model_a': 0.4, 'model_b': 0.6})
        self.router.update_weights({'model_a': 0.5, 'model_b': 0.5})
        self.assertEqual(self.router.weights, {'model_a': 0.5, 'model_b': 0.5})
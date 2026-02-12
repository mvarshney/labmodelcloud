from unittest import TestCase
from deployment_service.services.deployer import Deployer

class TestDeployment(TestCase):
    def setUp(self):
        self.deployer = Deployer()

    def test_deploy_model(self):
        model_name = "recommendation_v1"
        version = "1"
        response = self.deployer.deploy_model(model_name, version)
        self.assertTrue(response['success'])
        self.assertEqual(response['model_name'], model_name)
        self.assertEqual(response['version'], version)

    def test_unload_model(self):
        model_name = "recommendation_v1"
        version = "1"
        response = self.deployer.unload_model(model_name, version)
        self.assertTrue(response['success'])
        self.assertEqual(response['model_name'], model_name)
        self.assertEqual(response['version'], version)

    def test_deploy_invalid_model(self):
        model_name = "invalid_model"
        version = "1"
        response = self.deployer.deploy_model(model_name, version)
        self.assertFalse(response['success'])
        self.assertIn('error', response)
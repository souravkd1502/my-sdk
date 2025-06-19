import unittest
from app import create_app

class RoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()

    def test_execute_route(self):
        response = self.app.post('/execute', json={'command': 'ls'})
        self.assertEqual(response.status_code, 200)

import unittest
from app.services import ssh_client

class ServicesTestCase(unittest.TestCase):
    def test_execute_command(self):
        output, error = ssh_client.execute_command('ls')
        self.assertIsInstance(output, str)

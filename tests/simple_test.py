import unittest
import importlib.util

class TestRequiredModules(unittest.TestCase):
    def test_pandas_installed(self):
        """Check if pandas is installed."""
        loader = importlib.util.find_spec("pandas")
        self.assertIsNotNone(loader, "pandas is not installed")

    def test_openpyxl_installed(self):
        """Check if openpyxl is installed."""
        loader = importlib.util.find_spec("openpyxl")
        self.assertIsNotNone(loader, "openpyxl is not installed")

if __name__ == '__main__':
    unittest.main()

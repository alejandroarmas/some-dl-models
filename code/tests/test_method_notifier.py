import unittest
from code.lib.notifier import MethodNotifier, MLEventType
from unittest.mock import MagicMock


class TestMethodNotifier(unittest.TestCase):
    def test_notify(self):

        listener = MagicMock()
        manager = MethodNotifier()

        manager.subscribe(listener, MLEventType("method"))
        manager.notify(MLEventType("method"), {"hello": "world"})

        listener.update.assert_called()


if __name__ == "__main__":
    unittest.main()

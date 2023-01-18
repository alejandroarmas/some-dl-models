import unittest
from unittest.mock import MagicMock

from code.lib.method_notifier import MethodNotifier, MLEventType


class TestMethodNotifier(unittest.TestCase):

    def test_notify(self):

        listener = MagicMock()
        manager = MethodNotifier()

        manager.subscribe(listener, MLEventType("method"))
        manager.notify(MLEventType("method"), {'hello': 'world'})

        listener.update.assert_called()


if __name__ == '__main__':
    unittest.main()

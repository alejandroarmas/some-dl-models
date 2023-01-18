import unittest
from unittest.mock import MagicMock, patch

from code.lib.notifier import MLEventType, MLEventNotifier

class TestBaseNotifier(unittest.TestCase):

    def setUp(cls) -> None:
        cls.num_mock_listeners = 10


    @patch.multiple(MLEventNotifier, __abstractmethods__=set())
    def test_add_subscriptions(self):
        self.manager = MLEventNotifier()
        self.mock_listeners = [MagicMock() for _ in range(0, self.num_mock_listeners)]

        event_type = MLEventType("example-event-name")

        for mock_listener in self.mock_listeners:
            self.manager.subscribe(mock_listener, event_type)
            
        self.assertEqual(len(self.manager.subscribers[event_type.event_str]), self.num_mock_listeners)
        

    @patch.multiple(MLEventNotifier, __abstractmethods__=set())
    def test_remove_subscriptions(self):

        self.manager = MLEventNotifier()
        self.assertEqual(len(self.manager.subscribers), 0)

        self.mock_listeners = [MagicMock() for _ in range(0, self.num_mock_listeners)]

        event_type = MLEventType("example-event-name")

        for mock_listener in self.mock_listeners:
            self.manager.subscribe(mock_listener, event_type)
            
        mock_listener = MagicMock()

        self.manager.subscribe(mock_listener, event_type)
        self.manager.unsubscribe(mock_listener, event_type)

        self.assertEqual(len(self.manager.subscribers[event_type.event_str]), self.num_mock_listeners)

    @patch.multiple(MLEventNotifier, __abstractmethods__=set())
    def test_subscriptions_exist(self):
        
        self.manager = MLEventNotifier()
        event_type = MLEventType("example-event-name")
        mock_listener = MagicMock()
        self.assertFalse(self.manager.subscription_exists(mock_listener, event_type), True)

        self.manager.subscribe(mock_listener, event_type)

        self.assertTrue(self.manager.subscription_exists(mock_listener, event_type), True)


    # def test_event_update(self):

    #     self.manager = MLEventNotifier()
    #     self.mock_listeners = [MagicMock() for _ in range(0, self.num_mock_listeners)]
        
    #     event_string = "example-event-name"
    #     event_type = MLEventType(event_string)

    #     for mock_listener in self.mock_listeners:
    #         self.manager.subscribe(mock_listener, deepcopy(event_type))


    #     self.manager.notify(event_type, {'data': 'run was successful'})

    #     for mock_listener in self.mock_listeners:
    #         self.assertEqual(mock_listener.update.called, True)


if __name__ == '__main__':
    unittest.main()


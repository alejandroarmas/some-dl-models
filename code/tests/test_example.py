import unittest

"""
    To run this class, perform `python -m code.tests.test_example`
    in the command line.
"""
class TestExample(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = [1, 2, 3, 4, 5] 

    def test_example(self):
        self.assertTrue(self.data == [1, 2, 3, 4, 5])

    def test_assert_false(self):
        self.assertFalse(self.data == ['this', 'is', 'wrong'])


    def test_assert_error(self):

        class Error():
            def raise_error(self):
                raise TypeError

        error = Error()

        with self.assertRaises(TypeError):
            error.raise_error()



if __name__ == '__main__':
    unittest.main()

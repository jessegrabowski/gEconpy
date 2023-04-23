import unittest

from gEconpy.classes.transformers import (
    IdentityTransformer,
    IntervalTransformer,
    PositiveTransformer,
)


class TestIdentityTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = IdentityTransformer()

    def test_constrain(self):
        cases = [-1, 1, 0.2]
        for case in cases:
            self.assertEqual(case, self.transformer.constrain(case), msg=f"{case}")

    def test_unconstrain(self):
        cases = [-1, 1, 0.2]
        for case in cases:
            self.assertAlmostEqual(
                case, self.transformer.unconstrain(self.transformer.constrain(case)), msg=f"{case}"
            )


class TestPositiveTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = PositiveTransformer()

    def test_constrain(self):
        cases = [-1, 1, 0.2]
        for case in cases:
            self.assertTrue(self.transformer.constrain(case) >= 0, msg=f"{case}")

    def test_unconstrain(self):
        cases = [-1, 1, 0.2]
        for case in cases:
            self.assertAlmostEqual(
                case, self.transformer.unconstrain(self.transformer.constrain(case)), msg=f"{case}"
            )


class TestIntervalTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = IntervalTransformer()

    def test_constrain(self):
        cases = [-5, 3, 0.2]
        for case in cases:
            constrained = self.transformer.constrain(case)
            self.assertTrue((0 < constrained) & (constrained < 1), msg=f"{case}")

    def test_unconstrain(self):
        cases = [-1, 1, 0.2]
        for case in cases:
            self.assertAlmostEqual(
                case, self.transformer.unconstrain(self.transformer.constrain(case)), msg=f"{case}"
            )


if __name__ == "__main__":
    unittest.main()

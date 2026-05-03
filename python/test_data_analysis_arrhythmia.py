import sys
import unittest
from unittest.mock import MagicMock
import math

# --- Mocking dependencies ---
# NOTE: The development environment lacks 'numpy', 'scipy', and 'pandas', and has no network access
# for pip installations. Therefore, we use 'sys.modules' to provide minimal functional mocks
# required for the tests to run. This approach is recommended for this specific restricted environment.

class MockArray(list):
    """A minimal list-based mock for numpy arrays to support basic arithmetic and vectorization."""
    def __init__(self, data, dtype=None):
        if isinstance(data, (int, float, bool)):
             super().__init__([data])
        else:
             super().__init__(data)
        self.dtype = dtype

    def __getattr__(self, name):
        return MagicMock()

    def __getitem__(self, item):
        res = super().__getitem__(item)
        if isinstance(item, slice):
            return MockArray(res)
        return res

    def astype(self, dtype):
        return self

    def sum(self):
        return sum(self)

    def mean(self):
        return sum(self) / len(self) if len(self) > 0 else float('nan')

    def any(self):
        return any(self)

    def all(self):
        return all(self)

    # Mathematical operations
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x - other for x in self])
        return MockArray([x - y for x, y in zip(self, other)])

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other - x for x in self])
        return MockArray([y - x for x, y in zip(self, other)])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x + other for x in self])
        return MockArray([x + y for x, y in zip(self, other)])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x * other for x in self])
        return MockArray([x * y for x, y in zip(self, other)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: return MockArray([float('inf')] * len(self))
            return MockArray([x / other for x in self])
        return MockArray([x / y if y != 0 else float('inf') for x, y in zip(self, other)])

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other / x if x != 0 else float('inf') for x in self])
        return MockArray([y / x if x != 0 else float('inf') for x, y in zip(self, other)])

    def __pow__(self, other):
        return MockArray([x ** other for x in self])

    def __abs__(self):
        return MockArray([abs(x) for x in self])

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x > other for x in self])
        return MockArray([x > y for x, y in zip(self, other)])

def mock_exp(x):
    if isinstance(x, (int, float)):
        return math.exp(x) if x < 700 else float('inf')
    return MockArray([math.exp(i) if i < 700 else float('inf') for i in x])

def mock_isnan(x):
    if isinstance(x, (int, float)):
        return math.isnan(x)
    return MockArray([math.isnan(i) for i in x])

mock_np = MagicMock()
mock_np.asarray.side_effect = lambda x, dtype=None: MockArray(x, dtype)
mock_np.nan = float('nan')
mock_np.isnan.side_effect = mock_isnan
mock_np.exp.side_effect = mock_exp
mock_np.mean.side_effect = lambda x: sum(x) / len(x) if len(x) > 0 else float('nan')
mock_np.std.side_effect = lambda x, ddof=0: math.sqrt(sum((i - (sum(x)/len(x)))**2 for i in x) / (len(x) - ddof)) if len(x) > ddof else 0.0
mock_np.diff.side_effect = lambda x: MockArray([x[i+1] - x[i] for i in range(len(x)-1)])
mock_np.sqrt.side_effect = lambda x: math.sqrt(x)
mock_np.median.side_effect = lambda x: sorted(x)[len(x)//2] if len(x) > 0 else float('nan')
mock_np.abs.side_effect = lambda x: abs(x)
mock_np.float64 = float

sys.modules['numpy'] = mock_np
sys.modules['pandas'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.cluster'] = MagicMock()
sys.modules['scipy.cluster.hierarchy'] = MagicMock()
sys.modules['scipy.cluster.vq'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

# Now we can import the module to test
import data_analysis

class TestArrhythmiaRisk(unittest.TestCase):
    def test_logistic_score(self):
        # midpoint 0.5, steepness 10
        # at 0.5 should be 0.5
        self.assertAlmostEqual(data_analysis._logistic_score(0.5, 0.5, 10), 0.5)
        # very high value should be close to 1
        self.assertAlmostEqual(data_analysis._logistic_score(100, 0.5, 10), 1.0)
        # very low value should be close to 0
        self.assertAlmostEqual(data_analysis._logistic_score(-100, 0.5, 10), 0.0)

    def test_compute_arrhythmia_risk_insufficient_data(self):
        ibi_ms = [500, 500] # Only 2
        result = data_analysis.compute_arrhythmia_risk(ibi_ms)
        self.assertFalse(result["arrhythmia_data_sufficient"])
        self.assertEqual(result["arrhythmia_quality_flag"], "insufficient_ibi")
        self.assertTrue(math.isnan(result["arrhythmia_risk_score"]))

    def test_compute_arrhythmia_risk_invalid_ibi(self):
        ibi_ms = [0, 0, 0]
        result = data_analysis.compute_arrhythmia_risk(ibi_ms)
        self.assertEqual(result["arrhythmia_quality_flag"], "invalid_ibi")
        self.assertTrue(math.isnan(result["arrhythmia_risk_score"]))

    def test_compute_arrhythmia_risk_regular(self):
        ibi_ms = [500, 500, 500, 500, 500, 500]
        result = data_analysis.compute_arrhythmia_risk(ibi_ms)
        self.assertTrue(result["arrhythmia_data_sufficient"])
        self.assertEqual(result["arrhythmia_quality_flag"], "ok")
        self.assertLess(result["arrhythmia_risk_score"], 0.1)

    def test_compute_arrhythmia_risk_irregular(self):
        ibi_ms = [500, 800, 400, 900, 300, 1000]
        result = data_analysis.compute_arrhythmia_risk(ibi_ms)
        self.assertTrue(result["arrhythmia_data_sufficient"])
        self.assertGreater(result["arrhythmia_risk_score"], 0.5)

    def test_compute_arrhythmia_risk_low_confidence(self):
        ibi_ms = [500, 510, 490] # 3 IBIs
        result = data_analysis.compute_arrhythmia_risk(ibi_ms)
        self.assertTrue(result["arrhythmia_data_sufficient"])
        self.assertEqual(result["arrhythmia_quality_flag"], "low_ibi_count")

    def test_arrhythmia_probability(self):
        ibi_ms = [500, 500, 500, 500, 500, 500]
        prob = data_analysis.arrhythmia_probability(ibi_ms)
        self.assertLess(prob, 0.1)

    def test_arrhythmia_decision(self):
        self.assertTrue(data_analysis.arrhythmia_decision(0.8, 0.5, True))
        self.assertFalse(data_analysis.arrhythmia_decision(0.3, 0.5, True))
        self.assertFalse(data_analysis.arrhythmia_decision(0.8, 0.5, False))
        self.assertFalse(data_analysis.arrhythmia_decision(float('nan'), 0.5, True))

if __name__ == '__main__':
    unittest.main()

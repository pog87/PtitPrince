import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_per_group = 50

    data = pd.DataFrame(
        {
            "category": ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group,
            "value": np.concatenate(
                [
                    np.random.normal(0, 1, n_per_group),
                    np.random.normal(2, 1.5, n_per_group),
                    np.random.normal(-1, 0.8, n_per_group),
                ]
            ),
            "hue": (["X", "Y"] * (n_per_group // 2 + 1))[:n_per_group] * 3,
        }
    )

    return data


@pytest.fixture
def simple_data():
    """Generate simple data with few points for edge case testing."""
    return pd.DataFrame(
        {
            "cat": ["A", "A", "A", "B", "B", "B"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "hue": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )


@pytest.fixture
def paired_data():
    """Repeated-measures sample: each subject measured pre and post."""
    np.random.seed(42)
    n = 30
    subjects = [f"s{i}" for i in range(n)]
    pre = np.random.normal(50, 8, n)
    post = pre + np.random.normal(6, 5, n)  # within-subject change
    return pd.DataFrame(
        {
            "subject": subjects * 2,
            "condition": ["pre"] * n + ["post"] * n,
            "score": np.concatenate([pre, post]),
        }
    )


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically close all plots after each test."""
    yield
    plt.close("all")

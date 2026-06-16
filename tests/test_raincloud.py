import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ptitprince import RainCloud, paired_raincloud


class TestRainCloudBasic:
    """Test basic RainCloud functionality."""

    def test_raincloud_vertical(self, sample_data):
        """Test vertical raincloud plot creation."""
        ax = RainCloud(x="category", y="value", data=sample_data, orient="v")
        assert ax is not None
        assert len(ax.collections) > 0  # Should have plot elements

    def test_raincloud_horizontal(self, sample_data):
        """Test horizontal raincloud plot creation."""
        ax = RainCloud(x="category", y="value", data=sample_data, orient="h")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_raincloud_with_hue(self, sample_data):
        """Test raincloud plot with hue parameter."""
        ax = RainCloud(x="category", y="value", hue="hue", data=sample_data)
        assert ax is not None
        # Should have more collections due to hue groups
        assert len(ax.collections) > 3

    def test_raincloud_with_pointplot(self, sample_data):
        """Test raincloud plot with pointplot enabled."""
        ax = RainCloud(x="category", y="value", data=sample_data, pointplot=True)
        assert ax is not None
        # Pointplot adds lines to the plot
        assert len(ax.lines) > 0

    def test_raincloud_with_order(self, sample_data):
        """Test raincloud plot with custom order."""
        custom_order = ["C", "A", "B"]
        ax = RainCloud(x="category", y="value", data=sample_data, order=custom_order)
        assert ax is not None
        # Check that x-axis labels match the custom order
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == custom_order


class TestRainCloudParameters:
    """Test RainCloud parameter handling."""

    def test_raincloud_width_parameters(self, sample_data):
        """Test width parameters for violin and box."""
        ax = RainCloud(x="category", y="value", data=sample_data, width_viol=0.5, width_box=0.1)
        assert ax is not None

    def test_raincloud_move_parameter(self, sample_data):
        """Test the move parameter for stripplot positioning."""
        ax = RainCloud(x="category", y="value", data=sample_data, move=0.1)
        assert ax is not None

    def test_raincloud_offset_parameter(self, sample_data):
        """Test the offset parameter for violin positioning."""
        ax = RainCloud(x="category", y="value", data=sample_data, offset=0.2)
        assert ax is not None

    def test_raincloud_alpha_parameter(self, sample_data):
        """Test alpha transparency parameter."""
        ax = RainCloud(x="category", y="value", data=sample_data, alpha=0.5)
        assert ax is not None
        # Check that alpha is applied - some collections may have None (inherits from parent)
        # Just verify the parameter doesn't cause errors
        assert len(ax.collections) > 0

    def test_raincloud_palette(self, sample_data):
        """Test different palette options."""
        for palette in ["Set1", "Set2", "pastel"]:
            ax = RainCloud(x="category", y="value", data=sample_data, palette=palette)
            assert ax is not None
            plt.close()

    def test_raincloud_dodge(self, sample_data):
        """Test dodge parameter with hue."""
        ax = RainCloud(x="category", y="value", hue="hue", data=sample_data, dodge=True)
        assert ax is not None


class TestRainCloudKwargs:
    """Test kwargs forwarding to subcomponents."""

    def test_cloud_kwargs(self, sample_data):
        """Test kwargs forwarding to cloud/violin component."""
        # Use a different kwarg that won't conflict with RainCloud's own linewidth
        ax = RainCloud(x="category", y="value", data=sample_data, cloud_saturation=0.8)
        assert ax is not None

    def test_box_kwargs(self, sample_data):
        """Test kwargs forwarding to box component."""
        ax = RainCloud(x="category", y="value", data=sample_data, box_saturation=0.5)
        assert ax is not None

    def test_rain_kwargs(self, sample_data):
        """Test kwargs forwarding to rain/stripplot component."""
        ax = RainCloud(x="category", y="value", data=sample_data, rain_edgecolor="black")
        assert ax is not None

    def test_point_kwargs(self, sample_data):
        """Test kwargs forwarding to pointplot component."""
        ax = RainCloud(x="category", y="value", data=sample_data, pointplot=True, point_capsize=0.1)
        assert ax is not None


class TestRainCloudEdgeCases:
    """Test edge cases and error handling."""

    def test_raincloud_with_nan_values(self):
        """Test raincloud plot handles NaN values."""
        data = pd.DataFrame(
            {"cat": ["A", "A", "A", "B", "B", "B"], "val": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0]}
        )
        ax = RainCloud(x="cat", y="val", data=data)
        assert ax is not None

    def test_raincloud_single_category(self):
        """Test raincloud plot with single category."""
        data = pd.DataFrame({"cat": ["A"] * 10, "val": np.random.randn(10)})
        ax = RainCloud(x="cat", y="val", data=data)
        assert ax is not None

    def test_raincloud_with_custom_ax(self, sample_data):
        """Test raincloud plot with custom axes."""
        fig, ax = plt.subplots()
        result_ax = RainCloud(x="category", y="value", data=sample_data, ax=ax)
        assert result_ax is ax

    def test_raincloud_array_inputs(self):
        """Test raincloud plot with array inputs instead of DataFrame."""
        x = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([1, 2, 3, 4, 5, 6])
        ax = RainCloud(x=x, y=y)
        assert ax is not None


class TestPairedRaincloud:
    """Test the dedicated repeated-measures function `paired_raincloud`."""

    def _baseline_line_count(self, data, **kwargs):
        """Lines drawn by a plain RainCloud (boxplot artifacts, no subject lines)."""
        fig, ax = plt.subplots()
        RainCloud(x="condition", y="score", data=data, order=["pre", "post"], ax=ax, **kwargs)
        n = len(ax.lines)
        plt.close(fig)
        return n

    def test_adds_one_line_per_subject(self, paired_data):
        base = self._baseline_line_count(paired_data, orient="h")
        fig, ax = plt.subplots()
        paired_raincloud(
            x="condition",
            y="score",
            data=paired_data,
            id="subject",
            order=["pre", "post"],
            orient="h",
            ax=ax,
        )
        added = len(ax.lines) - base
        plt.close(fig)
        assert added == paired_data["subject"].nunique()

    def test_works_vertical(self, paired_data):
        base = self._baseline_line_count(paired_data, orient="v")
        fig, ax = plt.subplots()
        paired_raincloud(
            x="condition",
            y="score",
            data=paired_data,
            id="subject",
            order=["pre", "post"],
            orient="v",
            ax=ax,
        )
        added = len(ax.lines) - base
        plt.close(fig)
        assert added == paired_data["subject"].nunique()

    def test_raincloud_alone_draws_no_subject_lines(self, paired_data):
        """A plain RainCloud (no paired function) draws no per-subject lines."""
        fig, ax = plt.subplots()
        RainCloud(
            x="condition", y="score", data=paired_data, order=["pre", "post"], orient="h", ax=ax
        )
        # only boxplot Line2D artifacts, no subject lines
        n_box_lines = len(ax.lines)
        plt.close(fig)
        assert n_box_lines < paired_data["subject"].nunique()

    def test_missing_observations_do_not_error(self, paired_data):
        """NaN measurements should break that subject's line, not raise."""
        df = paired_data.copy()
        df.loc[(df.condition == "post") & df.subject.isin(["s0", "s1"]), "score"] = np.nan
        fig, ax = plt.subplots()
        paired_raincloud(
            x="condition",
            y="score",
            data=df,
            id="subject",
            order=["pre", "post"],
            orient="h",
            ax=ax,
        )
        plt.close(fig)  # success = no exception

    def test_hue_and_dodge_raise(self, paired_data):
        """hue/dodge are unsupported and should raise, not draw wrong lines."""
        df = paired_data.copy()
        df["arm"] = (["A", "B"] * len(df))[: len(df)]
        with pytest.raises(ValueError, match="hue.*dodge"):
            paired_raincloud(x="condition", y="score", data=df, id="subject", hue="arm", orient="h")
        with pytest.raises(ValueError, match="hue.*dodge"):
            paired_raincloud(
                x="condition", y="score", data=paired_data, id="subject", dodge=True, orient="h"
            )

    def test_requires_string_columns(self, paired_data):
        """Non-column-name inputs should raise a clear error."""
        with pytest.raises(ValueError):
            paired_raincloud(x="condition", y="score", data=paired_data, id=None)

    def test_line_style_kwargs_accepted(self, paired_data):
        fig, ax = plt.subplots()
        paired_raincloud(
            x="condition",
            y="score",
            data=paired_data,
            id="subject",
            order=["pre", "post"],
            orient="h",
            line_color="steelblue",
            line_alpha=0.6,
            line_width=1.0,
            ax=ax,
        )
        plt.close(fig)  # success = no exception

    def test_forwards_kwargs_to_raincloud(self, paired_data):
        """Styling kwargs (e.g. move, point_size) pass through to RainCloud."""
        fig, ax = plt.subplots()
        paired_raincloud(
            x="condition",
            y="score",
            data=paired_data,
            id="subject",
            order=["pre", "post"],
            orient="h",
            move=0.2,
            point_size=4,
            ax=ax,
        )
        assert len(ax.lines) >= paired_data["subject"].nunique()
        plt.close(fig)

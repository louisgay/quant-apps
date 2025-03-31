"""Unit tests for the structured product pricing engine."""

import numpy as np
import pytest

from engine.market_data import MarketData
from engine.monte_carlo import MonteCarloEngine
from engine.products import AutocallablePhoenix, AutocallableAthena
from engine.greeks import GreeksCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_asset_market():
    """Simple single-asset market for deterministic tests."""
    return MarketData(
        tickers=["TEST"],
        spots=np.array([100.0]),
        volatilities=np.array([0.20]),
        correlation_matrix=np.array([[1.0]]),
        risk_free_rate=0.05,
        dividend_yields=np.array([0.02]),
    )


@pytest.fixture
def basket_market():
    """3-asset correlated market."""
    corr = np.array([
        [1.00, 0.60, 0.40],
        [0.60, 1.00, 0.50],
        [0.40, 0.50, 1.00],
    ])
    return MarketData(
        tickers=["AAPL", "MSFT", "GOOGL"],
        spots=np.array([180.0, 370.0, 140.0]),
        volatilities=np.array([0.28, 0.25, 0.30]),
        correlation_matrix=corr,
        risk_free_rate=0.04,
        dividend_yields=np.array([0.005, 0.008, 0.0]),
    )


@pytest.fixture
def simple_product():
    return AutocallablePhoenix(
        underlyings=["TEST"],
        notional=1_000_000,
        maturity_years=2.0,
        observation_frequency="quarterly",
        autocall_barrier=1.00,
        coupon_barrier=0.70,
        coupon_rate=0.08,
        put_barrier=0.60,
        memory_feature=True,
    )


@pytest.fixture
def basket_product():
    return AutocallablePhoenix(
        underlyings=["AAPL", "MSFT", "GOOGL"],
        notional=1_000_000,
        maturity_years=3.0,
        observation_frequency="quarterly",
        autocall_barrier=1.00,
        coupon_barrier=0.65,
        coupon_rate=0.07,
        put_barrier=0.55,
        memory_feature=True,
    )


@pytest.fixture
def simple_athena():
    return AutocallableAthena(
        underlyings=["TEST"],
        notional=1_000_000,
        maturity_years=2.0,
        observation_frequency="quarterly",
        autocall_barrier=1.00,
        coupon_rate=0.08,
        put_barrier=0.60,
    )


@pytest.fixture
def basket_athena():
    return AutocallableAthena(
        underlyings=["AAPL", "MSFT", "GOOGL"],
        notional=1_000_000,
        maturity_years=3.0,
        observation_frequency="quarterly",
        autocall_barrier=1.00,
        coupon_rate=0.07,
        put_barrier=0.55,
    )


@pytest.fixture
def engine():
    return MonteCarloEngine(n_paths=50_000, seed=42)


# ---------------------------------------------------------------------------
# MarketData tests
# ---------------------------------------------------------------------------

class TestMarketData:
    def test_validation_shape(self):
        with pytest.raises(AssertionError):
            MarketData(
                tickers=["A", "B"],
                spots=np.array([100.0]),  # wrong shape
                volatilities=np.array([0.2, 0.3]),
                correlation_matrix=np.eye(2),
            )

    def test_validation_symmetry(self):
        with pytest.raises(AssertionError):
            MarketData(
                tickers=["A", "B"],
                spots=np.array([100.0, 100.0]),
                volatilities=np.array([0.2, 0.3]),
                correlation_matrix=np.array([[1.0, 0.5], [0.3, 1.0]]),
            )

    def test_bump_spot(self, single_asset_market):
        bumped = single_asset_market.bump_spot(0, 1.01)
        assert bumped.spots[0] == pytest.approx(101.0, abs=1e-10)
        assert single_asset_market.spots[0] == 100.0  # original unchanged

    def test_bump_vol(self, single_asset_market):
        bumped = single_asset_market.bump_vol(0, 0.05)
        assert bumped.volatilities[0] == pytest.approx(0.25, abs=1e-10)

    def test_bump_correlation_stays_valid(self, basket_market):
        bumped = basket_market.bump_correlation(-0.20)
        eigvals = np.linalg.eigvalsh(bumped.correlation_matrix)
        assert np.all(eigvals > -1e-6), "correlation matrix must remain PSD"
        np.testing.assert_allclose(np.diag(bumped.correlation_matrix), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# MonteCarloEngine tests
# ---------------------------------------------------------------------------

class TestMonteCarloEngine:
    def test_output_shape(self, single_asset_market, engine):
        dates = np.array([0.25, 0.50, 0.75, 1.0])
        paths = engine.simulate(single_asset_market, dates)
        assert paths.shape == (engine.n_paths, 1, 4)

    def test_positive_prices(self, basket_market, engine):
        dates = np.array([0.25, 0.50, 0.75, 1.0])
        paths = engine.simulate(basket_market, dates)
        assert np.all(paths > 0), "GBM paths must be strictly positive"

    def test_reproducibility(self, single_asset_market):
        dates = np.array([0.5, 1.0])
        eng1 = MonteCarloEngine(n_paths=1000, seed=123)
        eng2 = MonteCarloEngine(n_paths=1000, seed=123)
        p1 = eng1.simulate(single_asset_market, dates)
        p2 = eng2.simulate(single_asset_market, dates)
        np.testing.assert_array_equal(p1, p2)

    def test_mean_convergence(self, single_asset_market):
        """E[S_T] under risk-neutral measure = S_0 * exp((r-q)*T)."""
        T = 1.0
        dates = np.array([T])
        eng = MonteCarloEngine(n_paths=500_000, seed=0)
        paths = eng.simulate(single_asset_market, dates)
        S_T_mean = paths[:, 0, 0].mean()
        r = single_asset_market.risk_free_rate
        q = single_asset_market.dividend_yields[0]
        expected = single_asset_market.spots[0] * np.exp((r - q) * T)
        assert S_T_mean == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Product tests — Phoenix
# ---------------------------------------------------------------------------

class TestAutocallablePhoenix:
    def test_observation_dates(self, simple_product):
        dates = simple_product.observation_dates
        assert len(dates) == 8  # 2 years x 4 quarters
        assert dates[0] == pytest.approx(0.25)
        assert dates[-1] == pytest.approx(2.0)

    def test_coupon_per_period(self, simple_product):
        assert simple_product.coupon_per_period == pytest.approx(0.02)

    def test_price_returns_required_keys(self, simple_product, single_asset_market, engine):
        paths = engine.simulate(single_asset_market, simple_product.observation_dates)
        result = simple_product.price(
            paths, single_asset_market.spots, single_asset_market.risk_free_rate
        )
        required = {"price", "std_error", "price_pct", "autocall_prob",
                     "avg_life_years", "coupon_barrier_breach_prob", "capital_loss_prob"}
        assert required.issubset(result.keys())

    def test_price_is_positive(self, simple_product, single_asset_market, engine):
        paths = engine.simulate(single_asset_market, simple_product.observation_dates)
        result = simple_product.price(
            paths, single_asset_market.spots, single_asset_market.risk_free_rate
        )
        assert result["price"] > 0

    def test_probabilities_in_range(self, basket_product, basket_market, engine):
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert 0 <= result["autocall_prob"] <= 1
        assert 0 <= result["capital_loss_prob"] <= 1

    def test_zero_vol_always_autocalls(self):
        """With sigma=0 and r>0, spot drifts up -> always autocalled at first date."""
        md = MarketData(
            tickers=["A"],
            spots=np.array([100.0]),
            volatilities=np.array([1e-10]),
            correlation_matrix=np.array([[1.0]]),
            risk_free_rate=0.05,
        )
        product = AutocallablePhoenix(
            underlyings=["A"],
            notional=100.0,
            maturity_years=1.0,
            observation_frequency="quarterly",
            autocall_barrier=1.0,
            coupon_barrier=0.70,
            coupon_rate=0.08,
            put_barrier=0.60,
        )
        eng = MonteCarloEngine(n_paths=10_000, seed=0)
        paths = eng.simulate(md, product.observation_dates)
        result = product.price(paths, md.spots, md.risk_free_rate)
        assert result["autocall_prob"] == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Product tests — Athena
# ---------------------------------------------------------------------------

class TestAutocallableAthena:
    def test_observation_dates(self, simple_athena):
        dates = simple_athena.observation_dates
        assert len(dates) == 8  # 2 years x 4 quarters
        assert dates[0] == pytest.approx(0.25)
        assert dates[-1] == pytest.approx(2.0)

    def test_price_returns_required_keys(self, simple_athena, single_asset_market, engine):
        paths = engine.simulate(single_asset_market, simple_athena.observation_dates)
        result = simple_athena.price(
            paths, single_asset_market.spots, single_asset_market.risk_free_rate
        )
        required = {"price", "std_error", "price_pct", "autocall_prob",
                     "avg_life_years", "capital_loss_prob"}
        assert required.issubset(result.keys())

    def test_price_is_positive(self, simple_athena, single_asset_market, engine):
        paths = engine.simulate(single_asset_market, simple_athena.observation_dates)
        result = simple_athena.price(
            paths, single_asset_market.spots, single_asset_market.risk_free_rate
        )
        assert result["price"] > 0

    def test_probabilities_in_range(self, basket_athena, basket_market, engine):
        paths = engine.simulate(basket_market, basket_athena.observation_dates)
        result = basket_athena.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert 0 <= result["autocall_prob"] <= 1
        assert 0 <= result["capital_loss_prob"] <= 1

    def test_zero_vol_always_autocalls(self):
        """With sigma=0 and r>0, spot drifts up -> always autocalled at first date."""
        md = MarketData(
            tickers=["A"],
            spots=np.array([100.0]),
            volatilities=np.array([1e-10]),
            correlation_matrix=np.array([[1.0]]),
            risk_free_rate=0.05,
        )
        product = AutocallableAthena(
            underlyings=["A"],
            notional=100.0,
            maturity_years=1.0,
            observation_frequency="quarterly",
            autocall_barrier=1.0,
            coupon_rate=0.08,
            put_barrier=0.60,
        )
        eng = MonteCarloEngine(n_paths=10_000, seed=0)
        paths = eng.simulate(md, product.observation_dates)
        result = product.price(paths, md.spots, md.risk_free_rate)
        assert result["autocall_prob"] == pytest.approx(1.0, abs=0.001)

    def test_coupon_proportional_to_time(self):
        """With sigma~0, autocall at t=0.25 -> coupon = coupon_rate * 0.25 * N * df."""
        md = MarketData(
            tickers=["A"],
            spots=np.array([100.0]),
            volatilities=np.array([1e-10]),
            correlation_matrix=np.array([[1.0]]),
            risk_free_rate=0.05,
        )
        product = AutocallableAthena(
            underlyings=["A"],
            notional=1_000_000,
            maturity_years=1.0,
            observation_frequency="quarterly",
            autocall_barrier=1.0,
            coupon_rate=0.08,
            put_barrier=0.60,
        )
        eng = MonteCarloEngine(n_paths=50_000, seed=0)
        paths = eng.simulate(md, product.observation_dates)
        result = product.price(paths, md.spots, md.risk_free_rate)
        c = result["components"]

        # All paths autocall at t=0.25 -> coupon = 0.08 * 0.25 * N * e^{-0.05*0.25}
        expected_coupon_pct = 0.08 * 0.25 * np.exp(-0.05 * 0.25) * 100
        assert c["coupons"]["pct"] == pytest.approx(expected_coupon_pct, abs=0.1)

    def test_no_coupon_at_maturity(self):
        """If product reaches maturity without autocall, no coupon is paid."""
        # Use high autocall barrier so it never triggers
        md = MarketData(
            tickers=["A"],
            spots=np.array([100.0]),
            volatilities=np.array([0.01]),
            correlation_matrix=np.array([[1.0]]),
            risk_free_rate=0.00,  # zero drift to keep spot ~100
        )
        product = AutocallableAthena(
            underlyings=["A"],
            notional=1_000_000,
            maturity_years=1.0,
            observation_frequency="quarterly",
            autocall_barrier=2.0,   # 200% -> never autocalls
            coupon_rate=0.08,
            put_barrier=0.60,
        )
        eng = MonteCarloEngine(n_paths=50_000, seed=0)
        paths = eng.simulate(md, product.observation_dates)
        result = product.price(paths, md.spots, md.risk_free_rate)
        c = result["components"]

        # No autocall -> no coupon
        assert result["autocall_prob"] == pytest.approx(0.0, abs=0.01)
        assert c["coupons"]["pct"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Decomposition tests — Phoenix
# ---------------------------------------------------------------------------

class TestDecomposition:
    """Verify that the payoff decomposition is consistent."""

    def test_components_present(self, basket_product, basket_market, engine):
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert "components" in result
        for key in ("zcb", "coupons", "short_put", "autocall_option"):
            assert key in result["components"], f"Missing component: {key}"
            comp = result["components"][key]
            for field in ("mean", "std_error", "pct", "std_error_pct"):
                assert field in comp, f"Missing field {field} in {key}"

    def test_sum_equals_full_price(self, basket_product, basket_market, engine):
        """ZCB + Coupons + Put == Full Price (to machine precision)."""
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        reconstructed = c["zcb"]["mean"] + c["coupons"]["mean"] + c["short_put"]["mean"]
        assert reconstructed == pytest.approx(result["price"], abs=1e-6), (
            f"Decomposition mismatch: {reconstructed} != {result['price']}"
        )

    def test_sum_equals_full_price_pct(self, basket_product, basket_market, engine):
        """Same check in % of notional."""
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        reconstructed_pct = c["zcb"]["pct"] + c["coupons"]["pct"] + c["short_put"]["pct"]
        assert reconstructed_pct == pytest.approx(result["price_pct"], abs=1e-4)

    def test_autocall_option_equals_zcb_minus_bond_floor(
        self, basket_product, basket_market, engine
    ):
        """Autocall option = ZCB - e^{-rT} * 100%."""
        r = basket_market.risk_free_rate
        T = basket_product.maturity_years
        bond_floor_pct = np.exp(-r * T) * 100  # deterministic ZCB to maturity

        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        expected_ac = c["zcb"]["pct"] - bond_floor_pct
        assert c["autocall_option"]["pct"] == pytest.approx(expected_ac, abs=1e-4)

    def test_component_signs(self, basket_product, basket_market, engine):
        """ZCB > 0, Coupons >= 0, Put <= 0, Autocall Option >= 0."""
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        assert c["zcb"]["mean"] > 0, "ZCB must be positive"
        assert c["coupons"]["mean"] >= 0, "Coupons must be non-negative"
        assert c["short_put"]["mean"] <= 0, "Short put must be non-positive"
        assert c["autocall_option"]["mean"] >= 0, "Autocall option must be non-negative"

    def test_zcb_bounded_by_notional(self, simple_product, single_asset_market, engine):
        """ZCB cannot exceed the (undiscounted) notional."""
        paths = engine.simulate(single_asset_market, simple_product.observation_dates)
        result = simple_product.price(
            paths, single_asset_market.spots, single_asset_market.risk_free_rate
        )
        assert result["components"]["zcb"]["mean"] <= simple_product.notional

    def test_decomposition_check_near_zero(self, basket_product, basket_market, engine):
        paths = engine.simulate(basket_market, basket_product.observation_dates)
        result = basket_product.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert result["decomposition_check"] < 1e-6

    def test_zero_vol_decomposition(self):
        """With sigma~0, r>0: autocall at Q1, no put loss, known analytics."""
        md = MarketData(
            tickers=["A"],
            spots=np.array([100.0]),
            volatilities=np.array([1e-10]),
            correlation_matrix=np.array([[1.0]]),
            risk_free_rate=0.05,
        )
        product = AutocallablePhoenix(
            underlyings=["A"],
            notional=1_000_000,
            maturity_years=1.0,
            observation_frequency="quarterly",
            autocall_barrier=1.0,
            coupon_barrier=0.70,
            coupon_rate=0.08,
            put_barrier=0.60,
        )
        eng = MonteCarloEngine(n_paths=50_000, seed=0)
        paths = eng.simulate(md, product.observation_dates)
        result = product.price(paths, md.spots, md.risk_free_rate)
        c = result["components"]

        # All paths autocall at t=0.25 -> ZCB = e^{-0.05*0.25} * N
        expected_zcb_pct = np.exp(-0.05 * 0.25) * 100
        assert c["zcb"]["pct"] == pytest.approx(expected_zcb_pct, abs=0.1)

        # Coupon at t=0.25: 0.08/4 = 2% * e^{-0.05*0.25}
        expected_coupon_pct = 2.0 * np.exp(-0.05 * 0.25)
        assert c["coupons"]["pct"] == pytest.approx(expected_coupon_pct, abs=0.1)

        # No put losses
        assert c["short_put"]["pct"] == pytest.approx(0.0, abs=0.01)

        # Autocall option > 0 (early redemption vs maturity)
        assert c["autocall_option"]["pct"] > 0


# ---------------------------------------------------------------------------
# Decomposition tests — Athena
# ---------------------------------------------------------------------------

class TestAthenaDecomposition:
    """Verify that the Athena payoff decomposition is consistent."""

    def test_components_present(self, basket_athena, basket_market, engine):
        paths = engine.simulate(basket_market, basket_athena.observation_dates)
        result = basket_athena.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert "components" in result
        for key in ("zcb", "coupons", "short_put", "autocall_option"):
            assert key in result["components"], f"Missing component: {key}"
            comp = result["components"][key]
            for field in ("mean", "std_error", "pct", "std_error_pct"):
                assert field in comp, f"Missing field {field} in {key}"

    def test_sum_equals_full_price(self, basket_athena, basket_market, engine):
        """ZCB + Coupons + Put == Full Price (to machine precision)."""
        paths = engine.simulate(basket_market, basket_athena.observation_dates)
        result = basket_athena.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        reconstructed = c["zcb"]["mean"] + c["coupons"]["mean"] + c["short_put"]["mean"]
        assert reconstructed == pytest.approx(result["price"], abs=1e-6), (
            f"Decomposition mismatch: {reconstructed} != {result['price']}"
        )

    def test_component_signs(self, basket_athena, basket_market, engine):
        """ZCB > 0, Coupons >= 0, Put <= 0, Autocall Option >= 0."""
        paths = engine.simulate(basket_market, basket_athena.observation_dates)
        result = basket_athena.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        c = result["components"]
        assert c["zcb"]["mean"] > 0, "ZCB must be positive"
        assert c["coupons"]["mean"] >= 0, "Coupons must be non-negative"
        assert c["short_put"]["mean"] <= 0, "Short put must be non-positive"
        assert c["autocall_option"]["mean"] >= 0, "Autocall option must be non-negative"

    def test_decomposition_check_near_zero(self, basket_athena, basket_market, engine):
        paths = engine.simulate(basket_market, basket_athena.observation_dates)
        result = basket_athena.price(
            paths, basket_market.spots, basket_market.risk_free_rate
        )
        assert result["decomposition_check"] < 1e-6


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_delta_sign(self, basket_product, basket_market, engine):
        """For a worst-of autocallable, delta should generally be positive
        (higher spot -> more autocall probability -> higher value)."""
        calc = GreeksCalculator(engine, basket_product, basket_market)
        d = calc.delta(bump_pct=0.02)
        assert all(v >= 0 for v in d.values()), f"Expected non-negative deltas, got {d}"

    def test_vega_sign(self, basket_product, basket_market, engine):
        """Higher vol on worst-of product -> lower price (more downside risk)."""
        calc = GreeksCalculator(engine, basket_product, basket_market)
        v = calc.vega(bump_vol=0.02)
        # At least one underlying should show negative vega for worst-of
        assert any(val < 0 for val in v.values())

    def test_correlation_sensitivity_structure(self, basket_product, basket_market, engine):
        calc = GreeksCalculator(engine, basket_product, basket_market)
        cs = calc.correlation_sensitivity(bumps=[-0.10, 0.0, 0.10])
        assert len(cs) == 3
        assert all(isinstance(v, float) for v in cs.values())

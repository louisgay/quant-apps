"""Public API."""

from engine.data_loader import (
    TICKER_UNIVERSE,
    compute_log_returns,
    fetch_prices,
)
from engine.garch import (
    FHSResult,
    GARCHResult,
    fhs_convergence,
    fit_gjr_garch,
    news_impact_curve,
    simulate_fhs,
)
from engine.var_models import (
    garch_normal_var,
    garch_student_t_var,
    gjr_garch_student_t_var,
    historical_simulation_var,
    monte_carlo_fhs_var,
    parametric_normal_var,
    parametric_student_t_var,
)
from engine.backtest import (
    christoffersen_test,
    combined_coverage_test,
    kupiec_test,
    run_full_backtest,
    var_exceedance_test,
)
from engine.plots import (
    conditional_volatility,
    exceedance_scatter,
    fhs_convergence_plot,
    fhs_fan_chart,
    fhs_return_distribution,
    news_impact_curve_plot,
    qq_plot,
    returns_distribution,
    rolling_var_comparison,
    var_comparison_bar,
)

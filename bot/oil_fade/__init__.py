"""Oil-fade package: strategy core + simulator + paper runner."""
from .strategy import Bar, OilFadeConfig, OilFadeStrategy, TradeSignal

__all__ = ["Bar", "OilFadeConfig", "OilFadeStrategy", "TradeSignal"]

"""Thread 3: Triangular arbitrage feasibility (single venue).
Uses REAL order books (top of book bid/ask, no mid-price fills) + retail taker fees.
Triangles: USD -> XBT -> ETH -> USD on Kraken; USD -> BTC -> ETH -> USD on Coinbase.
A cycle is profitable iff product of legs * (1-fee)^3 > 1.
"""
from __future__ import annotations

import requests

FEE_KRAKEN = 0.0040  # taker
FEE_CB = 0.0060      # taker


def kraken_book(pair: str):
    r = requests.get("https://api.kraken.com/0/public/Depth",
                     params={"pair": pair, "count": 5}, timeout=15)
    r.raise_for_status()
    d = r.json()["result"]
    key = [k for k in d if k != "last"][0]
    asks = d[key]["asks"]
    bids = d[key]["bids"]
    return float(asks[0][0]), float(bids[0][0])  # best ask, best bid


def cb_book(product: str):
    r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/book",
                     params={"level": 1}, timeout=15)
    r.raise_for_status()
    d = r.json()
    return float(d["asks"][0][0]), float(d["bids"][0][0])


def tri(venue: str, legs: dict, fee: float):
    """legs: dict of pair -> (ask, bid). path defined by caller."""
    return None


def main():
    print("=== Kraken: USD -> XBT -> ETH -> USD ===")
    ask_xbtusd, bid_xbtusd = kraken_book("XBTUSD")
    ask_ethusd, bid_ethusd = kraken_book("ETHUSD")
    ask_ethxbt, bid_ethxbt = kraken_book("ETHXBT")
    # USD -> XBT at ask_xbtusd; XBT -> ETH at ask_ethxbt; ETH -> USD at bid_ethusd
    gross = (1 / ask_xbtusd) * (1 / ask_ethxbt) * bid_ethusd
    net = gross * (1 - FEE_KRAKEN) ** 3
    print(f"XBTUSD ask {ask_xbtusd:,.1f} | ETHXBT ask {ask_ethxbt:.6f} | ETHUSD bid {bid_ethusd:,.1f}")
    print(f"gross cycle ratio: {gross:.6f}  (needs > 1.0)")
    print(f"net after 3x {FEE_KRAKEN:.2%} taker fees: {net:.6f}  -> {'PROFIT' if net > 1 else 'LOSS'}")
    print(f"  fee drag alone: {(1-FEE_KRAKEN)**3:.4f}; break-even gross needed: {1/(1-FEE_KRAKEN)**3:.4f}")

    # reverse direction: USD -> ETH -> XBT -> USD
    gross_rev = (1 / ask_ethusd) * bid_ethxbt * (1 / ask_xbtusd) * ask_xbtusd  # careful
    # USD->ETH at ask_ethusd, ETH->XBT at bid_ethxbt, XBT->USD at bid_xbtusd
    gross_rev = (1 / ask_ethusd) * bid_ethxbt * bid_xbtusd
    net_rev = gross_rev * (1 - FEE_KRAKEN) ** 3
    print(f"\nreverse (USD->ETH->XBT->USD): gross {gross_rev:.6f}, net {net_rev:.6f} -> {'PROFIT' if net_rev > 1 else 'LOSS'}")

    print("\n=== Coinbase: USD -> BTC -> ETH -> USD ===")
    ask_btcusd, _ = cb_book("BTC-USD")
    ask_ethusd_c, bid_ethusd_c = cb_book("ETH-USD")
    ask_ethbtc, _ = cb_book("ETH-BTC")
    gross_c = (1 / ask_btcusd) * (1 / ask_ethbtc) * bid_ethusd_c
    net_c = gross_c * (1 - FEE_CB) ** 3
    print(f"BTC-USD ask {ask_btcusd:,.1f} | ETH-BTC ask {ask_ethbtc:.6f} | ETH-USD bid {bid_ethusd_c:,.1f}")
    print(f"gross {gross_c:.6f}, net after 3x {FEE_CB:.2%}: {net_c:.6f} -> {'PROFIT' if net_c > 1 else 'LOSS'}")

    print("\n=== Verdict inputs ===")
    print(f"Kraken gross mispricing (best case): {(max(gross, gross_rev)-1)*100:.4f}%")
    print(f"Coinbase gross mispricing: {(gross_c-1)*100:.4f}%")
    print("Anything under ~1.2% gross is dead on fees alone, before depth/queue risk.")


if __name__ == "__main__":
    main()

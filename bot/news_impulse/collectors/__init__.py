"""Source collectors for the news-impulse engine.

One module per source; each yields RawPost objects. All are free, read-only,
and rate-limit aware. Stocktwits' public endpoint died 2026-09-20, so its
collector is kept as a graceful no-op. Live social comes from Arctic Shift
(Reddit search), Reddit .rss feeds, 4chan /biz/, ApeWisdom attention ranks,
and X (needs X_BEARER_TOKEN, heavily rate-limited so it stays secondary).
Filings and wire headlines come from EDGAR 8-K Atom feeds and the
headlines.csv archive tail.
"""

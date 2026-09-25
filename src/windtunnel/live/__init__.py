"""Live trading with real money. This is the ONLY package allowed to touch API credentials.

Safeguards (see CLAUDE.md hard rules):

* the triple opt-in gate (`gate.resolve_mode`);
* keys read from environment variables only, and redacted from logs;
* a hard notional cap, limit orders only, spot only, long/flat only;
* deterministic client order ids (no double orders after a restart);
* startup reconciliation that halts, rather than "fixes", on any mismatch.
"""

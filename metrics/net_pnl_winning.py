# metrics/net_pnl_winning.py
from __future__ import annotations

"""
DEPRECATO.

La metrica "Net P&L (Winning Trades Only)" è stata sostituita da:
- "Gross Profit (Winning Trades Only)"
- "Gross Loss (Losing Trades Only)"

Le metriche sono implementate in: metrics/gross_pnl_trades.py
Questo modulo rimane solo per retro-compatibilità del loader (_ensure_loaded),
ma non registra più indicatori.
"""

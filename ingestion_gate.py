"""Market-data ingestion is not deployed.

This module is a closed gate. It does not store API keys, does not choose
an upstream URL, and does not open a socket. A future ingestion service
would have to be reviewed separately and would stay outside the forecast
worker.
"""

from kronos_safety import SafetyError

INGESTION_DEPLOYED = False


def refuse_market_ingestion(source: str = "market-data") -> None:
    if INGESTION_DEPLOYED:
        raise SafetyError("ingestion_misconfigured", "ingestion must stay disabled in this tree")
    raise SafetyError(
        "ingestion_not_deployed",
        "Market-data ingestion is not deployed. "
        "The forecast worker reads sanitized local price snapshots only. "
        f"Refused source label: {source[:40]}",
    )

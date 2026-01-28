#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sec_loader import SECLoader
from src.utils.logging import setup_logging, get_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    loader = SECLoader()
    for ticker in args.tickers.split(","):
        ticker = ticker.strip().upper()
        logger.info(f"Ingesting {ticker}")
        result = loader.ingest(ticker, args.limit)
        logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()

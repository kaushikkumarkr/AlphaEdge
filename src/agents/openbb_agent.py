from typing import List, Dict, Tuple
from openbb import obb
from src.agents.base_agent import BaseAgent
from src.guardrails.schemas import RetrievedContext, Citation
from src.config.constants import AgentName
from datetime import datetime, timedelta
import re


class OpenBBAgent(BaseAgent):
    """
    Financial data agent using OpenBB SDK.
    
    Supports:
    - Stock quotes and historical prices
    - Fundamental data (income, balance sheet, cash flow)
    - Key metrics and ratios
    - Analyst estimates and price targets
    - Insider/institutional ownership
    - Earnings calendar
    - Company news
    - Options data
    """
    
    def __init__(self, **kwargs):
        super().__init__(name=AgentName.OPENBB, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are a financial data analyst with access to comprehensive market data.

Rules:
1. Report exact numbers from the data provided
2. Always cite the data source (quote, fundamentals, estimates, etc.)
3. Explain what metrics mean in context
4. Compare to industry benchmarks when available
5. Note the data freshness and any limitations
6. For valuation questions, use multiple metrics (P/E, P/S, EV/EBITDA)
7. For growth analysis, compare YoY and sequential changes"""
    
    def _detect_query_intent(self, query: str) -> List[str]:
        """Detect what data the query needs."""
        query_lower = query.lower()
        intents = []
        
        # Price-related
        if any(w in query_lower for w in ['price', 'stock', 'trading', 'quote', 'current']):
            intents.append('quote')
        
        # Historical prices
        if any(w in query_lower for w in ['historical', 'history', 'chart', 'trend', 'performance', '52 week', 'ytd']):
            intents.append('historical')
        
        # Valuation
        if any(w in query_lower for w in ['p/e', 'pe ratio', 'valuation', 'multiple', 'ev/ebitda', 'p/s', 'price to']):
            intents.append('metrics')
        
        # Fundamentals
        if any(w in query_lower for w in ['revenue', 'income', 'profit', 'margin', 'earnings', 'ebitda', 'net income']):
            intents.append('income')
        if any(w in query_lower for w in ['balance sheet', 'debt', 'assets', 'liabilities', 'cash', 'equity']):
            intents.append('balance')
        if any(w in query_lower for w in ['cash flow', 'free cash flow', 'fcf', 'capex', 'operating cash']):
            intents.append('cashflow')
        
        # Analyst/Estimates
        if any(w in query_lower for w in ['analyst', 'estimate', 'forecast', 'target', 'rating', 'consensus', 'eps estimate']):
            intents.append('estimates')
        
        # Ownership
        if any(w in query_lower for w in ['insider', 'institutional', 'ownership', 'holders', 'bought', 'sold']):
            intents.append('ownership')
        
        # Dividends
        if any(w in query_lower for w in ['dividend', 'yield', 'payout']):
            intents.append('dividends')
        
        # News
        if any(w in query_lower for w in ['news', 'headlines', 'announcement', 'press release']):
            intents.append('news')
        
        # Options
        if any(w in query_lower for w in ['option', 'call', 'put', 'strike', 'expiry', 'implied volatility']):
            intents.append('options')
        
        # Default to comprehensive view
        if not intents:
            intents = ['quote', 'metrics', 'income']
        
        return intents
    
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        ticker = filters.get("ticker", "").upper()
        if not ticker:
            # Try to extract ticker from query
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
            if ticker_match:
                ticker = ticker_match.group(1)
            else:
                return []
        
        intents = self._detect_query_intent(query)
        contexts = []
        
        # === STOCK QUOTE ===
        if 'quote' in intents:
            try:
                quote = obb.equity.price.quote(ticker)
                if quote.results:
                    q = quote.results[0]
                    text_parts = [f"=== {ticker} Stock Quote ==="]
                    
                    price = getattr(q, 'last_price', None) or getattr(q, 'price', None) or getattr(q, 'close', None)
                    if price: text_parts.append(f"Current Price: ${price:,.2f}" if isinstance(price, (int, float)) else f"Current Price: ${price}")
                    
                    change = getattr(q, 'change', None)
                    change_pct = getattr(q, 'change_percent', None) or getattr(q, 'percent_change', None)
                    if change is not None and change_pct is not None:
                        text_parts.append(f"Change: ${change:+,.2f} ({change_pct:+.2f}%)")
                    
                    volume = getattr(q, 'volume', None)
                    if volume: text_parts.append(f"Volume: {volume:,.0f}")
                    
                    high = getattr(q, 'high', None)
                    low = getattr(q, 'low', None)
                    if high and low: text_parts.append(f"Day Range: ${low:,.2f} - ${high:,.2f}")
                    
                    high_52 = getattr(q, 'year_high', None) or getattr(q, 'fifty_two_week_high', None)
                    low_52 = getattr(q, 'year_low', None) or getattr(q, 'fifty_two_week_low', None)
                    if high_52 and low_52: text_parts.append(f"52-Week Range: ${low_52:,.2f} - ${high_52:,.2f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-quote-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.95,
                        metadata={"type": "quote", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === HISTORICAL PRICES ===
        if 'historical' in intents:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                hist = obb.equity.price.historical(ticker, start_date=start_date.strftime("%Y-%m-%d"))
                if hist.results and len(hist.results) > 0:
                    df = hist.to_dataframe()
                    if not df.empty:
                        latest = df.iloc[-1]
                        earliest = df.iloc[0]
                        ytd_return = ((latest['close'] - earliest['close']) / earliest['close']) * 100
                        
                        # Calculate high/low
                        period_high = df['high'].max()
                        period_low = df['low'].min()
                        avg_volume = df['volume'].mean()
                        
                        text = f"""=== {ticker} Historical Performance (1 Year) ===
Current: ${latest['close']:,.2f}
1-Year Return: {ytd_return:+.2f}%
Period High: ${period_high:,.2f}
Period Low: ${period_low:,.2f}
Avg Daily Volume: {avg_volume:,.0f}"""
                        
                        contexts.append(RetrievedContext(
                            source_id=f"openbb-historical-{ticker}",
                            text=text,
                            relevance_score=0.85,
                            metadata={"type": "historical", "ticker": ticker}
                        ))
            except Exception as e:
                pass
        
        # === KEY METRICS ===
        if 'metrics' in intents:
            try:
                metrics = obb.equity.fundamental.metrics(ticker)
                if metrics.results:
                    m = metrics.results[0]
                    text_parts = [f"=== {ticker} Valuation Metrics ==="]
                    
                    for attr, label in [
                        ('pe_ratio', 'P/E Ratio'), ('pe_ratio_ttm', 'P/E (TTM)'),
                        ('price_to_sales_ratio', 'P/S Ratio'), ('price_to_book_ratio', 'P/B Ratio'),
                        ('ev_to_ebitda', 'EV/EBITDA'), ('peg_ratio', 'PEG Ratio'),
                        ('market_cap', 'Market Cap'), ('enterprise_value', 'Enterprise Value'),
                        ('eps', 'EPS'), ('eps_ttm', 'EPS (TTM)'),
                        ('dividend_yield', 'Dividend Yield'), ('beta', 'Beta'),
                        ('return_on_equity', 'ROE'), ('return_on_assets', 'ROA'),
                        ('profit_margin', 'Profit Margin'), ('operating_margin', 'Operating Margin'),
                        ('revenue_growth', 'Revenue Growth'), ('earnings_growth', 'Earnings Growth'),
                    ]:
                        val = getattr(m, attr, None)
                        if val is not None:
                            if 'cap' in attr or 'value' in attr:
                                text_parts.append(f"{label}: ${val:,.0f}")
                            elif 'yield' in attr or 'margin' in attr or 'growth' in attr or 'return' in attr:
                                text_parts.append(f"{label}: {val:.2%}" if val < 1 else f"{label}: {val:.2f}%")
                            else:
                                text_parts.append(f"{label}: {val:.2f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-metrics-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.9,
                        metadata={"type": "metrics", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === INCOME STATEMENT ===
        if 'income' in intents:
            try:
                income = obb.equity.fundamental.income(ticker, period="annual", limit=2)
                if income.results:
                    latest = income.results[0]
                    text_parts = [f"=== {ticker} Income Statement ==="]
                    
                    for attr, label in [
                        ('revenue', 'Revenue'), ('gross_profit', 'Gross Profit'),
                        ('operating_income', 'Operating Income'), ('net_income', 'Net Income'),
                        ('ebitda', 'EBITDA'), ('eps', 'EPS'), ('eps_diluted', 'Diluted EPS'),
                    ]:
                        val = getattr(latest, attr, None)
                        if val is not None:
                            if attr in ['eps', 'eps_diluted']:
                                text_parts.append(f"{label}: ${val:.2f}")
                            else:
                                text_parts.append(f"{label}: ${val:,.0f}")
                    
                    # Calculate margins
                    revenue = getattr(latest, 'revenue', None)
                    gross = getattr(latest, 'gross_profit', None)
                    operating = getattr(latest, 'operating_income', None)
                    net = getattr(latest, 'net_income', None)
                    
                    if revenue and gross:
                        text_parts.append(f"Gross Margin: {(gross/revenue)*100:.1f}%")
                    if revenue and operating:
                        text_parts.append(f"Operating Margin: {(operating/revenue)*100:.1f}%")
                    if revenue and net:
                        text_parts.append(f"Net Margin: {(net/revenue)*100:.1f}%")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-income-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.88,
                        metadata={"type": "income", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === BALANCE SHEET ===
        if 'balance' in intents:
            try:
                balance = obb.equity.fundamental.balance(ticker, period="annual", limit=1)
                if balance.results:
                    b = balance.results[0]
                    text_parts = [f"=== {ticker} Balance Sheet ==="]
                    
                    for attr, label in [
                        ('total_assets', 'Total Assets'), ('total_liabilities', 'Total Liabilities'),
                        ('total_equity', 'Total Equity'), ('cash_and_cash_equivalents', 'Cash'),
                        ('total_debt', 'Total Debt'), ('long_term_debt', 'Long-Term Debt'),
                        ('short_term_debt', 'Short-Term Debt'), ('inventory', 'Inventory'),
                        ('accounts_receivable', 'Accounts Receivable'),
                    ]:
                        val = getattr(b, attr, None)
                        if val is not None:
                            text_parts.append(f"{label}: ${val:,.0f}")
                    
                    # Debt ratios
                    debt = getattr(b, 'total_debt', None)
                    equity = getattr(b, 'total_equity', None)
                    if debt and equity and equity != 0:
                        text_parts.append(f"Debt/Equity Ratio: {debt/equity:.2f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-balance-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.85,
                        metadata={"type": "balance", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === CASH FLOW ===
        if 'cashflow' in intents:
            try:
                cashflow = obb.equity.fundamental.cash(ticker, period="annual", limit=1)
                if cashflow.results:
                    cf = cashflow.results[0]
                    text_parts = [f"=== {ticker} Cash Flow Statement ==="]
                    
                    for attr, label in [
                        ('operating_cash_flow', 'Operating Cash Flow'),
                        ('investing_cash_flow', 'Investing Cash Flow'),
                        ('financing_cash_flow', 'Financing Cash Flow'),
                        ('free_cash_flow', 'Free Cash Flow'),
                        ('capital_expenditure', 'CapEx'),
                        ('dividends_paid', 'Dividends Paid'),
                        ('share_repurchases', 'Share Buybacks'),
                    ]:
                        val = getattr(cf, attr, None)
                        if val is not None:
                            text_parts.append(f"{label}: ${val:,.0f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-cashflow-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.85,
                        metadata={"type": "cashflow", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === ANALYST ESTIMATES ===
        if 'estimates' in intents:
            try:
                # Price targets
                targets = obb.equity.estimates.price_target(ticker)
                if targets.results:
                    t = targets.results[0]
                    text_parts = [f"=== {ticker} Analyst Estimates ==="]
                    
                    for attr, label in [
                        ('target_high', 'High Target'), ('target_low', 'Low Target'),
                        ('target_mean', 'Mean Target'), ('target_median', 'Median Target'),
                        ('num_analysts', 'Number of Analysts'),
                    ]:
                        val = getattr(t, attr, None)
                        if val is not None:
                            if 'target' in attr:
                                text_parts.append(f"{label}: ${val:,.2f}")
                            else:
                                text_parts.append(f"{label}: {val}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-estimates-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.82,
                        metadata={"type": "estimates", "ticker": ticker}
                    ))
            except Exception as e:
                pass
            
            try:
                # EPS estimates
                consensus = obb.equity.estimates.consensus(ticker)
                if consensus.results:
                    c = consensus.results[0]
                    text_parts = [f"=== {ticker} Consensus Estimates ==="]
                    for attr, label in [
                        ('estimated_eps_avg', 'Est. EPS (Avg)'),
                        ('estimated_eps_high', 'Est. EPS (High)'),
                        ('estimated_eps_low', 'Est. EPS (Low)'),
                        ('estimated_revenue_avg', 'Est. Revenue (Avg)'),
                        ('number_of_analysts', 'Analysts'),
                    ]:
                        val = getattr(c, attr, None)
                        if val is not None:
                            if 'revenue' in attr:
                                text_parts.append(f"{label}: ${val:,.0f}")
                            elif 'eps' in attr:
                                text_parts.append(f"{label}: ${val:.2f}")
                            else:
                                text_parts.append(f"{label}: {val}")
                    
                    if len(text_parts) > 1:
                        contexts.append(RetrievedContext(
                            source_id=f"openbb-consensus-{ticker}",
                            text="\n".join(text_parts),
                            relevance_score=0.80,
                            metadata={"type": "consensus", "ticker": ticker}
                        ))
            except Exception as e:
                pass
        
        # === OWNERSHIP ===
        if 'ownership' in intents:
            try:
                # Insider trading
                insider = obb.equity.ownership.insider_trading(ticker, limit=10)
                if insider.results:
                    text_parts = [f"=== {ticker} Recent Insider Trading ==="]
                    for i, trade in enumerate(insider.results[:5]):
                        name = getattr(trade, 'owner_name', 'Unknown')
                        trans_type = getattr(trade, 'transaction_type', 'N/A')
                        shares = getattr(trade, 'shares', 0)
                        price = getattr(trade, 'price', 0)
                        text_parts.append(f"{name}: {trans_type} {shares:,.0f} shares @ ${price:.2f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-insider-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.75,
                        metadata={"type": "insider", "ticker": ticker}
                    ))
            except Exception as e:
                pass
            
            try:
                # Institutional ownership
                inst = obb.equity.ownership.institutional(ticker)
                if inst.results:
                    text_parts = [f"=== {ticker} Top Institutional Holders ==="]
                    for holder in inst.results[:5]:
                        name = getattr(holder, 'investor_name', 'Unknown')
                        shares = getattr(holder, 'shares', 0)
                        pct = getattr(holder, 'percent_of_total', 0)
                        text_parts.append(f"{name}: {shares:,.0f} shares ({pct:.2f}%)")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-institutional-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.75,
                        metadata={"type": "institutional", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === DIVIDENDS ===
        if 'dividends' in intents:
            try:
                divs = obb.equity.fundamental.dividends(ticker)
                if divs.results:
                    text_parts = [f"=== {ticker} Dividend History ==="]
                    recent = divs.results[:4]  # Last 4 dividends
                    for d in recent:
                        ex_date = getattr(d, 'ex_dividend_date', 'N/A')
                        amount = getattr(d, 'amount', 0)
                        text_parts.append(f"Ex-Date: {ex_date}, Amount: ${amount:.4f}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-dividends-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.78,
                        metadata={"type": "dividends", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === NEWS ===
        if 'news' in intents:
            try:
                news = obb.news.company(symbol=ticker, limit=5)
                if news.results:
                    text_parts = [f"=== {ticker} Recent News ==="]
                    for article in news.results[:5]:
                        title = getattr(article, 'title', 'N/A')
                        date = getattr(article, 'date', 'N/A')
                        text_parts.append(f"• [{date}] {title}")
                    
                    contexts.append(RetrievedContext(
                        source_id=f"openbb-news-{ticker}",
                        text="\n".join(text_parts),
                        relevance_score=0.70,
                        metadata={"type": "news", "ticker": ticker}
                    ))
            except Exception as e:
                pass
        
        # === OPTIONS ===
        if 'options' in intents:
            try:
                chains = obb.derivatives.options.chains(ticker)
                if chains.results:
                    # Get summary stats
                    df = chains.to_dataframe()
                    if not df.empty:
                        calls = df[df['option_type'] == 'call'] if 'option_type' in df.columns else df
                        puts = df[df['option_type'] == 'put'] if 'option_type' in df.columns else df
                        
                        text = f"""=== {ticker} Options Overview ===
Total Contracts: {len(df)}
Calls: {len(calls)} | Puts: {len(puts)}
Expirations: {df['expiration'].nunique() if 'expiration' in df.columns else 'N/A'}"""
                        
                        contexts.append(RetrievedContext(
                            source_id=f"openbb-options-{ticker}",
                            text=text,
                            relevance_score=0.72,
                            metadata={"type": "options", "ticker": ticker}
                        ))
            except Exception as e:
                pass
        
        return contexts
    
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        if not contexts:
            return "Unable to retrieve financial data. Please specify a valid ticker symbol (e.g., AAPL, MSFT, GOOGL).", []
        
        context_text = self._format_contexts(contexts)
        
        prompt = f"""Question: {query}

Financial Data:
{context_text}

Provide a comprehensive analysis answering the question. Include specific numbers and cite the data sources."""

        response = await self.model.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3
        )
        
        citations = [
            Citation(
                source_type="financial_data",
                source_id=ctx.source_id,
                text_excerpt=ctx.text[:300],
                relevance_score=ctx.relevance_score
            )
            for ctx in contexts
        ]
        
        return response.content, citations

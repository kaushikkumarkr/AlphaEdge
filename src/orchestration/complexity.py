"""
Query complexity detection for routing decisions.
Determines if a query requires multi-task decomposition.
"""
import re
from typing import List
from src.config.constants import Intent


class ComplexityDetector:
    """Detects if a query is complex and needs multi-task decomposition."""
    
    COMPLEXITY_KEYWORDS = [
        "compare", "contrast", "versus", "vs", "and also",
        "how does", "impact", "relationship", "correlation",
        "combined with", "along with", "as well as",
        "difference between", "similarities", "both",
        "together with", "in context of", "relative to",
        "compared to", "against", "alongside"
    ]
    
    MULTI_QUESTION_INDICATORS = ["?", ";", "also", "additionally"]
    
    @classmethod
    def is_complex(cls, query: str, intent: Intent | None = None) -> bool:
        """
        Determine if query requires multi-task decomposition.
        
        Args:
            query: User query string
            intent: Classified intent (if available)
        
        Returns:
            True if query is complex, False if simple
        """
        query_lower = query.lower()
        
        # Always complex if intent is SYNTHESIS
        if intent == Intent.SYNTHESIS:
            return True
        
        # Check for complexity keywords
        has_keywords = any(kw in query_lower for kw in cls.COMPLEXITY_KEYWORDS)
        
        # Check for multiple questions
        question_marks = query.count("?")
        has_semicolons = ";" in query
        has_multiple_questions = question_marks > 1 or has_semicolons
        
        # Check for multiple tickers (2+ uppercase 2-5 letter words)
        tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        # Filter out common words like USA, CEO, etc.
        common_words = {"USA", "US", "CEO", "CFO", "CTO", "AI", "ML", "IT"}
        tickers = [t for t in tickers if t not in common_words]
        has_multiple_tickers = len(tickers) > 1
        
        # Check for "and" between entities
        has_and_conjunction = " and " in query_lower and (
            has_multiple_tickers or 
            any(word in query_lower for word in ["stock", "company", "companies"])
        )
        
        # Check for cross-domain patterns (e.g., stock metrics + macro indicators)
        financial_terms = ["p/e", "pe ratio", "p/e ratio", "price", "earnings", "stock", "share", "dividend"]
        macro_terms = ["gdp", "inflation", "unemployment", "interest rate", "fed", "economy", "economic"]
        has_financial = any(term in query_lower for term in financial_terms)
        has_macro = any(term in query_lower for term in macro_terms)
        is_cross_domain = has_financial and has_macro
        
        # Complex if any indicator present
        return (
            has_keywords or 
            has_multiple_questions or 
            has_multiple_tickers or
            has_and_conjunction or
            is_cross_domain
        )
    
    @classmethod
    def get_complexity_score(cls, query: str) -> float:
        """
        Calculate complexity score (0.0-1.0).
        Higher scores indicate more complex queries.
        
        Args:
            query: User query string
        
        Returns:
            Float between 0.0 (simple) and 1.0 (very complex)
        """
        score = 0.0
        query_lower = query.lower()
        
        # Keyword presence (0.3)
        keyword_count = sum(1 for kw in cls.COMPLEXITY_KEYWORDS if kw in query_lower)
        score += min(keyword_count * 0.1, 0.3)
        
        # Multiple questions (0.2)
        if query.count("?") > 1:
            score += 0.2
        
        # Multiple tickers (0.3)
        tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        if len(tickers) > 1:
            score += min(len(tickers) * 0.15, 0.3)
        
        # Query length (0.2) - longer queries tend to be more complex
        word_count = len(query.split())
        if word_count > 15:
            score += min((word_count - 15) * 0.02, 0.2)
        
        return min(score, 1.0)
    
    @classmethod
    def get_complexity_reason(cls, query: str) -> List[str]:
        """
        Get human-readable reasons why query is complex.
        
        Args:
            query: User query string
        
        Returns:
            List of reason strings
        """
        reasons = []
        query_lower = query.lower()
        
        # Check keywords
        found_keywords = [kw for kw in cls.COMPLEXITY_KEYWORDS if kw in query_lower]
        if found_keywords:
            reasons.append(f"Contains complexity keywords: {', '.join(found_keywords[:3])}")
        
        # Check multiple questions
        if query.count("?") > 1:
            reasons.append(f"Multiple questions ({query.count('?')} question marks)")
        
        # Check multiple tickers
        tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        common_words = {"USA", "US", "CEO", "CFO", "CTO", "AI", "ML", "IT"}
        tickers = [t for t in tickers if t not in common_words]
        if len(tickers) > 1:
            reasons.append(f"Multiple tickers: {', '.join(tickers)}")
        
        # Check conjunctions
        if " and " in query_lower:
            reasons.append("Contains 'and' conjunction (potential multiple entities)")
        
        return reasons if reasons else ["Query is simple (single-domain)"]


def detect_complexity(query: str, intent: Intent | None = None) -> dict:
    """
    Convenience function to get full complexity analysis.
    
    Args:
        query: User query string
        intent: Classified intent (if available)
    
    Returns:
        Dictionary with complexity analysis
    """
    detector = ComplexityDetector()
    
    return {
        "is_complex": detector.is_complex(query, intent),
        "score": detector.get_complexity_score(query),
        "reasons": detector.get_complexity_reason(query),
        "route": "multi_task" if detector.is_complex(query, intent) else "simple"
    }

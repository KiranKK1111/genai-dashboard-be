"""
PRINCIPLE 7: Few-Shot Examples (Embedding Search + Retrieval)
=============================================================
Keep library of ~50 real question → SQL examples for your schema.
Retrieve top 2-3 most similar to current user query.
Include in LLM prompt as demonstrations.

Benefits:
1. LLM learns your schema patterns
2. Reduces hallucination by showing correct practices
3. Improves accuracy on similar questions

Examples to keep:
- "credit card customers" → join customers + cards + filter by card_type
- "debit card customers in Mumbai" → add city filter
- "customers with both credit and debit" → GROUP BY + HAVING
- "high-value transactions" → transactions JOIN customers + amount filter
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryExample:
    """A single demonstration example."""
    
    id: str
    user_question: str
    sql_query: str
    explanation: str  # What this demonstrates
    tables_used: List[str]
    complexity: str  # 'simple', 'moderate', 'complex'
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_question": self.user_question,
            "sql_query": self.sql_query,
            "explanation": self.explanation,
            "tables_used": self.tables_used,
            "complexity": self.complexity
        }


class FewShotExampleLibrary:
    """Manages example library and retrieval."""

    def __init__(self):
        self.examples: Dict[str, QueryExample] = {}
        self.embedder = None  # Will be set later

    def add_example(
        self,
        question: str,
        sql: str,
        explanation: str,
        tables: List[str],
        complexity: str = "moderate"
    ) -> str:
        """
        Add an example to the library.
        
        Args:
            question: User's natural language question
            sql: Corresponding SQL query
            explanation: What this demonstrates
            tables: Tables involved
            complexity: 'simple', 'moderate', or 'complex'
            
        Returns:
            Example ID
        """
        ex_id = f"ex_{len(self.examples):04d}"
        
        example = QueryExample(
            id=ex_id,
            user_question=question,
            sql_query=sql,
            explanation=explanation,
            tables_used=tables,
            complexity=complexity
        )
        
        self.examples[ex_id] = example
        logger.info(f"[EXAMPLES] Added: {ex_id} - {question[:50]}")
        return ex_id

    def get_default_examples(self) -> None:
        """Load default examples for banking schema."""
        default_examples = [
            {
                "q": "Show me all credit card customers",
                "sql": """
                    SELECT DISTINCT c.customer_id, c.customer_name, ca.card_type
                    FROM genai.customers c
                    LEFT JOIN genai.cards ca ON c.customer_id = ca.customer_id
                    WHERE ca.card_type = 'CREDIT'
                    LIMIT 100
                """,
                "exp": "Filter by enum value using exact match",
                "tables": ["customers", "cards"],
                "complexity": "simple"
            },
            {
                "q": "Find customers with both debit and credit cards",
                "sql": """
                    SELECT c.customer_id, c.customer_name,
                           COUNT(DISTINCT CASE WHEN ca.card_type = 'CREDIT' THEN 1 END) as credit_cards,
                           COUNT(DISTINCT CASE WHEN ca.card_type = 'DEBIT' THEN 1 END) as debit_cards
                    FROM genai.customers c
                    LEFT JOIN genai.cards ca ON c.customer_id = ca.customer_id
                    GROUP BY c.customer_id, c.customer_name
                    HAVING COUNT(DISTINCT ca.card_type) = 2
                    LIMIT 100
                """,
                "exp": "GROUP BY + HAVING to find multi-card customers (complex)",
                "tables": ["customers", "cards"],
                "complexity": "complex"
            },
            {
                "q": "Show transactions in Mumbai for VISA cards",
                "sql": """
                    SELECT t.txn_id, t.amount, t.txn_type, c.card_type, b.name as branch
                    FROM genai.transactions t
                    LEFT JOIN genai.cards c ON t.card_id = c.card_id
                    LEFT JOIN genai.branches b ON c.branch_id = b.branch_id
                    WHERE c.card_type = 'VISA'
                      AND b.city = 'Mumbai'
                    LIMIT 100
                """,
                "exp": "Multi-table join with location filter",
                "tables": ["transactions", "cards", "branches"],
                "complexity": "moderate"
            },
            {
                "q": "List customers with high account balances",
                "sql": """
                    SELECT c.customer_id, c.customer_name, a.account_no, a.balance
                    FROM genai.customers c
                    LEFT JOIN genai.accounts a ON c.customer_id = a.customer_id
                    WHERE a.balance > 500000
                    ORDER BY a.balance DESC
                    LIMIT 50
                """,
                "exp": "Filter on numeric column with ordering",
                "tables": ["customers", "accounts"],
                "complexity": "simple"
            },
            {
                "q": "Find ATM withdrawal patterns by card type",
                "sql": """
                    SELECT ca.card_type, COUNT(*) as withdrawal_count, AVG(t.amount) as avg_amount
                    FROM genai.transactions t
                    LEFT JOIN genai.cards ca ON t.card_id = ca.card_id
                    WHERE t.txn_type = 'ATM_WITHDRAWAL'
                    GROUP BY ca.card_type
                    ORDER BY withdrawal_count DESC
                """,
                "exp": "GROUP BY + aggregation on enum + filtering",
                "tables": ["transactions", "cards"],
                "complexity": "complex"
            }
        ]

        for i, ex in enumerate(default_examples):
            self.add_example(
                question=ex["q"],
                sql=ex["sql"],
                explanation=ex["exp"],
                tables=ex["tables"],
                complexity=ex["complexity"]
            )

    def set_embedder(self, embedder_func) -> None:
        """
        Set embedding function for similarity search.
        
        Args:
            embedder_func: Function that takes string → returns embedding (list or ndarray)
        """
        self.embedder = embedder_func
        logger.info("[EXAMPLES] Embedder set, computing embeddings...")
        
        # Compute embeddings for all examples
        for example in self.examples.values():
            embedding = embedder_func(example.user_question)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            example.embedding = embedding

    def find_similar_examples(
        self,
        user_question: str,
        top_k: int = 3,
        complexity_filter: Optional[str] = None
    ) -> List[QueryExample]:
        """
        Find top-K similar examples to user question.
        
        Uses embedding similarity (cosine distance).
        
        Args:
            user_question: User's query
            top_k: Number of examples to return
            complexity_filter: Optionally filter by 'simple', 'moderate', or 'complex'
            
        Returns:
            List of most similar examples
        """
        if not self.embedder:
            logger.warning("[EXAMPLES] No embedder set; returning random examples")
            examples = list(self.examples.values())
            return examples[:top_k]

        # Get query embedding
        query_embedding = self.embedder(user_question)
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        # Score all examples
        scores = []
        for ex in self.examples.values():
            if ex.embedding is None:
                continue

            # Filter by complexity if specified
            if complexity_filter and ex.complexity != complexity_filter:
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, ex.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ex.embedding) + 1e-10
            )
            scores.append((similarity, ex))

        # Sort by similarity
        scores.sort(key=lambda x: x[0], reverse=True)
        similar_examples = [ex for _, ex in scores[:top_k]]

        logger.info(f"[EXAMPLES] Found {len(similar_examples)} similar examples (top {scores[0][0]:.3f})")
        return similar_examples

    def generate_few_shot_prompt_section(
        self,
        similar_examples: List[QueryExample]
    ) -> str:
        """
        Generate prompt section with examples to include in LLM request.
        
        Args:
            similar_examples: List of QueryExample objects
            
        Returns:
            Formatted text for LLM prompt
        """
        if not similar_examples:
            return ""

        lines = [
            "\nSIMILAR EXAMPLES (use these as reference):",
            "=" * 70,
            ""
        ]

        for i, example in enumerate(similar_examples, 1):
            lines.append(f"EXAMPLE {i}: {example.explanation}")
            lines.append(f"  Question: {example.user_question}")
            lines.append(f"  SQL: {example.sql_query.strip()}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("Generate similar SQL for the user's question above.\n")

        return "\n".join(lines)


class SemanticExampleMatcher:
    """
    Advanced matching: keyword + semantic similarity.
    Falls back to keyword matching if embedding fails.
    """

    def __init__(self, library: FewShotExampleLibrary):
        self.library = library

    def find_examples_hybrid(
        self,
        user_question: str,
        top_k: int = 3
    ) -> List[QueryExample]:
        """
        Hybrid search: semantic + keyword fallback.
        
        1. If embedder available: use embedding similarity
        2. Else: keyword matching
        """
        if self.library.embedder:
            return self.library.find_similar_examples(user_question, top_k)

        # Fallback: keyword matching
        question_lower = user_question.lower()
        keywords = question_lower.split()

        scores = []
        for ex in self.library.examples.values():
            ex_lower = ex.user_question.lower()
            
            # Simple keyword match score
            matches = sum(1 for kw in keywords if kw in ex_lower)
            if matches > 0:
                scores.append((matches, ex))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scores[:top_k]]

    def extract_table_requirements(self, user_question: str) -> List[str]:
        """
        Infer which tables user likely needs.
        Used to pre-filter examples to relevant tables.
        """
        table_keywords = {
            "customers": ["customer", "clients", "users"],
            "cards": ["card", "credit", "debit", "visa", "mastercard"],
            "transactions": ["transaction", "transfer", "payment", "withdrawal", "deposit"],
            "accounts": ["account", "balance", "deposit"],
            "branches": ["branch", "office", "location", "city"]
        }

        question_lower = user_question.lower()
        required_tables = []

        for table, keywords in table_keywords.items():
            for kw in keywords:
                if kw in question_lower:
                    required_tables.append(table)
                    break

        return required_tables

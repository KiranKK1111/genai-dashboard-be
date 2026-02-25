"""
PRINCIPLE 9: Semantic Normalization Layer (Domain Synonyms)
===========================================================
Users say different things for the same concept:
- "credit card", "CC", "Visa", "MasterCard" → cards.card_type = 'CREDIT'
- "debit card", "ATM card" → cards.card_type = 'DEBIT'
- "customer list", "customer records", "all customers" → SELECT * FROM customers

Normalize intent BEFORE SQL generation:
1. Extract domain-specific synonyms from user query
2. Map to canonical database terminology
3. Pass normalized intent to LLM (or pre-filter)

Impact: Improves LLM accuracy by removing ambiguity, reduces hallucinations.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DomainConcept(str, Enum):
    """Canonical domain concepts for banking."""
    
    # Card types
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PREPAID_CARD = "prepaid_card"
    
    # Transaction types
    ATM_WITHDRAWAL = "atm_withdrawal"
    CARD_PURCHASE = "card_purchase"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    
    # Customer segments
    HIGH_VALUE_CUSTOMER = "high_value_customer"
    ACTIVE_CUSTOMER = "active_customer"
    INACTIVE_CUSTOMER = "inactive_customer"
    
    # Account types
    SAVINGS_ACCOUNT = "savings_account"
    CHECKING_ACCOUNT = "checking_account"
    BUSINESS_ACCOUNT = "business_account"


@dataclass
class NormalizationResult:
    """Result of semantic normalization."""
    
    original_text: str
    normalized_text: str
    concepts_found: List[DomainConcept]
    entity_filters: Dict[str, str]  # {column: filter_value}
    table_hints: Set[str]  # Likely tables needed
    confidence: float  # 0.0-1.0

    def to_dict(self) -> Dict:
        return {
            "original": self.original_text,
            "normalized": self.normalized_text,
            "concepts": [c.value for c in self.concepts_found],
            "entity_filters": self.entity_filters,
            "table_hints": list(self.table_hints),
            "confidence": self.confidence
        }


class SemanticNormalizer:
    """Normalize user input to canonical domain concepts."""

    def __init__(self):
        # Mapping: user variations → canonical concept
        self.synonym_map = {
            # Credit cards
            "credit card": DomainConcept.CREDIT_CARD,
            "cc": DomainConcept.CREDIT_CARD,
            "visa": DomainConcept.CREDIT_CARD,
            "mastercard": DomainConcept.CREDIT_CARD,
            "amex": DomainConcept.CREDIT_CARD,
            "rupay": DomainConcept.CREDIT_CARD,
            "diners": DomainConcept.CREDIT_CARD,

            # Debit cards
            "debit card": DomainConcept.DEBIT_CARD,
            "atm card": DomainConcept.DEBIT_CARD,
            "bank card": DomainConcept.DEBIT_CARD,

            # ATM
            "atm": DomainConcept.ATM_WITHDRAWAL,
            "cash withdrawal": DomainConcept.ATM_WITHDRAWAL,
            "withdraw": DomainConcept.ATM_WITHDRAWAL,

            # Purchases
            "purchase": DomainConcept.CARD_PURCHASE,
            "transaction": DomainConcept.CARD_PURCHASE,
            "charge": DomainConcept.CARD_PURCHASE,
            "spend": DomainConcept.CARD_PURCHASE,

            # Transfers
            "transfer": DomainConcept.TRANSFER,
            "send money": DomainConcept.TRANSFER,
            "payment": DomainConcept.TRANSFER,

            # Savings account
            "savings": DomainConcept.SAVINGS_ACCOUNT,
            "savings account": DomainConcept.SAVINGS_ACCOUNT,
            "deposit account": DomainConcept.SAVINGS_ACCOUNT,

            # Checking
            "checking": DomainConcept.CHECKING_ACCOUNT,
            "checkings": DomainConcept.CHECKING_ACCOUNT,
            "current account": DomainConcept.CHECKING_ACCOUNT,
        }

        # Mapping: concept → database filter
        self.concept_to_filter = {
            DomainConcept.CREDIT_CARD: ("card_type", "CREDIT"),
            DomainConcept.DEBIT_CARD: ("card_type", "DEBIT"),
            DomainConcept.ATM_WITHDRAWAL: ("txn_type", "ATM_WITHDRAWAL"),
            DomainConcept.CARD_PURCHASE: ("txn_type", "CARD_PURCHASE"),
            DomainConcept.TRANSFER: ("txn_type", "TRANSFER_IN"),  # Or TRANSFER_OUT
            DomainConcept.SAVINGS_ACCOUNT: ("account_type", "SAVINGS"),
            DomainConcept.CHECKING_ACCOUNT: ("account_type", "CURRENT"),
        }

        # Mapping: concept → likely tables
        self.concept_to_tables = {
            DomainConcept.CREDIT_CARD: {"cards", "customers"},
            DomainConcept.DEBIT_CARD: {"cards", "customers"},
            DomainConcept.ATM_WITHDRAWAL: {"transactions", "cards"},
            DomainConcept.CARD_PURCHASE: {"transactions", "cards"},
            DomainConcept.TRANSFER: {"transactions", "accounts"},
            DomainConcept.SAVINGS_ACCOUNT: {"accounts", "customers"},
            DomainConcept.CHECKING_ACCOUNT: {"accounts", "customers"},
        }

    def normalize(self, user_query: str) -> NormalizationResult:
        """
        Normalize user query to canonical concepts.
        
        Returns:
            NormalizationResult with normalized text, concepts, and filters
        """
        query_lower = user_query.lower()
        concepts_found = []
        entity_filters = {}
        table_hints = set()
        matched_synonyms = []

        # Find all matching concepts
        for synonym, concept in self.synonym_map.items():
            if synonym in query_lower:
                if concept not in concepts_found:
                    concepts_found.append(concept)
                    matched_synonyms.append(synonym)

                # Extract filter if available
                if concept in self.concept_to_filter:
                    column, value = self.concept_to_filter[concept]
                    entity_filters[column] = value

                # Extract tables
                if concept in self.concept_to_tables:
                    table_hints.update(self.concept_to_tables[concept])

        # Generate normalized text
        normalized_text = query_lower
        for synonym in matched_synonyms:
            # Replace synonyms with canonical concept name
            concept = self.synonym_map[synonym]
            canonical = concept.value.replace("_", " ")
            normalized_text = normalized_text.replace(synonym, f"[{canonical}]")

        # Compute confidence based on matches
        confidence = min(1.0, len(concepts_found) / 3) if concepts_found else 0.0

        logger.info(
            f"[SEMANTIC_NORM] Found {len(concepts_found)} concepts: "
            f"{[c.value for c in concepts_found]}"
        )

        return NormalizationResult(
            original_text=user_query,
            normalized_text=normalized_text,
            concepts_found=concepts_found,
            entity_filters=entity_filters,
            table_hints=table_hints,
            confidence=confidence
        )

    def generate_prompt_instruction(self, result: NormalizationResult) -> str:
        """
        Generate instruction for LLM based on normalized concepts.
        
        Args:
            result: NormalizationResult
            
        Returns:
            Prompt instruction
        """
        instruction = ""

        if result.entity_filters:
            instruction += "\nINTENTION DETECTED:\n"
            instruction += "The user is asking about:\n"

            for col, value in result.entity_filters.items():
                instruction += f"  • Filter: {col} = '{value}'\n"

        if result.table_hints:
            instruction += "\nLIKELY TABLES NEEDED:\n"
            for table in sorted(result.table_hints):
                instruction += f"  • {table}\n"

        instruction += "\nUse these hints to generate the SQL.\n"

        return instruction


class IntentClassifier:
    """Classify user intent from semantic concepts."""

    def __init__(self):
        self.intent_patterns = {
            "list": r"(show|list|display|get|find|retrieve|what are)(?:\s+all)?",
            "count": r"how many|count|total|number of",
            "filter": r"with|where|that have|containing|matching",
            "compare": r"compare|difference|vs|versus|between",
            "top": r"top|highest|largest|biggest",
            "bottom": r"bottom|lowest|smallest|last",
            "trend": r"trend|over time|growth|change|pattern",
        }

    def classify(self, user_query: str) -> List[str]:
        """
        Classify query intent from patterns.
        
        Returns:
            List of detected intents
        """
        query_lower = user_query.lower()
        intents = []

        for intent_type, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                intents.append(intent_type)

        return intents


class DomainKnowledgeBase:
    """Encodes domain-specific knowledge about the database."""

    def __init__(self):
        self.entity_relationships = {
            # Card → Customer
            "card": {"parent": "customer", "join_key": "customer_id"},
            # Account → Customer
            "account": {"parent": "customer", "join_key": "customer_id"},
            # Transaction → Card
            "transaction": {"parent": "card", "join_key": "card_id"},
            # Branch is standalone
            "branch": {"parent": None},
        }

        self.entity_attributes = {
            "card": {
                "types": ["CREDIT", "DEBIT", "PREPAID"],
                "networks": ["VISA", "MASTERCARD", "AMEX", "RUPAY"],
            },
            "account": {
                "types": ["SAVINGS", "CURRENT", "SALARY"],
            },
            "transaction": {
                "types": ["ATM_WITHDRAWAL", "CARD_PURCHASE", "TRANSFER_IN", "TRANSFER_OUT"],
            },
        }

    def get_valid_values(self, entity_type: str, attribute: str) -> List[str]:
        """Get valid values for an entity attribute."""
        if entity_type in self.entity_attributes:
            if attribute in self.entity_attributes[entity_type]:
                return self.entity_attributes[entity_type][attribute]
        return []

    def get_join_path(self, from_entity: str, to_entity: str) -> Optional[List[Tuple[str, str]]]:
        """
        Find join path between two entities.
        
        Returns:
            List of (from_table, to_table) hops, or None if not connected
        """
        # Simple BFS
        if from_entity == to_entity:
            return []

        visited = {from_entity}
        queue = [(from_entity, [])]

        while queue:
            current, path = queue.pop(0)

            for entity, info in self.entity_relationships.items():
                if entity in visited:
                    continue

                if info.get("parent") == current or (
                    # Check reverse
                    self.entity_relationships.get(current, {}).get("parent") == entity
                ):
                    new_path = path + [(current, entity)]

                    if entity == to_entity:
                        return new_path

                    visited.add(entity)
                    queue.append((entity, new_path))

        return None


class SemanticNormalizationPipeline:
    """Full pipeline: normalize → classify → enrich."""

    def __init__(self):
        self.normalizer = SemanticNormalizer()
        self.classifier = IntentClassifier()
        self.knowledge_base = DomainKnowledgeBase()

    def process(self, user_query: str) -> Dict:
        """
        Full semantic normalization pipeline.
        
        Returns:
            {
                "original_query": "...",
                "normalized": NormalizationResult,
                "intents": [...],
                "enriched_prompt_instruction": "..."
            }
        """
        # Step 1: Normalize
        normalized = self.normalizer.normalize(user_query)

        # Step 2: Classify intent
        intents = self.classifier.classify(user_query)

        # Step 3: Generate instruction
        instruction = self.normalizer.generate_prompt_instruction(normalized)
        instruction += f"\n\nINTENT: {', '.join(intents)}\n"

        result = {
            "original_query": user_query,
            "normalized": normalized.to_dict(),
            "intents": intents,
            "entity_filters": normalized.entity_filters,
            "table_hints": list(normalized.table_hints),
            "prompt_instruction": instruction,
            "confidence": normalized.confidence
        }

        logger.info(f"[SEMANTIC_PIPELINE] Normalized: {user_query[:50]}...")
        return result

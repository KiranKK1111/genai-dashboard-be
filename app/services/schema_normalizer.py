"""
Schema Normalizer - Maps domain concepts to database tables/columns.

This service:
1. Maintains canonical domain models (Transaction, Customer, Order, etc.)
2. Maps DB-specific table/column names to canonical concepts
3. Generates candidates for semantic queries
4. Provides scoring for candidate ranking

Makes the system database-agnostic by decoupling domain logic from DB schema.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DomainEntity(Enum):
    """High-level domain entities the system understands."""
    TRANSACTION = "transaction"
    CUSTOMER = "customer"
    ORDER = "order"
    PRODUCT = "product"
    MERCHANT = "merchant"
    ACCOUNT = "account"
    INVOICE = "invoice"
    PAYMENT = "payment"
    USER = "user"
    LEDGER = "ledger"
    EVENT = "event"
    AUDIT = "audit"


@dataclass
class ConceptMapping:
    """Maps a domain concept to possible DB table/column names."""
    entity: DomainEntity
    instance_synonyms: List[str]  # e.g., ["transaction", "txn", "trans", "payment"]
    id_synonyms: List[str]  # e.g., ["transaction_id", "txn_id", "id", "ref_no"]
    timestamp_synonyms: List[str]  # e.g., ["created_at", "transaction_time", "posted_at"]
    amount_synonyms: List[str]  # e.g., ["amount", "value", "total", "debit", "credit"]
    signature_columns: List[str]  # Columns that identify this entity


class SchemaNormalizer:
    """
    Maps actual database schema to canonical domain concepts.
    
    Uses:
    - Concept mappings (synonyms for tables, columns, etc.)
    - Heuristics for table/column identification
    - LLM scoring in hybrid matcher
    """
    
    def __init__(self):
        # Define canonical mappings for common domain entities
        self.concepts: Dict[DomainEntity, ConceptMapping] = {
            DomainEntity.TRANSACTION: ConceptMapping(
                entity=DomainEntity.TRANSACTION,
                instance_synonyms=[
                    "transaction", "txn", "transact", "transfer", "payment",
                    "posting", "ledger_entry", "entry", "trade", "deal"
                ],
                id_synonyms=[
                    "transaction_id", "txn_id", "trans_id", "id", "ref_no",
                    "reference_id", "txn_ref", "rrn", "utr", "trace_no"
                ],
                timestamp_synonyms=[
                    "created_at", "transaction_time", "posted_at", "value_date",
                    "transaction_date", "txn_date", "settled_at", "dated"
                ],
                amount_synonyms=[
                    "amount", "value", "total", "debit", "credit", "sum",
                    "transaction_amount", "txn_amount", "price", "cost"
                ],
                signature_columns=[
                    "amount", "transaction_id", "customer_id", "merchant_id",
                    "account_id", "timestamp", "status"
                ]
            ),
            DomainEntity.CUSTOMER: ConceptMapping(
                entity=DomainEntity.CUSTOMER,
                instance_synonyms=[
                    "customer", "cust", "client", "account_holder",
                    "user", "buyer", "party", "counterparty"
                ],
                id_synonyms=[
                    "customer_id", "cust_id", "customer_code", "cust_code",
                    "account_id", "account_number", "id"
                ],
                timestamp_synonyms=[
                    "created_at", "onboarded_at", "joined_date", "start_date"
                ],
                amount_synonyms=[],
                signature_columns=[
                    "customer_id", "customer_code", "customer_name", "account_number"
                ]
            ),
            DomainEntity.ORDER: ConceptMapping(
                entity=DomainEntity.ORDER,
                instance_synonyms=[
                    "order", "purchase_order", "po", "sale", "sales_order"
                ],
                id_synonyms=[
                    "order_id", "order_number", "order_num", "id",
                    "purchase_order_id", "po_number"
                ],
                timestamp_synonyms=[
                    "created_at", "order_date", "placed_at", "shipped_at"
                ],
                amount_synonyms=[
                    "amount", "total", "price", "cost", "order_amount"
                ],
                signature_columns=[
                    "order_id", "customer_id", "order_date", "status", "amount"
                ]
            ),
            DomainEntity.MERCHANT: ConceptMapping(
                entity=DomainEntity.MERCHANT,
                instance_synonyms=[
                    "merchant", "vendor", "seller", "supplier",
                    "store", "retailer", "business"
                ],
                id_synonyms=[
                    "merchant_id", "vendor_id", "merchant_code",
                    "merchant_num", "id"
                ],
                timestamp_synonyms=[
                    "created_at", "onboarded_at", "joined_date"
                ],
                amount_synonyms=[],
                signature_columns=[
                    "merchant_id", "merchant_name", "merchant_code", "category"
                ]
            ),
            DomainEntity.ACCOUNT: ConceptMapping(
                entity=DomainEntity.ACCOUNT,
                instance_synonyms=[
                    "account", "acct", "bank_account", "wallet",
                    "fund", "ledger"
                ],
                id_synonyms=[
                    "account_id", "acct_id", "account_number", "acct_number",
                    "iban", "swift", "id"
                ],
                timestamp_synonyms=[
                    "created_at", "opened_at", "closed_at"
                ],
                amount_synonyms=[
                    "balance", "amount", "available_balance", "ledger_balance"
                ],
                signature_columns=[
                    "account_id", "account_number", "account_type", "balance"
                ]
            ),
            DomainEntity.PAYMENT: ConceptMapping(
                entity=DomainEntity.PAYMENT,
                instance_synonyms=[
                    "payment", "pay", "disbursement", "settlement",
                    "remittance", "transfer"
                ],
                id_synonyms=[
                    "payment_id", "payment_number", "pay_id", "id",
                    "settlement_id"
                ],
                timestamp_synonyms=[
                    "created_at", "payment_date", "paid_at", "processed_at"
                ],
                amount_synonyms=[
                    "amount", "payment_amount", "value", "total"
                ],
                signature_columns=[
                    "payment_id", "payment_amount", "status", "created_at"
                ]
            ),
        }
    
    def get_synonyms_for_entity(self, entity: DomainEntity) -> Dict[str, List[str]]:
        """Get all synonyms for a domain entity."""
        mapping = self.concepts.get(entity)
        if not mapping:
            return {}
        
        return {
            "instance": mapping.instance_synonyms,
            "id": mapping.id_synonyms,
            "timestamp": mapping.timestamp_synonyms,
            "amount": mapping.amount_synonyms,
        }
    
    def guess_entity_from_term(self, term: str) -> Optional[DomainEntity]:
        """Guess which domain entity a user term refers to."""
        term_lower = term.lower().strip()
        
        for entity, mapping in self.concepts.items():
            if term_lower in mapping.instance_synonyms:
                return entity
        
        return None
    
    def find_candidate_tables(
        self,
        entity: DomainEntity,
        all_tables: Dict[str, any]
    ) -> List[Tuple[str, float]]:
        """
        Find candidate tables for an entity.
        
        Returns:
            List of (table_name, confidence) tuples, sorted by confidence descending.
        """
        if entity not in self.concepts:
            return []
        
        mapping = self.concepts[entity]
        candidates: Dict[str, float] = {}
        
        for table_name in all_tables.keys():
            table_lower = table_name.lower()
            score = 0.0
            
            # Check table name synonyms
            for synonym in mapping.instance_synonyms:
                if synonym in table_lower or table_lower in synonym:
                    # Exact match in name gets higher score
                    if table_lower == synonym:
                        score = max(score, 1.0)
                    # Substring match
                    elif synonym in table_lower or table_lower.endswith(synonym):
                        score = max(score, 0.8)
                    # Partial match
                    else:
                        score = max(score, 0.5)
            
            # Check signature columns presence (penalty/reward)
            table_obj = all_tables.get(table_name)
            if table_obj and hasattr(table_obj, 'columns'):
                col_names = set(table_obj.columns.keys())
                matching_sig_cols = sum(
                    1 for sig_col in mapping.signature_columns
                    if any(sig_col in col_name.lower() for col_name in col_names)
                )
                
                if matching_sig_cols > 0:
                    sig_score = min(0.3, matching_sig_cols * 0.1)
                    score += sig_score
            
            if score > 0:
                candidates[table_name] = min(score, 1.0)
        
        # Return sorted by confidence descending
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    def find_candidate_id_columns(
        self,
        entity: DomainEntity,
        table_name: str,
        column_names: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Find candidate ID columns for an entity in a table.
        
        Returns:
            List of (column_name, confidence) tuples.
        """
        if entity not in self.concepts:
            return []
        
        mapping = self.concepts[entity]
        candidates: Dict[str, float] = {}
        
        for column_name in column_names:
            col_lower = column_name.lower()
            score = 0.0
            
            # Check ID synonyms
            for id_syn in mapping.id_synonyms:
                if id_syn in col_lower or col_lower in id_syn:
                    if col_lower == id_syn:
                        score = max(score, 1.0)
                    elif id_syn in col_lower:
                        score = max(score, 0.9)
                    else:
                        score = max(score, 0.6)
            
            # Primary key columns get bonus
            if "id" in col_lower or "pk" in col_lower:
                score = max(score, 0.7)
            
            if score > 0:
                candidates[column_name] = score
        
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    def is_numeric_type(self, data_type: str) -> bool:
        """Check if a data type is numeric."""
        numeric_types = [
            "integer", "bigint", "smallint", "numeric",
            "decimal", "double", "float", "real", "int"
        ]
        return any(t in data_type.lower() for t in numeric_types)
    
    def is_datetime_type(self, data_type: str) -> bool:
        """Check if a data type is datetime-like."""
        datetime_types = [
            "timestamp", "date", "time", "datetime"
        ]
        return any(t in data_type.lower() for t in datetime_types)

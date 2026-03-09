"""
PROMPT INJECTION GUARDIAN - Detect and prevent prompt injection attacks.

Protects against:
- Jailbreak attempts
- System prompt manipulation
- Instruction injection
- Encoded payloads
- Role confusion attacks
"""

from __future__ import annotations

import re
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InjectionRiskLevel(str, Enum):
    """Risk levels for prompt injection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardianConfig:
    """Configuration for prompt injection guardian."""
    block_threshold: InjectionRiskLevel = InjectionRiskLevel.HIGH
    sanitize_threshold: InjectionRiskLevel = InjectionRiskLevel.MEDIUM
    enable_pattern_detection: bool = True
    enable_encoding_detection: bool = True
    enable_semantic_analysis: bool = True


@dataclass
class InjectionDetectionResult:
    """Result of injection detection."""
    is_injection: bool
    risk_level: InjectionRiskLevel
    explanation: str
    patterns_found: list[str]
    sanitized_input: Optional[str] = None


class PromptInjectionGuardian:
    """Detects and prevents prompt injection attacks."""
    
    def __init__(self, config: Optional[GuardianConfig] = None):
        """Initialize guardian with configuration."""
        self.config = config or GuardianConfig()
        
        # Common injection patterns
        self.injection_patterns = [
            (r'ignore\s+(?:previous|all|above)\s+(?:instructions|prompts|rules)', InjectionRiskLevel.HIGH),
            (r'disregard\s+(?:previous|all|above)', InjectionRiskLevel.HIGH),
            (r'forget\s+(?:previous|all|above)', InjectionRiskLevel.MEDIUM),
            (r'system\s*:\s*you\s+are', InjectionRiskLevel.HIGH),
            (r'new\s+instructions?:', InjectionRiskLevel.HIGH),
            (r'</?\s*(?:system|assistant|user)\s*>', InjectionRiskLevel.HIGH),
            (r'jailbreak|dan\s+mode|developer\s+mode', InjectionRiskLevel.CRITICAL),
            (r'\\x[0-9a-f]{2}|\\u[0-9a-f]{4}', InjectionRiskLevel.MEDIUM),  # Encoded characters
            (r'base64|decode|decrypt', InjectionRiskLevel.LOW),
        ]
    
    async def detect_injection(self, user_input: str) -> InjectionDetectionResult:
        """
        Detect if input contains prompt injection attempts.
        
        Args:
            user_input: User's input text
            
        Returns:
            InjectionDetectionResult with risk assessment
        """
        if not user_input:
            return InjectionDetectionResult(
                is_injection=False,
                risk_level=InjectionRiskLevel.NONE,
                explanation="Empty input",
                patterns_found=[],
            )
        
        input_lower = user_input.lower()
        patterns_found = []
        max_risk = InjectionRiskLevel.NONE
        
        # Pattern-based detection
        if self.config.enable_pattern_detection:
            for pattern, risk_level in self.injection_patterns:
                if re.search(pattern, input_lower, re.IGNORECASE):
                    patterns_found.append(pattern)
                    if self._risk_greater_than(risk_level, max_risk):
                        max_risk = risk_level
        
        # Encoding detection
        if self.config.enable_encoding_detection:
            if self._has_suspicious_encoding(user_input):
                patterns_found.append("suspicious_encoding")
                if self._risk_greater_than(InjectionRiskLevel.MEDIUM, max_risk):
                    max_risk = InjectionRiskLevel.MEDIUM
        
        # Determine if this is an injection
        is_injection = self._risk_greater_than(max_risk, InjectionRiskLevel.NONE)
        
        # Generate explanation
        if is_injection:
            explanation = f"Detected {len(patterns_found)} suspicious patterns: {', '.join(patterns_found[:3])}"
        else:
            explanation = "No injection patterns detected"
        
        # Sanitize if needed
        sanitized = None
        if is_injection and self._risk_greater_than(self.config.sanitize_threshold, max_risk):
            sanitized = self._sanitize_input(user_input)
        
        return InjectionDetectionResult(
            is_injection=is_injection,
            risk_level=max_risk,
            explanation=explanation,
            patterns_found=patterns_found,
            sanitized_input=sanitized,
        )
    
    def _has_suspicious_encoding(self, text: str) -> bool:
        """Check for suspicious encoding patterns."""
        # Check for excessive special characters
        special_count = len(re.findall(r'[\\<>{}[\]|]', text))
        return special_count > len(text) * 0.1
    
    def _sanitize_input(self, text: str) -> str:
        """Remove or escape suspicious patterns."""
        sanitized = text
        
        # Remove HTML-like tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Remove excessive backslashes
        sanitized = re.sub(r'\\+', ' ', sanitized)
        
        return sanitized.strip()
    
    def _risk_greater_than(self, risk1: InjectionRiskLevel, risk2: InjectionRiskLevel) -> bool:
        """Compare risk levels."""
        risk_order = {
            InjectionRiskLevel.NONE: 0,
            InjectionRiskLevel.LOW: 1,
            InjectionRiskLevel.MEDIUM: 2,
            InjectionRiskLevel.HIGH: 3,
            InjectionRiskLevel.CRITICAL: 4,
        }
        return risk_order.get(risk1, 0) > risk_order.get(risk2, 0)

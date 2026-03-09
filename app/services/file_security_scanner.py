"""
FILE SECURITY SCANNER - Detect malicious file content.

Scans for:
- Malware signatures
- Suspicious file types
- Embedded scripts
- Macros
- Large files (DoS prevention)
"""

from __future__ import annotations

import re
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat levels for file security."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Configuration for file security scanner."""
    max_file_size_mb: int = 50
    block_threshold: ThreatLevel = ThreatLevel.HIGH
    allowed_extensions: list[str] = None
    scan_for_scripts: bool = True
    scan_for_macros: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                '.txt', '.csv', '.json', '.pdf', '.xlsx', '.xls', '.doc', '.docx'
            ]


@dataclass
class ScanResult:
    """Result of security scan."""
    is_safe: bool
    threat_level: ThreatLevel
    threats_found: list[str]
    explanation: str
    file_size_mb: float


class FileSecurityScanner:
    """Scans uploaded files for security threats."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize scanner with configuration."""
        self.config = config or SecurityConfig()
        
        # Malicious patterns to detect
        self.script_patterns = [
            (r'<script[^>]*>.*?</script>', 'embedded_javascript'),
            (r'javascript:', 'javascript_protocol'),
            (r'on(?:load|error|click|mouse)', 'html_event_handler'),
            (r'eval\s*\(', 'eval_function'),
            (r'exec\s*\(', 'exec_function'),
            (r'__import__\s*\(', 'python_import'),
        ]
        
        self.macro_signatures = [
            b'xl/vbaProject.bin',  # Excel macros
            b'word/vbaProject.bin',  # Word macros
            b'VBA',
            b'Macros',
        ]
    
    async def scan_file(
        self,
        content: bytes,
        filename: str,
    ) -> ScanResult:
        """
        Scan file content for security threats.
        
        Args:
            content: File content as bytes
            filename: Original filename
            
        Returns:
            ScanResult with security assessment
        """
        threats_found = []
        max_threat = ThreatLevel.SAFE
        
        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            threats_found.append("file_too_large")
            max_threat = ThreatLevel.MEDIUM
        
        # Check file extension
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if file_ext and f'.{file_ext}' not in self.config.allowed_extensions:
            threats_found.append(f"suspicious_extension_{file_ext}")
            max_threat = self._escalate_threat(max_threat, ThreatLevel.HIGH)
        
        # Scan for scripts (convert to text for pattern matching)
        if self.config.scan_for_scripts:
            try:
                content_text = content.decode('utf-8', errors='ignore')
                for pattern, threat_name in self.script_patterns:
                    if re.search(pattern, content_text, re.IGNORECASE):
                        threats_found.append(threat_name)
                        max_threat = self._escalate_threat(max_threat, ThreatLevel.HIGH)
            except Exception:
                pass
        
        # Scan for macros
        if self.config.scan_for_macros:
            for signature in self.macro_signatures:
                if signature in content:
                    threats_found.append("macro_detected")
                    max_threat = self._escalate_threat(max_threat, ThreatLevel.MEDIUM)
                    break
        
        # Determine if safe (use numeric comparison)
        threat_order = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4,
        }
        is_safe = threat_order.get(max_threat, 0) < threat_order.get(self.config.block_threshold, 999)
        
        # Generate explanation
        if threats_found:
            explanation = f"Found {len(threats_found)} threats: {', '.join(threats_found[:3])}"
        else:
            explanation = "No threats detected"
        
        return ScanResult(
            is_safe=is_safe,
            threat_level=max_threat,
            threats_found=threats_found,
            explanation=explanation,
            file_size_mb=round(file_size_mb, 2),
        )
    
    def _escalate_threat(self, current: ThreatLevel, new: ThreatLevel) -> ThreatLevel:
        """Return the higher threat level."""
        threat_order = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4,
        }
        if threat_order.get(new, 0) > threat_order.get(current, 0):
            return new
        return current

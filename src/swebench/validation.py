#!/usr/bin/env python3
"""
SWE-bench patch validation utilities.

This module provides functions to validate generated patches against
the actual SWE-bench test harness.

Note: Full SWE-bench validation requires Docker and the official harness.
This provides a lighter-weight validation that checks:
1. Patch syntax validity (can it be parsed?)
2. Patch format (does it look like a valid unified diff?)
3. Optional: Apply patch to repo and run tests (requires Docker)
"""

import subprocess
import tempfile
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of patch validation."""
    is_valid_syntax: bool        # Can the patch be parsed?
    is_valid_format: bool        # Does it look like a unified diff?
    has_diff_header: bool        # Starts with "diff --git"?
    has_hunks: bool              # Contains @@ -x,y +x,y @@ markers?
    files_touched: int           # Number of files in the patch
    lines_added: int             # Number of + lines
    lines_removed: int           # Number of - lines
    error_message: Optional[str] # Any error that occurred
    
    @property
    def is_generated(self) -> bool:
        """Was any patch generated (even if invalid)?"""
        return True  # If we got here, something was generated
    
    @property
    def is_valid(self) -> bool:
        """Is this a valid-looking patch?"""
        return self.is_valid_syntax and self.is_valid_format and self.has_hunks


def validate_patch_syntax(patch: str) -> Tuple[bool, str]:
    """
    Check if the patch has valid syntax.
    
    Args:
        patch: Generated patch string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not patch or not patch.strip():
        return False, "Empty patch"
    
    # Check for common error patterns
    if patch.startswith("# Error"):
        return False, "Patch generation failed"
    
    if "```" in patch:
        return False, "Patch contains markdown code blocks (not cleaned)"
    
    return True, ""


def validate_patch_format(patch: str) -> Dict[str, bool]:
    """
    Check if the patch looks like a valid unified diff.
    
    Args:
        patch: Generated patch string
        
    Returns:
        Dictionary with format validation results
    """
    lines = patch.strip().split("\n")
    
    has_diff_header = any(line.startswith("diff --git") for line in lines)
    has_file_markers = any(line.startswith("---") or line.startswith("+++") for line in lines)
    has_hunks = bool(re.search(r"@@ -\d+,?\d* \+\d+,?\d* @@", patch))
    
    return {
        "has_diff_header": has_diff_header,
        "has_file_markers": has_file_markers,
        "has_hunks": has_hunks,
        "is_valid_format": has_diff_header and has_hunks
    }


def count_patch_stats(patch: str) -> Dict[str, int]:
    """
    Count statistics about the patch.
    
    Args:
        patch: Generated patch string
        
    Returns:
        Dictionary with patch statistics
    """
    lines = patch.strip().split("\n")
    
    # Count files (diff --git lines)
    files_touched = sum(1 for line in lines if line.startswith("diff --git"))
    
    # Count added/removed lines (lines starting with +/- but not +++/---)
    lines_added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    lines_removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    
    return {
        "files_touched": files_touched,
        "lines_added": lines_added,
        "lines_removed": lines_removed
    }


def validate_patch(patch: str) -> ValidationResult:
    """
    Comprehensive patch validation.
    
    Args:
        patch: Generated patch string
        
    Returns:
        ValidationResult with all validation checks
    """
    # Check syntax
    is_valid_syntax, error_msg = validate_patch_syntax(patch)
    
    if not is_valid_syntax:
        return ValidationResult(
            is_valid_syntax=False,
            is_valid_format=False,
            has_diff_header=False,
            has_hunks=False,
            files_touched=0,
            lines_added=0,
            lines_removed=0,
            error_message=error_msg
        )
    
    # Check format
    format_result = validate_patch_format(patch)
    
    # Count stats
    stats = count_patch_stats(patch)
    
    return ValidationResult(
        is_valid_syntax=True,
        is_valid_format=format_result["is_valid_format"],
        has_diff_header=format_result["has_diff_header"],
        has_hunks=format_result["has_hunks"],
        files_touched=stats["files_touched"],
        lines_added=stats["lines_added"],
        lines_removed=stats["lines_removed"],
        error_message=None
    )


def try_apply_patch(patch: str, repo_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Try to apply a patch using git apply --check.
    
    This is a lightweight check that doesn't require the full SWE-bench harness.
    
    Args:
        patch: Generated patch string
        repo_path: Path to git repository (or None to use temp dir)
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch)
            patch_file = f.name
        
        # Try to parse the patch with git apply --check
        # Note: This won't actually apply, just checks if it's valid
        cmd = ["git", "apply", "--check", patch_file]
        if repo_path:
            cmd = ["git", "-C", repo_path] + cmd[1:]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        os.unlink(patch_file)
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout while checking patch"
    except Exception as e:
        return False, str(e)


# For full SWE-bench validation with Docker, see:
# https://github.com/princeton-nlp/SWE-bench
# 
# The full validation requires:
# 1. Docker with SWE-bench harness
# 2. Cloning the target repository
# 3. Applying the patch
# 4. Running the test suite
# 5. Checking if the failing test now passes
#
# This is computationally expensive and typically done as a separate
# batch process after all patches are generated.

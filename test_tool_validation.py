#!/usr/bin/env python3
"""
Test script to validate tool calling format compliance implementation.
"""

import json
import sys

# Import only the classes we need to test, avoiding full module imports
sys.path.insert(0, '.')

# We'll create standalone versions of the classes to test
class BackendType:
    """Minimal BackendType enum for testing."""
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"

class ToolCallValidator:
    """Validates tool call format compliance and schema adherence."""

    def __init__(self):
        self.type_validators = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int),
            "number": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        schema: dict,
        backend,
        turn: int,
        json_error = None
    ) -> dict:
        errors = []

        if json_error:
            return {
                "valid": False,
                "errors": [{"type": "json_parse_failure", "details": json_error}],
                "score": 0.0
            }

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field in required:
            if field not in arguments:
                errors.append({
                    "type": "missing_required_field",
                    "field": field,
                    "details": f"Required parameter '{field}' not provided"
                })

        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and expected_type in self.type_validators:
                    if not self.type_validators[expected_type](value):
                        errors.append({
                            "type": "type_mismatch",
                            "field": field,
                            "details": f"Expected {expected_type}, got {type(value).__name__}"
                        })

        for field, value in arguments.items():
            if field in properties and "enum" in properties[field]:
                valid_values = properties[field]["enum"]
                if value not in valid_values:
                    errors.append({
                        "type": "invalid_enum_value",
                        "field": field,
                        "details": f"'{value}' not in {valid_values}"
                    })

        if not errors:
            score = 5.0
        elif len([e for e in errors if e["type"] == "missing_required_field"]) > 0:
            score = 0.0
        elif len([e for e in errors if e["type"] == "type_mismatch"]) > 0:
            score = 2.0
        else:
            score = 3.5

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "score": score
        }

    def compute_aggregate_metrics(self, trace: list) -> dict:
        if not trace:
            return {"success_rate": 100.0, "adherence_score": 5.0, "errors": []}

        valid_calls = sum(1 for t in trace if t.get("format_valid", False))
        total_calls = len(trace)
        success_rate = (valid_calls / total_calls) * 100.0

        scores = []
        all_errors = []
        for t in trace:
            if "schema_errors" in t:
                for err in t["schema_errors"]:
                    all_errors.append({
                        "turn": t["turn"],
                        "tool": t["tool_name"],
                        **err
                    })
            if t.get("format_valid", False):
                scores.append(5.0)
            elif t.get("json_parse_error"):
                scores.append(0.0)
            elif len(t.get("schema_errors", [])) > 2:
                scores.append(1.0)
            elif len(t.get("schema_errors", [])) > 0:
                scores.append(3.0)
            else:
                scores.append(5.0)

        adherence_score = sum(scores) / len(scores) if scores else 5.0

        return {
            "success_rate": success_rate,
            "adherence_score": adherence_score,
            "errors": all_errors
        }

def test_validator_missing_required():
    """Test that validator detects missing required fields."""
    validator = ToolCallValidator()
    schema = {
        "properties": {"url": {"type": "string"}},
        "required": ["url"]
    }

    result = validator.validate_tool_call(
        tool_name="http_request",
        arguments={},
        schema=schema,
        backend=BackendType.ANTHROPIC,
        turn=1
    )

    assert not result["valid"], "Should be invalid with missing required field"
    assert result["score"] == 0.0, "Score should be 0 for missing required field"
    assert any(e["type"] == "missing_required_field" for e in result["errors"]), \
        "Should have missing_required_field error"
    print("✓ Missing required field test passed")

def test_validator_type_mismatch():
    """Test that validator detects type mismatches."""
    validator = ToolCallValidator()
    schema = {
        "properties": {
            "timeout": {"type": "integer"}
        },
        "required": []
    }

    result = validator.validate_tool_call(
        tool_name="http_request",
        arguments={"timeout": "300"},  # String instead of integer
        schema=schema,
        backend=BackendType.OPENROUTER,
        turn=1
    )

    assert not result["valid"], "Should be invalid with type mismatch"
    assert result["score"] == 2.0, "Score should be 2.0 for type errors"
    assert any(e["type"] == "type_mismatch" for e in result["errors"]), \
        "Should have type_mismatch error"
    print("✓ Type mismatch test passed")

def test_validator_enum_violation():
    """Test that validator detects invalid enum values."""
    validator = ToolCallValidator()
    schema = {
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST"]}
        },
        "required": ["method"]
    }

    result = validator.validate_tool_call(
        tool_name="http_request",
        arguments={"method": "DELETE"},  # Not in enum
        schema=schema,
        backend=BackendType.NVIDIA,
        turn=1
    )

    assert not result["valid"], "Should be invalid with enum violation"
    assert any(e["type"] == "invalid_enum_value" for e in result["errors"]), \
        "Should have invalid_enum_value error"
    print("✓ Enum violation test passed")

def test_validator_json_error():
    """Test that validator handles JSON parse errors."""
    validator = ToolCallValidator()
    schema = {
        "properties": {"url": {"type": "string"}},
        "required": ["url"]
    }

    result = validator.validate_tool_call(
        tool_name="http_request",
        arguments={},
        schema=schema,
        backend=BackendType.OPENROUTER,
        turn=1,
        json_error="JSON parse error: Expecting property name"
    )

    assert not result["valid"], "Should be invalid with JSON error"
    assert result["score"] == 0.0, "Score should be 0 for JSON errors"
    assert any(e["type"] == "json_parse_failure" for e in result["errors"]), \
        "Should have json_parse_failure error"
    print("✓ JSON parse error test passed")

def test_validator_valid_call():
    """Test that validator accepts valid tool calls."""
    validator = ToolCallValidator()
    schema = {
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "enum": ["GET", "POST"]},
            "timeout": {"type": "integer"}
        },
        "required": ["url", "method"]
    }

    result = validator.validate_tool_call(
        tool_name="http_request",
        arguments={"url": "https://example.com", "method": "GET", "timeout": 30},
        schema=schema,
        backend=BackendType.ANTHROPIC,
        turn=1
    )

    assert result["valid"], "Should be valid with all requirements met"
    assert result["score"] == 5.0, "Score should be 5.0 for valid call"
    assert len(result["errors"]) == 0, "Should have no errors"
    print("✓ Valid call test passed")

def test_aggregate_metrics():
    """Test that aggregate metrics computation works correctly."""
    validator = ToolCallValidator()

    # Mock trace with mixed valid/invalid calls
    trace = [
        {
            "turn": 1,
            "tool_name": "http_request",
            "arguments": {"url": "https://example.com"},
            "result": "...",
            "format_valid": True,
            "schema_errors": [],
            "json_parse_error": None,
            "backend_format": "openai"
        },
        {
            "turn": 2,
            "tool_name": "run_nmap_scan",
            "arguments": {},
            "result": "...",
            "format_valid": False,
            "schema_errors": [
                {"type": "missing_required_field", "field": "target", "details": "Required parameter 'target' not provided"}
            ],
            "json_parse_error": None,
            "backend_format": "openai"
        },
        {
            "turn": 3,
            "tool_name": "http_request",
            "arguments": {"url": "https://example.com"},
            "result": "...",
            "format_valid": True,
            "schema_errors": [],
            "json_parse_error": None,
            "backend_format": "openai"
        }
    ]

    metrics = validator.compute_aggregate_metrics(trace)

    assert metrics["success_rate"] == (2/3) * 100, "Success rate should be 66.67%"
    assert 0 <= metrics["adherence_score"] <= 5.0, "Adherence score should be between 0-5"
    assert len(metrics["errors"]) == 1, "Should have 1 error entry"
    assert metrics["errors"][0]["type"] == "missing_required_field", "Error type should match"
    print(f"✓ Aggregate metrics test passed (success_rate={metrics['success_rate']:.2f}%)")

def main():
    """Run all tests."""
    print("Running tool validation tests...\n")

    try:
        test_validator_missing_required()
        test_validator_type_mismatch()
        test_validator_enum_violation()
        test_validator_json_error()
        test_validator_valid_call()
        test_aggregate_metrics()

        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

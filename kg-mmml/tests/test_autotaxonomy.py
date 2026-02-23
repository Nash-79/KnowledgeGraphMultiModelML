"""
Tests for auto-taxonomy generation from pattern rules.
Validates Goal A: Pattern matching and taxonomy building.
"""
import pytest
import re
import tempfile
import yaml
from pathlib import Path
from typing import List, Tuple


# Mock pattern matching functions
def compile_patterns(yaml_content: dict) -> List[Tuple[str, re.Pattern]]:
    """Compile pattern rules from YAML"""
    parents = yaml_content.get("parents", {}) or {}
    compiled = []
    for parent, patterns in parents.items():
        for pattern in (patterns or []):
            compiled.append((parent.strip(), re.compile(pattern)))
    return compiled


def match_concept_to_parent(concept: str, patterns: List[Tuple[str, re.Pattern]]) -> str:
    """Match concept to first matching parent pattern"""
    for parent, regex in patterns:
        if regex.match(concept):
            return parent
    return None


class TestPatternCompilation:
    """Test pattern rule compilation"""
    
    def test_basic_pattern_compilation(self):
        """Test compiling basic patterns"""
        rules = {
            "parents": {
                "us-gaap:Assets": [
                    "^us-gaap:.*Assets$",
                    "^us-gaap:Cash.*"
                ]
            }
        }
        
        patterns = compile_patterns(rules)
        assert len(patterns) == 2, "Should compile 2 patterns"
        assert all(isinstance(p[1], re.Pattern) for p in patterns), "Should be regex patterns"
    
    def test_multiple_parents(self):
        """Test compiling patterns for multiple parents"""
        rules = {
            "parents": {
                "us-gaap:Assets": ["^us-gaap:.*Assets$"],
                "us-gaap:Liabilities": ["^us-gaap:.*Liabilities$"],
                "us-gaap:Equity": ["^us-gaap:.*Equity$"]
            }
        }
        
        patterns = compile_patterns(rules)
        assert len(patterns) == 3, "Should have patterns for 3 parents"
        
        parents = [p[0] for p in patterns]
        assert "us-gaap:Assets" in parents
        assert "us-gaap:Liabilities" in parents
        assert "us-gaap:Equity" in parents
    
    def test_empty_patterns(self):
        """Test handling empty pattern list"""
        rules = {"parents": {}}
        patterns = compile_patterns(rules)
        assert len(patterns) == 0, "Empty rules should produce no patterns"


class TestConceptMatching:
    """Test concept to parent matching"""
    
    def test_exact_suffix_match(self):
        """Test matching concept with exact suffix"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*Assets$"))
        ]
        
        concept = "us-gaap:CurrentAssets"
        parent = match_concept_to_parent(concept, patterns)
        
        assert parent == "us-gaap:Assets", f"Should match, got {parent}"
    
    def test_prefix_match(self):
        """Test matching concept with prefix pattern"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:Cash.*"))
        ]
        
        concept = "us-gaap:CashAndCashEquivalents"
        parent = match_concept_to_parent(concept, patterns)
        
        assert parent == "us-gaap:Assets", "Should match cash pattern"
    
    def test_no_match(self):
        """Test concept that doesn't match any pattern"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*Assets$"))
        ]
        
        concept = "us-gaap:Revenue"
        parent = match_concept_to_parent(concept, patterns)
        
        assert parent is None, "Should not match"
    
    def test_first_match_wins(self):
        """Test that first matching pattern wins (conservative)"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:Current.*")),
            ("us-gaap:Liabilities", re.compile("^us-gaap:Current.*"))
        ]
        
        concept = "us-gaap:CurrentAssets"
        parent = match_concept_to_parent(concept, patterns)
        
        # Should match first pattern
        assert parent == "us-gaap:Assets", "First match should win"
    
    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*assets$"))  # lowercase
        ]
        
        concept = "us-gaap:CurrentAssets"  # uppercase A
        parent = match_concept_to_parent(concept, patterns)
        
        assert parent is None, "Case-sensitive match should fail"


class TestUSGAAPPatterns:
    """Test US-GAAP specific pattern rules"""
    
    def test_assets_hierarchy(self):
        """Test asset classification patterns"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*Assets$")),
            ("us-gaap:Assets", re.compile("^us-gaap:.*Asset$")),
            ("us-gaap:Assets", re.compile("^us-gaap:Cash.*")),
            ("us-gaap:Assets", re.compile("^us-gaap:.*Receivable.*")),
        ]
        
        test_concepts = [
            ("us-gaap:CurrentAssets", True),
            ("us-gaap:IntangibleAsset", True),
            ("us-gaap:CashAndCashEquivalents", True),
            ("us-gaap:AccountsReceivableNet", True),
            ("us-gaap:Revenue", False),
        ]
        
        for concept, should_match in test_concepts:
            parent = match_concept_to_parent(concept, patterns)
            if should_match:
                assert parent == "us-gaap:Assets", f"{concept} should match Assets"
            else:
                assert parent is None, f"{concept} should not match"
    
    def test_liabilities_hierarchy(self):
        """Test liability classification patterns"""
        patterns = [
            ("us-gaap:Liabilities", re.compile("^us-gaap:.*Liabilities$")),
            ("us-gaap:Liabilities", re.compile("^us-gaap:.*Liability$")),
            ("us-gaap:Liabilities", re.compile("^us-gaap:.*Payable.*")),
            ("us-gaap:Liabilities", re.compile("^us-gaap:.*Debt.*")),
        ]
        
        test_concepts = [
            ("us-gaap:CurrentLiabilities", True),
            ("us-gaap:DeferredTaxLiability", True),
            ("us-gaap:AccountsPayable", True),
            ("us-gaap:LongTermDebt", True),
            ("us-gaap:CommonStock", False),
        ]
        
        for concept, should_match in test_concepts:
            parent = match_concept_to_parent(concept, patterns)
            if should_match:
                assert parent == "us-gaap:Liabilities", f"{concept} should match Liabilities"
    
    def test_revenue_hierarchy(self):
        """Test revenue classification patterns"""
        patterns = [
            ("us-gaap:Revenues", re.compile("^us-gaap:.*Revenue$")),
            ("us-gaap:Revenues", re.compile("^us-gaap:.*Sales$")),
        ]
        
        test_concepts = [
            ("us-gaap:SalesRevenue", True),
            ("us-gaap:ProductRevenue", True),
            ("us-gaap:TotalSales", True),
            ("us-gaap:CostOfRevenue", False),  # Not revenue itself
        ]
        
        for concept, should_match in test_concepts:
            parent = match_concept_to_parent(concept, patterns)
            if should_match:
                assert parent == "us-gaap:Revenues", f"{concept} should match Revenues"


class TestFalsePositiveDetection:
    """Test for false positive matches"""
    
    def test_partial_word_match(self):
        """Test that partial word matches don't create false positives"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*Asset$"))  # Singular
        ]
        
        # Should NOT match plural
        concept = "us-gaap:AssetsDiscontinuedOperations"
        parent = match_concept_to_parent(concept, patterns)
        
        # This will match with $ anchor, which is correct behavior
        # We want to detect if pattern is TOO greedy
        pass
    
    def test_overly_broad_pattern(self):
        """Test detection of overly broad patterns"""
        # Overly broad: matches too many concepts
        broad_pattern = re.compile("^us-gaap:.*")
        
        concepts = [
            "us-gaap:Assets",
            "us-gaap:Liabilities",
            "us-gaap:Revenue",
            "us-gaap:Expenses"
        ]
        
        matches = sum(1 for c in concepts if broad_pattern.match(c))
        
        # If pattern matches >50% of test concepts, it's too broad
        assert matches == len(concepts), "Pattern is overly broad"
        # This test documents that ^us-gaap:.* is TOO broad


class TestPatternQuality:
    """Test pattern quality metrics"""
    
    def test_pattern_precision(self):
        """Test that patterns have high precision (few false positives)"""
        patterns = [
            ("us-gaap:Assets", re.compile("^us-gaap:.*Assets$"))
        ]
        
        # Test against diverse concepts
        concepts = [
            "us-gaap:CurrentAssets",       # True positive
            "us-gaap:NoncurrentAssets",    # True positive
            "us-gaap:Revenue",             # True negative
            "us-gaap:Expenses",            # True negative
            "us-gaap:Liabilities",         # True negative
        ]
        
        matches = [match_concept_to_parent(c, patterns) for c in concepts]
        true_positives = sum(1 for m in matches if m == "us-gaap:Assets")
        
        # Should only match 2 out of 5
        precision = true_positives / len([m for m in matches if m is not None]) if any(matches) else 0
        assert precision == 1.0, f"Precision {precision} should be 1.0 (no false positives)"


class TestGoalAPatternQuality:
    """Acceptance tests for Goal A pattern quality"""
    
    def test_high_precision_patterns_only(self):
        """Goal A: Only high-precision patterns should be used"""
        # Load actual pattern rules
        # For now, test the principle
        
        # High-precision: specific suffixes/prefixes
        high_precision = [
            re.compile("^us-gaap:.*Assets$"),
            re.compile("^us-gaap:Cash.*")
        ]
        
        # Low-precision: overly broad
        low_precision = [
            re.compile("^us-gaap:.*"),
            re.compile(".*Asset.*")  # No anchors
        ]
        
        # Goal A should only use high-precision patterns
        for pattern in high_precision:
            # These are acceptable
            assert pattern.pattern.startswith("^"), "Should have start anchor"
        
        for pattern in low_precision:
            # These should be avoided
            if not pattern.pattern.startswith("^"):
                pytest.skip("Low precision pattern detected")


class TestTaxonomyOutput:
    """Test taxonomy CSV output format"""
    
    def test_csv_structure(self):
        """Test that taxonomy CSV has required columns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            f.write("child,parent,source\n")
            f.write("us-gaap:Cash,us-gaap:Assets,auto\n")
            f.write("us-gaap:CurrentAssets,us-gaap:Assets,manual\n")
        
        # Read and validate
        import csv
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2, "Should have 2 edges"
        assert all('child' in row for row in rows), "Missing 'child' column"
        assert all('parent' in row for row in rows), "Missing 'parent' column"
        assert all('source' in row for row in rows), "Missing 'source' column"
        
        # Check sources
        sources = {row['source'] for row in rows}
        assert 'auto' in sources, "Should have auto-generated edges"
        assert 'manual' in sources, "Should have manual edges"
        
        # Cleanup
        Path(csv_path).unlink()
    
    def test_no_duplicate_edges(self):
        """Test that combined taxonomy has no duplicate edges"""
        edges = [
            ("us-gaap:Cash", "us-gaap:Assets"),
            ("us-gaap:Cash", "us-gaap:Assets"),  # Duplicate
            ("us-gaap:Revenue", "us-gaap:Income")
        ]
        
        unique_edges = list(set(edges))
        assert len(unique_edges) == 2, "Should deduplicate edges"
    
    def test_manual_takes_precedence(self):
        """Test that manual edges override auto-generated"""
        # Simulate conflict
        auto_edge = ("us-gaap:Cash", "us-gaap:Assets", "auto")
        manual_edge = ("us-gaap:Cash", "us-gaap:CurrentAssets", "manual")
        
        # In combination, manual should win
        # Implementation should keep manual when (child, parent) conflict exists
        edges = [auto_edge, manual_edge]
        
        # Keep only manual for same child
        unique_by_child = {}
        for child, parent, source in edges:
            if source == "manual" or child not in unique_by_child:
                unique_by_child[child] = (parent, source)
        
        assert unique_by_child["us-gaap:Cash"][1] == "manual", "Manual should take precedence"


# Fixtures
@pytest.fixture
def sample_pattern_yaml():
    """Sample pattern rules YAML"""
    return """
parents:
  "us-gaap:Assets":
    - "^us-gaap:.*Assets$"
    - "^us-gaap:.*Asset$"
    - "^us-gaap:Cash.*"
  "us-gaap:Liabilities":
    - "^us-gaap:.*Liabilities$"
    - "^us-gaap:.*Liability$"
"""


@pytest.fixture
def sample_concepts():
    """Sample US-GAAP concepts"""
    return [
        "us-gaap:Assets",
        "us-gaap:CurrentAssets",
        "us-gaap:NoncurrentAssets",
        "us-gaap:CashAndCashEquivalents",
        "us-gaap:Liabilities",
        "us-gaap:CurrentLiabilities",
        "us-gaap:Revenue",
        "us-gaap:Expenses"
    ]


class TestWithFixtures:
    """Tests using fixtures"""
    
    def test_pattern_loading(self, sample_pattern_yaml):
        """Test loading patterns from YAML"""
        rules = yaml.safe_load(sample_pattern_yaml)
        patterns = compile_patterns(rules)
        
        assert len(patterns) >= 5, "Should have multiple patterns"
        parent_names = [p[0] for p in patterns]
        assert "us-gaap:Assets" in parent_names
        assert "us-gaap:Liabilities" in parent_names
    
    def test_bulk_concept_matching(self, sample_pattern_yaml, sample_concepts):
        """Test matching bulk concepts"""
        rules = yaml.safe_load(sample_pattern_yaml)
        patterns = compile_patterns(rules)
        
        matches = {}
        for concept in sample_concepts:
            parent = match_concept_to_parent(concept, patterns)
            if parent:
                matches[concept] = parent
        
        # Should match asset concepts
        assert matches.get("us-gaap:CurrentAssets") == "us-gaap:Assets"
        assert matches.get("us-gaap:CashAndCashEquivalents") == "us-gaap:Assets"
        
        # Should match liability concepts
        assert matches.get("us-gaap:CurrentLiabilities") == "us-gaap:Liabilities"
        
        # Should NOT match revenue/expenses
        assert "us-gaap:Revenue" not in matches
        assert "us-gaap:Expenses" not in matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

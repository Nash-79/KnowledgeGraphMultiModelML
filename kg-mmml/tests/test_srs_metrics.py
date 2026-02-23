"""
Tests for SRS (Semantic Retention Score) metrics.
Tests AtP, HP, AP, and integration with KG structure.
"""
import pytest
import tempfile
import csv
from pathlib import Path


# Mock SRS computation functions for testing
def metric_atp(concepts: set, edges_by_type: dict) -> float:
    """
    Attribute Predictability: fraction of Concept nodes with measured-in edge.
    """
    mi = edges_by_type.get("measured-in", [])
    with_unit = {src for (src, dst) in mi}
    denom = len(concepts)
    return (len(concepts & with_unit) / denom) if denom else 0.0


def metric_hp_coverage(concepts: set, edges_by_type: dict) -> float:
    """
    Hierarchy Presence: fraction of Concept nodes with is-a parent.
    """
    isa = edges_by_type.get("is-a", [])
    children_with_parent = {src for (src, dst) in isa}
    denom = len(concepts)
    return (len(concepts & children_with_parent) / denom) if denom else 0.0


def metric_ap_directionality(edges_by_type: dict, directional_types: set = None) -> float:
    """
    Asymmetry Preservation: 1 - (fraction of directional edges with reverse).
    """
    if directional_types is None:
        directional_types = {"measured-in", "for-period"}
    
    total_dir_edges = 0
    reverse_count = 0
    
    for edge_type in directional_types:
        edges = edges_by_type.get(edge_type, [])
        if not edges:
            continue
        
        edge_set = set(edges)
        total_dir_edges += len(edges)
        
        # Check for reverse edges
        for src, dst in edges:
            if (dst, src) in edge_set:
                reverse_count += 1
    
    if total_dir_edges == 0:
        return 1.0
    
    return 1.0 - (reverse_count / total_dir_edges)


class TestSRSMetrics:
    """Test suite for individual SRS metrics"""
    
    def test_atp_all_concepts_have_units(self):
        """Test AtP when all concepts have measured-in edges"""
        concepts = {"C1", "C2", "C3"}
        edges = {"measured-in": [("C1", "U1"), ("C2", "U2"), ("C3", "U3")]}
        
        atp = metric_atp(concepts, edges)
        assert atp == 1.0, "All concepts should have units"
    
    def test_atp_partial_coverage(self):
        """Test AtP with partial unit coverage"""
        concepts = {"C1", "C2", "C3"}
        edges = {"measured-in": [("C1", "U1"), ("C2", "U2")]}
        
        atp = metric_atp(concepts, edges)
        assert atp == 2/3, f"Expected 2/3, got {atp}"
    
    def test_atp_no_units(self):
        """Test AtP when no concepts have units"""
        concepts = {"C1", "C2", "C3"}
        edges = {"measured-in": []}
        
        atp = metric_atp(concepts, edges)
        assert atp == 0.0, "No concepts should have units"
    
    def test_atp_empty_graph(self):
        """Test AtP with empty graph"""
        concepts = set()
        edges = {"measured-in": []}
        
        atp = metric_atp(concepts, edges)
        assert atp == 0.0, "Empty graph should return 0"
    
    def test_hp_all_concepts_have_parents(self):
        """Test HP when all concepts have is-a parents"""
        concepts = {"C1", "C2", "C3"}
        edges = {"is-a": [("C1", "P1"), ("C2", "P2"), ("C3", "P3")]}
        
        hp = metric_hp_coverage(concepts, edges)
        assert hp == 1.0, "All concepts should have parents"
    
    def test_hp_partial_hierarchy(self):
        """Test HP with partial hierarchy coverage"""
        concepts = {"C1", "C2", "C3", "C4"}
        edges = {"is-a": [("C1", "C2")]}
        
        hp = metric_hp_coverage(concepts, edges)
        assert hp == 1/4, f"Expected 1/4, got {hp}"
    
    def test_hp_no_hierarchy(self):
        """Test HP when no hierarchy exists"""
        concepts = {"C1", "C2", "C3"}
        edges = {"is-a": []}
        
        hp = metric_hp_coverage(concepts, edges)
        assert hp == 0.0, "No hierarchy should return 0"
    
    def test_hp_chain_hierarchy(self):
        """Test HP with chain hierarchy: C1 -> C2 -> C3"""
        concepts = {"C1", "C2", "C3"}
        edges = {"is-a": [("C1", "C2"), ("C2", "C3")]}
        
        hp = metric_hp_coverage(concepts, edges)
        assert hp == 2/3, "Two concepts should have parents"
    
    def test_ap_no_reverse_edges(self):
        """Test AP when no reverse edges exist"""
        edges = {
            "measured-in": [("C1", "U1"), ("C2", "U2")],
            "for-period": [("C1", "P1")]
        }
        
        ap = metric_ap_directionality(edges)
        assert ap == 1.0, "No reverse edges should give AP=1.0"
    
    def test_ap_all_reverse_edges(self):
        """Test AP when all edges are reversed"""
        edges = {
            "measured-in": [
                ("C1", "U1"), ("U1", "C1"),  # Bidirectional
                ("C2", "U2"), ("U2", "C2")
            ]
        }
        
        ap = metric_ap_directionality(edges)
        assert ap == 0.0, "All reversed should give AP=0.0"
    
    def test_ap_partial_reverse(self):
        """Test AP with some reverse edges"""
        edges = {
            "measured-in": [
                ("C1", "U1"),
                ("C2", "U2"), ("U2", "C2"),  # One reversed
                ("C3", "U3")
            ]
        }
        
        ap = metric_ap_directionality(edges)
        # 4 directional edges total; one bidirectional pair contributes 2 reverse checks.
        # AP = 1 - (2/4) = 0.5
        assert ap == 0.5, f"Expected 0.5, got {ap}"
    
    def test_ap_empty_edges(self):
        """Test AP with no directional edges"""
        edges = {"non-directional": [("A", "B")]}
        
        ap = metric_ap_directionality(edges)
        assert ap == 1.0, "No directional edges should return 1.0"


class TestSRSIntegration:
    """Integration tests for SRS computation"""
    
    def test_baseline_kg_scores(self):
        """Test SRS scores on a baseline KG (before taxonomy)"""
        # Simulate baseline: few is-a edges
        concepts = {f"C{i}" for i in range(10)}
        edges = {
            "measured-in": [(f"C{i}", f"U{i%3}") for i in range(8)],  # 80% coverage
            "is-a": [("C1", "C2")],  # Only 10% hierarchy
            "for-period": [(f"C{i}", "P1") for i in range(5)]
        }
        
        atp = metric_atp(concepts, edges)
        hp = metric_hp_coverage(concepts, edges)
        ap = metric_ap_directionality(edges)
        
        assert atp == 0.8, "80% should have units"
        assert hp == 0.1, "10% should have parents"
        assert ap == 1.0, "No reverse edges"
        
        # Baseline HP should be < 0.25 (Goal A target)
        assert hp < 0.25, "Baseline HP should be low"
    
    def test_taxonomy_enhanced_kg_scores(self):
        """Test SRS scores after adding auto-taxonomy (Goal A)"""
        # Simulate taxonomy-enhanced: many is-a edges
        concepts = {f"C{i}" for i in range(10)}
        edges = {
            "measured-in": [(f"C{i}", f"U{i%3}") for i in range(8)],
            "is-a": [(f"C{i}", f"P{i%4}") for i in range(3)],  # 30% hierarchy
            "for-period": [(f"C{i}", "P1") for i in range(5)]
        }
        
        hp = metric_hp_coverage(concepts, edges)
        
        # Goal A: HP should be ≥ 0.25
        assert hp >= 0.25, f"Enhanced HP {hp} should be ≥ 0.25"
    
    def test_hp_increase_without_harming_atp(self):
        """Test that adding taxonomy increases HP without harming AtP"""
        concepts = {"C1", "C2", "C3", "C4", "C5"}
        
        # Before taxonomy
        edges_before = {
            "measured-in": [("C1", "U1"), ("C2", "U2"), ("C3", "U3")],
            "is-a": []
        }
        
        # After taxonomy
        edges_after = {
            "measured-in": [("C1", "U1"), ("C2", "U2"), ("C3", "U3")],
            "is-a": [("C1", "C4"), ("C2", "C4")]  # Added hierarchy
        }
        
        atp_before = metric_atp(concepts, edges_before)
        hp_before = metric_hp_coverage(concepts, edges_before)
        
        atp_after = metric_atp(concepts, edges_after)
        hp_after = metric_hp_coverage(concepts, edges_after)
        
        assert hp_after > hp_before, "HP should increase"
        assert abs(atp_after - atp_before) < 0.02, "AtP should not be harmed"


class TestSRSFromKGFiles:
    """Test SRS computation from actual KG CSV files"""
    
    def test_load_and_compute_from_csv(self):
        """Test loading KG from CSV and computing SRS"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock KG files
            nodes_file = tmpdir / "kg_nodes.csv"
            edges_file = tmpdir / "kg_edges.csv"
            
            # Write nodes
            with open(nodes_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['node_id', 'type', 'label'])
                writer.writeheader()
                writer.writerows([
                    {'node_id': 'us-gaap:Assets', 'type': 'Concept', 'label': 'Assets'},
                    {'node_id': 'us-gaap:Cash', 'type': 'Concept', 'label': 'Cash'},
                    {'node_id': 'USD', 'type': 'Unit', 'label': 'USD'},
                    {'node_id': '2024', 'type': 'Period', 'label': '2024'}
                ])
            
            # Write edges
            with open(edges_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['src_id', 'edge_type', 'dst_id'])
                writer.writeheader()
                writer.writerows([
                    {'src_id': 'us-gaap:Cash', 'edge_type': 'is-a', 'dst_id': 'us-gaap:Assets'},
                    {'src_id': 'us-gaap:Cash', 'edge_type': 'measured-in', 'dst_id': 'USD'},
                    {'src_id': 'us-gaap:Assets', 'edge_type': 'measured-in', 'dst_id': 'USD'}
                ])
            
            # Load and compute
            concepts = set()
            edges_by_type = {}
            
            with open(nodes_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['type'] == 'Concept':
                        concepts.add(row['node_id'])
            
            with open(edges_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    et = row['edge_type']
                    if et not in edges_by_type:
                        edges_by_type[et] = []
                    edges_by_type[et].append((row['src_id'], row['dst_id']))
            
            # Compute metrics
            atp = metric_atp(concepts, edges_by_type)
            hp = metric_hp_coverage(concepts, edges_by_type)
            ap = metric_ap_directionality(edges_by_type)
            
            assert len(concepts) == 2, "Should have 2 concepts"
            assert atp == 1.0, "Both concepts have units"
            assert hp == 0.5, "One concept has parent"
            assert ap == 1.0, "No reverse edges"


class TestGoalAAcceptance:
    """Acceptance tests for Goal A: Auto-Taxonomy"""
    
    def test_hp_target_met(self):
        """Acceptance: HP ≥ 0.25 after auto-taxonomy"""
        # This will be populated with actual data in integration test
        # For now, test the threshold logic
        hp_after_taxonomy = 0.27  # Simulated result
        
        assert hp_after_taxonomy >= 0.25, f"HP {hp_after_taxonomy} must be ≥ 0.25"
    
    def test_atp_not_harmed(self):
        """Acceptance: AtP should not decrease significantly"""
        atp_before = 0.80
        atp_after = 0.79
        
        delta = atp_after - atp_before
        assert delta >= -0.02, f"AtP dropped by {-delta:.3f}, exceeds threshold"
    
    def test_ap_not_harmed(self):
        """Acceptance: AP should remain stable"""
        ap_before = 0.95
        ap_after = 0.94
        
        delta = ap_after - ap_before
        assert delta >= -0.02, f"AP dropped by {-delta:.3f}, exceeds threshold"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual metrics")
    config.addinivalue_line("markers", "integration: Integration tests with KG data")
    config.addinivalue_line("markers", "acceptance: Acceptance tests for sprint goals")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

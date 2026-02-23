"""
Tests for Knowledge Graph building from SEC EDGAR data.
"""
import pytest
from collections import defaultdict


class MockKnowledgeGraph:
    """Mock KG for testing."""
    
    def __init__(self):
        self.nodes = set()
        self.edges_by_type = defaultdict(list)
    
    def add_node(self, node_id: str, node_type: str):
        self.nodes.add((node_id, node_type))
    
    def add_edge(self, src: str, dst: str, edge_type: str):
        self.edges_by_type[edge_type].append((src, dst))
    
    def get_node_count(self, node_type: str = None) -> int:
        if node_type is None:
            return len(self.nodes)
        return sum(1 for _, ntype in self.nodes if ntype == node_type)
    
    def get_edge_count(self, edge_type: str = None) -> int:
        if edge_type is None:
            return sum(len(edges) for edges in self.edges_by_type.values())
        return len(self.edges_by_type.get(edge_type, []))


class TestKGConstruction:
    """Test KG construction from facts."""
    
    @pytest.fixture
    def empty_kg(self):
        return MockKnowledgeGraph()
    
    def test_empty_kg_initialization(self, empty_kg):
        assert empty_kg.get_node_count() == 0
        assert empty_kg.get_edge_count() == 0
    
    def test_add_concept_nodes(self, empty_kg):
        empty_kg.add_node('us-gaap:Revenue', 'Concept')
        empty_kg.add_node('us-gaap:Assets', 'Concept')
        assert empty_kg.get_node_count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

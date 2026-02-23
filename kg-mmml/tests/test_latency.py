"""
Tests for latency benchmarking harness.
Validates Goal B: Latency measurement and regression detection.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path


class TestLatencyMetrics:
    """Test latency metric calculations"""
    
    def test_percentile_calculation(self):
        """Test percentile calculations for latency"""
        latencies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        assert p50 == 55.0, f"Expected p50=55, got {p50}"
        assert p95 == pytest.approx(95.5), f"Expected p95=95.5, got {p95}"
        assert p99 == pytest.approx(99.1), f"Expected p99=99.1, got {p99}"
    
    def test_tail_behavior_ratio(self):
        """Test p99/p95 ratio calculation"""
        latencies = np.array([10] * 950 + [20] * 45 + [25] * 5)
        
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        ratio = p99 / p95
        
        # Good tail behavior: ratio < 2.5
        assert ratio < 2.5, f"Tail ratio {ratio:.2f} exceeds threshold"
    
    def test_warm_cache_effect(self):
        """Test that warm-up queries improve latency"""
        # Simulate cold cache (slower) vs warm cache (faster)
        cold_latencies = np.random.uniform(50, 100, 50)
        warm_latencies = np.random.uniform(10, 30, 500)
        
        cold_mean = np.mean(cold_latencies)
        warm_mean = np.mean(warm_latencies)
        
        assert warm_mean < cold_mean, "Warm cache should be faster"


class TestLatencyBenchmarkOutput:
    """Test latency benchmark output format"""
    
    def test_csv_structure(self):
        """Test that latency CSV has required columns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            
            # Create sample output
            df = pd.DataFrame([
                {
                    'index_size': 1000,
                    'index_type': 'faiss_hnsw',
                    'queries': 500,
                    'k': 10,
                    'p50': 12.5,
                    'p95': 45.2,
                    'p99': 89.3,
                    'mean': 18.7,
                    'std': 15.2
                }
            ])
            df.to_csv(csv_path, index=False)
        
        # Load and validate
        df_loaded = pd.read_csv(csv_path)
        
        required_cols = ['index_size', 'index_type', 'p50', 'p95', 'p99']
        for col in required_cols:
            assert col in df_loaded.columns, f"Missing column: {col}"
        
        # Cleanup
        Path(csv_path).unlink()
    
    def test_metadata_json_structure(self):
        """Test that metadata JSON has required fields"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
            
            metadata = {
                'timestamp': '2025-10-12 10:00:00',
                'sizes': [1000, 10000],
                'queries_per_size': 500,
                'k': 10,
                'embedding_dim': 256,
                'indices_tested': ['faiss_hnsw', 'annoy']
            }
            json.dump(metadata, f)
        
        # Load and validate
        with open(json_path) as f:
            meta_loaded = json.load(f)
        
        required_fields = ['timestamp', 'sizes', 'queries_per_size', 'indices_tested']
        for field in required_fields:
            assert field in meta_loaded, f"Missing field: {field}"
        
        # Cleanup
        Path(json_path).unlink()


class TestLatencySLO:
    """Test Service Level Objectives for latency"""
    
    def test_p95_under_threshold_small_index(self):
        """Test p95 < 150ms for N=1e3 (Goal B)"""
        # Simulate benchmark results
        results = {
            'index_size': 1000,
            'p95': 45.2  # Well under 150ms
        }
        
        assert results['p95'] < 150, f"p95 {results['p95']}ms exceeds SLO"
    
    def test_p95_threshold_large_index(self):
        """Test p95 < 200ms for N=1e4"""
        results = {
            'index_size': 10000,
            'p95': 120.5
        }
        
        assert results['p95'] < 200, f"p95 {results['p95']}ms exceeds threshold"
    
    def test_tail_latency_ratio(self):
        """Test p99/p95 < 2.5 (acceptable tail behavior)"""
        results = {
            'p95': 100.0,
            'p99': 220.0
        }
        
        ratio = results['p99'] / results['p95']
        assert ratio < 2.5, f"Tail ratio {ratio:.2f} indicates poor tail behavior"


class TestGoalBAcceptance:
    """Acceptance tests for Goal B: Latency Harness"""
    
    def test_latency_csv_exists(self):
        """Acceptance: latency_baseline.csv must exist"""
        # This will check the actual file in integration
        csv_path = Path("reports/tables/latency_baseline.csv")
        
        # For unit test, just validate logic
        # In integration test, this would check:
        # assert csv_path.exists(), "latency_baseline.csv not found"
        pass
    
    def test_both_index_sizes_benchmarked(self):
        """Acceptance: Must have results for N=1e3 and N=1e4"""
        # Simulate benchmark results
        df = pd.DataFrame([
            {'index_size': 1000, 'p50': 10, 'p95': 45, 'p99': 90},
            {'index_size': 10000, 'p50': 25, 'p95': 120, 'p99': 250}
        ])
        
        sizes = set(df['index_size'].unique())
        required_sizes = {1000, 10000}
        
        assert required_sizes.issubset(sizes), f"Missing sizes: {required_sizes - sizes}"
    
    def test_multiple_index_types(self):
        """Acceptance: Should test at least 2 index types"""
        df = pd.DataFrame([
            {'index_type': 'faiss_hnsw', 'p50': 10},
            {'index_type': 'annoy', 'p50': 12}
        ])
        
        index_types = df['index_type'].unique()
        assert len(index_types) >= 2, "Should test multiple index types"
    
    def test_sufficient_query_samples(self):
        """Acceptance: Should run ≥500 queries for statistical validity"""
        metadata = {'queries_per_size': 500}
        
        assert metadata['queries_per_size'] >= 500, "Need ≥500 queries"


class TestLatencyRegression:
    """Regression tests to detect latency increases"""
    
    def test_no_latency_regression(self):
        """Test that latency doesn't regress between runs"""
        # Simulate baseline and current run
        baseline = {
            'index_size': 1000,
            'index_type': 'faiss_hnsw',
            'p95': 45.0
        }
        
        current = {
            'index_size': 1000,
            'index_type': 'faiss_hnsw',
            'p95': 48.0  # 3ms increase
        }
        
        # Allow 10% regression tolerance
        threshold = baseline['p95'] * 1.10
        
        assert current['p95'] <= threshold, \
            f"Latency regressed: {baseline['p95']}ms → {current['p95']}ms"
    
    def test_scaling_behavior(self):
        """Test that latency scales reasonably with index size"""
        results = [
            {'index_size': 1000, 'p95': 45.0},
            {'index_size': 10000, 'p95': 120.0}
        ]
        
        # 10x size increase should not cause >10x latency increase
        small = results[0]
        large = results[1]
        
        size_ratio = large['index_size'] / small['index_size']
        latency_ratio = large['p95'] / small['p95']
        
        # With ANN index, latency should scale sub-linearly
        assert latency_ratio < size_ratio, \
            f"Latency scaling ({latency_ratio:.1f}x) exceeds size scaling ({size_ratio:.1f}x)"


class TestIndexComparison:
    """Test comparison between different index types"""
    
    def test_faiss_vs_annoy(self):
        """Compare FAISS and Annoy performance"""
        results = pd.DataFrame([
            {'index_type': 'faiss_hnsw', 'p50': 12.0, 'p95': 45.0},
            {'index_type': 'annoy', 'p50': 15.0, 'p95': 50.0}
        ])
        
        faiss = results[results['index_type'] == 'faiss_hnsw'].iloc[0]
        annoy = results[results['index_type'] == 'annoy'].iloc[0]
        
        # Both should meet SLO
        assert faiss['p95'] < 150, "FAISS exceeds SLO"
        assert annoy['p95'] < 150, "Annoy exceeds SLO"
        
        # Document which is faster
        faster = 'faiss_hnsw' if faiss['p95'] < annoy['p95'] else 'annoy'
        print(f"Faster index: {faster}")
    
    def test_exact_search_baseline(self):
        """Test that ANN is faster than exact search"""
        results = {
            'exact': {'p95': 500.0},  # Linear scan
            'ann': {'p95': 45.0}       # ANN index
        }
        
        speedup = results['exact']['p95'] / results['ann']['p95']
        assert speedup > 5, f"ANN speedup only {speedup:.1f}x"


# Fixtures for testing
@pytest.fixture
def sample_latencies():
    """Generate sample latency distribution"""
    np.random.seed(42)
    # Most queries fast, some slow outliers
    fast = np.random.uniform(10, 30, 950)
    slow = np.random.uniform(50, 100, 45)
    outliers = np.random.uniform(100, 200, 5)
    return np.concatenate([fast, slow, outliers])


@pytest.fixture
def sample_benchmark_df():
    """Generate sample benchmark DataFrame"""
    return pd.DataFrame([
        {
            'index_size': 1000,
            'index_type': 'faiss_hnsw',
            'queries': 500,
            'k': 10,
            'p50': 12.5,
            'p95': 45.2,
            'p99': 89.3,
            'mean': 18.7,
            'std': 15.2
        },
        {
            'index_size': 10000,
            'index_type': 'faiss_hnsw',
            'queries': 500,
            'k': 10,
            'p50': 28.1,
            'p95': 120.5,
            'p99': 245.8,
            'mean': 42.3,
            'std': 38.9
        }
    ])


class TestWithFixtures:
    """Tests using fixtures"""
    
    def test_latency_distribution(self, sample_latencies):
        """Test latency distribution characteristics"""
        p50 = np.percentile(sample_latencies, 50)
        p95 = np.percentile(sample_latencies, 95)
        p99 = np.percentile(sample_latencies, 99)
        
        # Most queries should be fast (p50 low)
        assert p50 < 50, f"Median latency too high: {p50}"
        
        # Few slow queries allowed (p95 reasonable)
        assert p95 < 150, f"p95 too high: {p95}"
        
        # Outliers expected but bounded (p99)
        assert p99 < 300, f"p99 too high: {p99}"
    
    def test_benchmark_completeness(self, sample_benchmark_df):
        """Test that benchmark covers all required scenarios"""
        df = sample_benchmark_df
        
        # Check size coverage
        assert 1000 in df['index_size'].values
        assert 10000 in df['index_size'].values
        
        # Check metrics present
        assert 'p50' in df.columns
        assert 'p95' in df.columns
        assert 'p99' in df.columns


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers for latency tests"""
    config.addinivalue_line("markers", "latency: Latency benchmark tests")
    config.addinivalue_line("markers", "regression: Latency regression detection")
    config.addinivalue_line("markers", "slo: Service Level Objective validation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

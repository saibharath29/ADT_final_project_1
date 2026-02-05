"""
Cost Model for Vector Search Operations

This module implements cost estimation for vector similarity search operations
using FAISS-style ANN indexes (HNSW, IVFFlat) integrated with PostgreSQL pgvector.

Key Innovation: Extend traditional database cost models to include vector operations.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorCostModel:
    """
    Cost model for vector similarity search operations.
    
    Based on empirical analysis of pgvector HNSW and IVFFlat performance.
    Cost units are abstract (normalized to PostgreSQL's cost model).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize cost model with configuration parameters.
        
        Args:
            config: Dictionary with cost parameters from config.yaml
        """
        self.distance_cost = config.get('vector_distance_cost', 1.0)
        self.comparison_cost = config.get('vector_comparison_cost', 0.1)
        self.cpu_operator_cost = config.get('cpu_operator_cost', 0.0025)
        
    def estimate_hnsw_search_cost(
        self, 
        n_vectors: int, 
        k: int, 
        m: int = 16, 
        ef_search: int = 40
    ) -> float:
        """
        Estimate cost of HNSW (Hierarchical Navigable Small World) search.
        
        HNSW Algorithm:
        1. Navigate through hierarchical graph layers
        2. At each layer, explore m nearest neighbors
        3. Final layer searches ef_search candidates
        4. Return top-k results
        
        Cost Formula:
            Cost = (log(n) * m * distance_cost)              # Graph traversal
                   + (ef_search * distance_cost)             # Final layer search
                   + (ef_search * log(k) * comparison_cost)  # Result ranking
        
        Args:
            n_vectors: Total number of vectors in index
            k: Number of nearest neighbors to return
            m: Number of connections per layer (HNSW parameter)
            ef_search: Size of dynamic candidate list
            
        Returns:
            Estimated cost in abstract units
        """
        if n_vectors <= 0:
            return 0.0
            
        # Cost of navigating through graph layers
        graph_traversal_cost = np.log2(max(n_vectors, 1)) * m * self.distance_cost
        
        # Cost of searching final layer
        final_layer_cost = ef_search * self.distance_cost
        
        # Cost of ranking and selecting top-k
        ranking_cost = ef_search * np.log2(max(k, 1)) * self.comparison_cost
        
        total_cost = graph_traversal_cost + final_layer_cost + ranking_cost
        
        logger.debug(f"HNSW cost: n={n_vectors}, k={k}, ef={ef_search} -> {total_cost:.2f}")
        
        return total_cost
    
    def estimate_ivfflat_search_cost(
        self, 
        n_vectors: int, 
        k: int, 
        n_lists: int = 100, 
        n_probes: int = 10
    ) -> float:
        """
        Estimate cost of IVFFlat (Inverted File with Flat storage) search.
        
        IVFFlat Algorithm:
        1. Vectors are partitioned into n_lists clusters
        2. Search probes n_probes nearest clusters
        3. Compute exact distances within those clusters
        4. Return top-k results
        
        Cost Formula:
            Cost = (n_probes * distance_cost)                    # Find nearest clusters
                   + (n_probes * avg_list_size * distance_cost)  # Scan cluster contents
                   + (n_probes * avg_list_size * log(k) * comparison_cost)  # Ranking
        
        Args:
            n_vectors: Total number of vectors in index
            k: Number of nearest neighbors to return
            n_lists: Number of inverted lists (clusters)
            n_probes: Number of lists to search
            
        Returns:
            Estimated cost in abstract units
        """
        if n_vectors <= 0 or n_lists <= 0:
            return 0.0
            
        # Average number of vectors per list
        avg_list_size = n_vectors / n_lists
        
        # Cost of finding nearest clusters (quantizer lookup)
        quantizer_cost = n_probes * self.distance_cost
        
        # Cost of scanning vectors in probed lists
        scan_cost = n_probes * avg_list_size * self.distance_cost
        
        # Cost of ranking results
        ranking_cost = n_probes * avg_list_size * np.log2(max(k, 1)) * self.comparison_cost
        
        total_cost = quantizer_cost + scan_cost + ranking_cost
        
        logger.debug(f"IVFFlat cost: n={n_vectors}, k={k}, lists={n_lists}, probes={n_probes} -> {total_cost:.2f}")
        
        return total_cost
    
    def estimate_sequential_vector_scan(self, n_vectors: int, k: int) -> float:
        """
        Estimate cost of sequential (brute-force) vector search.
        
        This is the baseline: compute all distances, then select top-k.
        Used when no index exists or for small datasets.
        
        Cost Formula:
            Cost = (n_vectors * distance_cost)              # All distance computations
                   + (n_vectors * log(k) * comparison_cost) # Heap-based top-k selection
        
        Args:
            n_vectors: Number of vectors to scan
            k: Number of top results to return
            
        Returns:
            Estimated cost in abstract units
        """
        if n_vectors <= 0:
            return 0.0
            
        distance_computations = n_vectors * self.distance_cost
        heap_operations = n_vectors * np.log2(max(k, 1)) * self.comparison_cost
        
        total_cost = distance_computations + heap_operations
        
        logger.debug(f"Sequential scan cost: n={n_vectors}, k={k} -> {total_cost:.2f}")
        
        return total_cost
    
    def estimate_filtered_vector_search(
        self, 
        n_vectors: int, 
        k: int, 
        filter_selectivity: float,
        index_type: str = 'hnsw',
        **index_params
    ) -> Tuple[float, int]:
        """
        Estimate cost when SQL filter is applied BEFORE vector search.
        
        Strategy: Filter first, then vector search on reduced set.
        
        Args:
            n_vectors: Total vectors in table
            k: Number of top results
            filter_selectivity: Fraction of rows passing filter (0.0 to 1.0)
            index_type: 'hnsw', 'ivfflat', or 'sequential'
            **index_params: Additional parameters for index
            
        Returns:
            Tuple of (total_cost, filtered_vector_count)
        """
        filtered_count = int(n_vectors * filter_selectivity)
        
        # Cost of applying SQL filter (handled by SQL cost model)
        # We only estimate the vector search part here
        
        if index_type == 'hnsw':
            vector_cost = self.estimate_hnsw_search_cost(filtered_count, k, **index_params)
        elif index_type == 'ivfflat':
            vector_cost = self.estimate_ivfflat_search_cost(filtered_count, k, **index_params)
        else:  # sequential
            vector_cost = self.estimate_sequential_vector_scan(filtered_count, k)
        
        logger.debug(f"Filter-first: {n_vectors} -> {filtered_count} vectors, cost={vector_cost:.2f}")
        
        return vector_cost, filtered_count
    
    def estimate_vector_then_filter(
        self, 
        n_vectors: int, 
        k: int, 
        filter_selectivity: float,
        index_type: str = 'hnsw',
        **index_params
    ) -> Tuple[float, int]:
        """
        Estimate cost when vector search is performed BEFORE SQL filter.
        
        Strategy: 
        1. Retrieve top-k' vectors (where k' > k to account for filtering)
        2. Apply SQL filter to those results
        3. Return top-k after filtering
        
        Challenge: Need to retrieve more than k vectors if filter is selective.
        
        Args:
            n_vectors: Total vectors in table
            k: Number of top results needed after filtering
            filter_selectivity: Fraction of rows passing filter
            index_type: 'hnsw', 'ivfflat', or 'sequential'
            **index_params: Additional parameters for index
            
        Returns:
            Tuple of (total_cost, candidate_vectors_retrieved)
        """
        # Estimate how many vectors to retrieve to get k after filtering
        # If selectivity is 0.1 and we need 10 results, retrieve ~100 candidates
        if filter_selectivity > 0:
            k_prime = min(int(k / filter_selectivity * 1.5), n_vectors)  # 1.5x safety margin
        else:
            k_prime = n_vectors  # Worst case: retrieve everything
        
        # Cost of vector search for k_prime results
        if index_type == 'hnsw':
            vector_cost = self.estimate_hnsw_search_cost(n_vectors, k_prime, **index_params)
        elif index_type == 'ivfflat':
            vector_cost = self.estimate_ivfflat_search_cost(n_vectors, k_prime, **index_params)
        else:
            vector_cost = self.estimate_sequential_vector_scan(n_vectors, k_prime)
        
        # Cost of filtering k_prime results (minimal - just tuple processing)
        filter_cost = k_prime * self.cpu_operator_cost
        
        total_cost = vector_cost + filter_cost
        
        logger.debug(f"Vector-first: retrieve {k_prime} candidates, cost={total_cost:.2f}")
        
        return total_cost, k_prime
    
    def recommend_strategy(
        self, 
        n_vectors: int, 
        k: int, 
        filter_selectivity: float,
        index_type: str = 'hnsw'
    ) -> str:
        """
        Recommend execution strategy based on cost estimates.
        
        Args:
            n_vectors: Total vectors
            k: Top-k to return
            filter_selectivity: Filter selectivity (0.0 to 1.0)
            index_type: Vector index type
            
        Returns:
            'filter_first' or 'vector_first'
        """
        filter_first_cost, _ = self.estimate_filtered_vector_search(
            n_vectors, k, filter_selectivity, index_type
        )
        
        vector_first_cost, _ = self.estimate_vector_then_filter(
            n_vectors, k, filter_selectivity, index_type
        )
        
        if filter_first_cost < vector_first_cost:
            logger.info(f"Recommend FILTER_FIRST: {filter_first_cost:.2f} < {vector_first_cost:.2f}")
            return 'filter_first'
        else:
            logger.info(f"Recommend VECTOR_FIRST: {vector_first_cost:.2f} < {filter_first_cost:.2f}")
            return 'vector_first'


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize cost model
    config = {
        'vector_distance_cost': 1.0,
        'vector_comparison_cost': 0.1,
        'cpu_operator_cost': 0.0025
    }
    
    cost_model = VectorCostModel(config)
    
    # Test scenario: 100K products, retrieve top-10
    n_vectors = 100000
    k = 10
    
    print("=" * 60)
    print("Cost Model Analysis")
    print("=" * 60)
    
    # HNSW search
    hnsw_cost = cost_model.estimate_hnsw_search_cost(n_vectors, k)
    print(f"\nHNSW Search (n={n_vectors:,}, k={k}): {hnsw_cost:.2f}")
    
    # IVFFlat search
    ivf_cost = cost_model.estimate_ivfflat_search_cost(n_vectors, k)
    print(f"IVFFlat Search (n={n_vectors:,}, k={k}): {ivf_cost:.2f}")
    
    # Sequential scan
    seq_cost = cost_model.estimate_sequential_vector_scan(n_vectors, k)
    print(f"Sequential Scan (n={n_vectors:,}, k={k}): {seq_cost:.2f}")
    
    # Filter scenarios
    print("\n" + "=" * 60)
    print("Filter Strategy Comparison")
    print("=" * 60)
    
    for selectivity in [0.01, 0.1, 0.5, 0.9]:
        print(f"\nFilter selectivity: {selectivity:.0%} ({int(n_vectors * selectivity):,} rows)")
        
        filter_first, _ = cost_model.estimate_filtered_vector_search(
            n_vectors, k, selectivity, 'hnsw'
        )
        
        vector_first, candidates = cost_model.estimate_vector_then_filter(
            n_vectors, k, selectivity, 'hnsw'
        )
        
        strategy = cost_model.recommend_strategy(n_vectors, k, selectivity, 'hnsw')
        
        print(f"  Filter-first cost: {filter_first:.2f}")
        print(f"  Vector-first cost: {vector_first:.2f} (retrieves {candidates} candidates)")
        print(f"  → Recommended: {strategy.upper()}")

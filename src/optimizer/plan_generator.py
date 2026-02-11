"""
Query Plan Generator

Generates alternative execution plans for hybrid SQL + Vector queries.
This is the core of our optimizer innovation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PlanType(Enum):
    """Execution plan types"""
    FILTER_FIRST = "filter_first"      # SQL filters → Vector search
    VECTOR_FIRST = "vector_first"      # Vector search → SQL filters
    HYBRID = "hybrid"                  # Interleaved execution
    SEQUENTIAL = "sequential"          # No optimization


@dataclass
class QueryPredicate:
    """Represents a SQL filter predicate"""
    column: str
    operator: str  # '=', '<', '>', '<=', '>=', 'IN', 'LIKE'
    value: Any
    predicate_type: str  # 'equality', 'range', 'like', 'in'
    selectivity: Optional[float] = None


@dataclass
class VectorOperation:
    """Represents a vector similarity search operation"""
    embedding_column: str
    query_vector: List[float]
    k: int  # top-k results
    distance_metric: str = "cosine"  # 'cosine', 'l2', 'inner_product'
    index_type: Optional[str] = None  # 'hnsw', 'ivfflat', None


@dataclass
class ExecutionPlan:
    """
    Represents a complete execution plan for a hybrid query.
    
    This is what the optimizer produces and the executor consumes.
    """
    plan_id: str
    plan_type: PlanType
    estimated_cost: float
    operations: List[Dict[str, Any]]  # Ordered list of operations
    metadata: Dict[str, Any]  # Additional plan information
    
    def __repr__(self):
        return f"Plan({self.plan_type.value}, cost={self.estimated_cost:.2f})"


class PlanGenerator:
    """
    Generates alternative execution plans for hybrid queries.
    
    Takes parsed query components and generates:
    1. Filter-first plans
    2. Vector-first plans
    3. Hybrid plans (optional)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize plan generator with configuration.
        
        Args:
            config: Optimizer configuration
        """
        self.enable_filter_first = config.get('enable_filter_first', True)
        self.enable_vector_first = config.get('enable_vector_first', True)
        self.enable_hybrid = config.get('enable_hybrid', False)  # Advanced
        self.plan_limit = config.get('plan_enumeration_limit', 10)
        
    def generate_plans(
        self,
        predicates: List[QueryPredicate],
        vector_op: VectorOperation,
        table_stats: Dict[str, Any]
    ) -> List[ExecutionPlan]:
        """
        Generate all viable execution plans for the query.
        
        Args:
            predicates: List of SQL filter predicates
            vector_op: Vector similarity search operation
            table_stats: Table statistics (row count, selectivities, etc.)
            
        Returns:
            List of ExecutionPlan objects
        """
        plans = []
        plan_id_counter = 0
        
        # Plan 1: Filter-first
        if self.enable_filter_first and predicates:
            plan = self._generate_filter_first_plan(
                f"plan_{plan_id_counter}",
                predicates,
                vector_op,
                table_stats
            )
            plans.append(plan)
            plan_id_counter += 1
            logger.info(f"Generated filter-first plan: {plan}")
        
        # Plan 2: Vector-first
        if self.enable_vector_first:
            plan = self._generate_vector_first_plan(
                f"plan_{plan_id_counter}",
                predicates,
                vector_op,
                table_stats
            )
            plans.append(plan)
            plan_id_counter += 1
            logger.info(f"Generated vector-first plan: {plan}")
        
        # Plan 3: Hybrid (advanced - optional)
        if self.enable_hybrid and predicates and len(predicates) > 1:
            hybrid_plans = self._generate_hybrid_plans(
                plan_id_counter,
                predicates,
                vector_op,
                table_stats
            )
            plans.extend(hybrid_plans)
            plan_id_counter += len(hybrid_plans)
        
        # Limit number of plans
        plans = plans[:self.plan_limit]
        
        logger.info(f"Generated {len(plans)} execution plans")
        return plans
    
    def _generate_filter_first_plan(
        self,
        plan_id: str,
        predicates: List[QueryPredicate],
        vector_op: VectorOperation,
        table_stats: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Generate a filter-first execution plan.
        
        Strategy:
        1. Apply all SQL filters first
        2. Perform vector search on filtered subset
        3. Return top-k results
        """
        operations = []
        
        # Step 1: Apply SQL filters
        operations.append({
            'type': 'filter',
            'predicates': predicates,
            'method': 'indexscan' if self._has_index(predicates, table_stats) else 'seqscan'
        })
        
        # Step 2: Vector search on filtered data
        operations.append({
            'type': 'vector_search',
            'embedding_column': vector_op.embedding_column,
            'k': vector_op.k,
            'distance_metric': vector_op.distance_metric,
            'index_type': vector_op.index_type or 'hnsw'
        })
        
        metadata = {
            'description': 'Filter predicates first, then vector search',
            'num_predicates': len(predicates),
            'expected_filter_output': self._estimate_filtered_rows(predicates, table_stats)
        }
        
        return ExecutionPlan(
            plan_id=plan_id,
            plan_type=PlanType.FILTER_FIRST,
            estimated_cost=0.0,  # Will be filled by cost model
            operations=operations,
            metadata=metadata
        )
    
    def _generate_vector_first_plan(
        self,
        plan_id: str,
        predicates: List[QueryPredicate],
        vector_op: VectorOperation,
        table_stats: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Generate a vector-first execution plan.
        
        Strategy:
        1. Perform vector search (retrieve top-k' candidates)
        2. Apply SQL filters to candidates
        3. Return top-k after filtering
        """
        operations = []
        
        # Step 1: Vector search (retrieve extra candidates if filters exist)
        if predicates:
            # Need to retrieve more to account for filtering
            combined_selectivity = self._estimate_combined_selectivity(predicates)
            k_prime = min(
                int(vector_op.k / max(combined_selectivity, 0.01) * 1.5),
                table_stats.get('n_tuples', 100000)
            )
        else:
            k_prime = vector_op.k
        
        operations.append({
            'type': 'vector_search',
            'embedding_column': vector_op.embedding_column,
            'k': k_prime,
            'distance_metric': vector_op.distance_metric,
            'index_type': vector_op.index_type or 'hnsw'
        })
        
        # Step 2: Apply filters to vector search results
        if predicates:
            operations.append({
                'type': 'filter',
                'predicates': predicates,
                'method': 'sequential'  # Small result set, just scan
            })
        
        # Step 3: Take top-k after filtering
        operations.append({
            'type': 'limit',
            'k': vector_op.k
        })
        
        metadata = {
            'description': 'Vector search first, then filter results',
            'num_predicates': len(predicates),
            'candidate_vectors': k_prime
        }
        
        return ExecutionPlan(
            plan_id=plan_id,
            plan_type=PlanType.VECTOR_FIRST,
            estimated_cost=0.0,  # Will be filled by cost model
            operations=operations,
            metadata=metadata
        )
    
    def _generate_hybrid_plans(
        self,
        start_id: int,
        predicates: List[QueryPredicate],
        vector_op: VectorOperation,
        table_stats: Dict[str, Any]
    ) -> List[ExecutionPlan]:
        """
        Generate hybrid execution plans (advanced).
        
        Strategy: Apply highly selective filters first, then vector search,
        then remaining filters.
        
        This is an optimization for queries with multiple predicates of
        varying selectivity.
        """
        plans = []
        
        # Sort predicates by selectivity (most selective first)
        sorted_predicates = sorted(
            predicates,
            key=lambda p: p.selectivity if p.selectivity else 0.5
        )
        
        # Try splitting at different points
        for split_idx in range(1, len(sorted_predicates)):
            early_filters = sorted_predicates[:split_idx]
            late_filters = sorted_predicates[split_idx:]
            
            operations = [
                {
                    'type': 'filter',
                    'predicates': early_filters,
                    'method': 'indexscan'
                },
                {
                    'type': 'vector_search',
                    'embedding_column': vector_op.embedding_column,
                    'k': vector_op.k,
                    'distance_metric': vector_op.distance_metric,
                    'index_type': vector_op.index_type or 'hnsw'
                },
                {
                    'type': 'filter',
                    'predicates': late_filters,
                    'method': 'sequential'
                }
            ]
            
            metadata = {
                'description': f'Hybrid: {len(early_filters)} early filters → vector → {len(late_filters)} late filters',
                'split_point': split_idx
            }
            
            plan = ExecutionPlan(
                plan_id=f"plan_{start_id + len(plans)}",
                plan_type=PlanType.HYBRID,
                estimated_cost=0.0,
                operations=operations,
                metadata=metadata
            )
            
            plans.append(plan)
        
        return plans
    
    def _has_index(self, predicates: List[QueryPredicate], table_stats: Dict) -> bool:
        """Check if any predicate can use an index"""
        indexed_columns = table_stats.get('indexed_columns', set())
        return any(p.column in indexed_columns for p in predicates)
    
    def _estimate_filtered_rows(self, predicates: List[QueryPredicate], table_stats: Dict) -> int:
        """Estimate number of rows after applying all filters"""
        n_tuples = table_stats.get('n_tuples', 100000)
        combined_selectivity = self._estimate_combined_selectivity(predicates)
        return int(n_tuples * combined_selectivity)
    
    def _estimate_combined_selectivity(self, predicates: List[QueryPredicate]) -> float:
        """Estimate combined selectivity of multiple predicates"""
        if not predicates:
            return 1.0
        
        combined = 1.0
        for pred in predicates:
            if pred.selectivity:
                combined *= pred.selectivity
            else:
                combined *= 0.1  # Default conservative estimate
        
        return combined


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example query: Find Apple products under $1000, semantically similar to "laptop"
    
    # Define predicates
    predicates = [
        QueryPredicate(
            column='brand',
            operator='=',
            value='Apple',
            predicate_type='equality',
            selectivity=0.02  # 2% of products are Apple
        ),
        QueryPredicate(
            column='price',
            operator='<',
            value=1000,
            predicate_type='range',
            selectivity=0.3  # 30% of products under $1000
        )
    ]
    
    # Define vector operation
    vector_op = VectorOperation(
        embedding_column='embedding',
        query_vector=[0.1] * 768,  # Dummy embedding
        k=10,
        distance_metric='cosine',
        index_type='hnsw'
    )
    
    # Table statistics
    table_stats = {
        'n_tuples': 100000,
        'indexed_columns': {'brand', 'price'},
        'avg_tuple_size': 200
    }
    
    # Generate plans
    config = {
        'enable_filter_first': True,
        'enable_vector_first': True,
        'enable_hybrid': True,
        'plan_enumeration_limit': 10
    }
    
    generator = PlanGenerator(config)
    plans = generator.generate_plans(predicates, vector_op, table_stats)
    
    print("=" * 60)
    print("Generated Execution Plans")
    print("=" * 60)
    
    for plan in plans:
        print(f"\n{plan.plan_id}: {plan.plan_type.value}")
        print(f"Description: {plan.metadata['description']}")
        print("Operations:")
        for i, op in enumerate(plan.operations, 1):
            print(f"  {i}. {op['type']}")

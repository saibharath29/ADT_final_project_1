"""
Load Generated Data into PostgreSQL

This script loads the sample data with embeddings into the database.
"""

import sys
import os
import pandas as pd
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from executor.pg_connector import PostgreSQLConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_products_data(db: PostgreSQLConnector, data_path: str):
    """Load products dataset into database"""
    
    logger.info(f"Loading data from {data_path}")
    
    # Load data with embeddings
    if data_path.endswith('.pkl'):
        df = pd.read_pickle(data_path)
    else:
        logger.error("Please generate data first: python scripts/generate_data.py")
        return
    
    logger.info(f"Loaded {len(df)} products")
    
    # Convert DataFrame to list of dicts
    data = df.to_dict('records')
    
    # Insert into database
    db.insert_vectors(
        table='products',
        data=data,
        embedding_column='embedding'
    )
    
    logger.info("✓ Data loaded successfully")


def verify_data(db: PostgreSQLConnector):
    """Verify data was loaded correctly"""
    
    # Check row count
    result = db.execute_query("SELECT COUNT(*) FROM products")
    count = result[0][0]
    logger.info(f"Products table: {count:,} rows")
    
    # Check for embeddings
    result = db.execute_query(
        "SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL"
    )
    embedding_count = result[0][0]
    logger.info(f"Rows with embeddings: {embedding_count:,}")
    
    # Sample data
    result = db.execute_query("SELECT name, brand, price FROM products LIMIT 5")
    logger.info("\nSample products:")
    for name, brand, price in result:
        logger.info(f"  • {name} ({brand}) - ${price}")
    
    # Check indexes
    result = db.execute_query("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'products'
    """)
    logger.info(f"\nIndexes: {len(result)} total")
    for idx_name, _ in result:
        logger.info(f"  • {idx_name}")


def main():
    """Main execution"""
    
    print("=" * 60)
    print("Load Data into PostgreSQL")
    print("=" * 60)
    
    # Check if data exists
    data_path = 'data/datasets/products.pkl'
    if not os.path.exists(data_path):
        print("\n❌ Data file not found!")
        print("Please generate data first:")
        print("  python scripts/generate_data.py")
        return
    
    try:
        # Connect to database
        db = PostgreSQLConnector('../config.yaml')
        db.connect()
        
        # Load data
        load_products_data(db, data_path)
        
        # Verify
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        verify_data(db)
        
        # Collect statistics
        print("\n" + "=" * 60)
        print("Table Statistics")
        print("=" * 60)
        stats = db.get_table_statistics('products')
        for key, value in stats.items():
            if isinstance(value, set):
                value = ', '.join(value)
            print(f"  {key}: {value}")
        
        db.disconnect()
        
        print("\n✅ Data loading complete!")
        print("\nNext steps:")
        print("  • Run benchmarks: python benchmarks/run_experiments.py")
        print("  • Open notebook: jupyter notebook notebooks/01_demo.ipynb")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        print("\n❌ Error loading data")
        print("\nMake sure PostgreSQL is running and configured correctly:")
        print("  1. Install PostgreSQL + pgvector")
        print("  2. Run: psql -U postgres -f sql/setup.sql")
        print("  3. Update config.yaml with your database credentials")


if __name__ == "__main__":
    main()

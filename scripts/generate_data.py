"""
Sample Data Generator

Generates realistic datasets with embeddings for benchmarking.
Creates product catalog data suitable for e-commerce search.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate sample datasets with embeddings"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize data generator with embedding model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def generate_products(self, n_products: int = 10000) -> pd.DataFrame:
        """
        Generate e-commerce product dataset.
        
        Args:
            n_products: Number of products to generate
            
        Returns:
            DataFrame with product data and embeddings
        """
        logger.info(f"Generating {n_products} products...")
        
        # Product categories and brands
        brands = ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Lenovo', 
                  'Microsoft', 'Google', 'Amazon', 'Bose', 'Canon', 'Nikon']
        
        categories = ['Laptops', 'Smartphones', 'Tablets', 'Headphones', 
                      'Cameras', 'Monitors', 'Keyboards', 'Mice', 'Speakers']
        
        products = []
        
        for i in range(n_products):
            brand = np.random.choice(brands)
            category = np.random.choice(categories)
            
            # Generate realistic product data
            product = {
                'name': f"{brand} {category[:-1]} {np.random.randint(1000, 9999)}",
                'brand': brand,
                'category': category,
                'price': round(np.random.lognormal(6, 1), 2),  # Lognormal distribution
                'rating': round(np.random.beta(8, 2) * 5, 2),   # Skewed towards high ratings
                'num_reviews': np.random.randint(0, 5000),
                'in_stock': np.random.random() > 0.1,  # 90% in stock
            }
            
            # Generate description
            adjectives = ['premium', 'high-quality', 'professional', 'compact', 
                          'powerful', 'efficient', 'innovative', 'sleek']
            features = ['performance', 'design', 'battery life', 'display', 
                        'connectivity', 'durability', 'portability']
            
            product['description'] = (
                f"{np.random.choice(adjectives).capitalize()} {product['name']} "
                f"featuring excellent {np.random.choice(features)} and "
                f"{np.random.choice(features)}. Perfect for "
                f"{'professionals' if product['price'] > 1000 else 'everyday use'}."
            )
            
            products.append(product)
        
        df = pd.DataFrame(products)
        
        # Generate embeddings for descriptions
        logger.info("Generating embeddings...")
        descriptions = df['description'].tolist()
        
        # Batch encode for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        df['embedding'] = embeddings
        
        logger.info(f"Generated {len(df)} products with {self.embedding_dim}-dim embeddings")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """
        Save dataset to CSV and embeddings separately.
        
        Args:
            df: DataFrame with data and embeddings
            output_path: Base path for output files
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data (without embeddings as CSV)
        df_csv = df.drop('embedding', axis=1)
        csv_path = output_path.replace('.pkl', '.csv')
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"Saved data to {csv_path}")
        
        # Save full dataset with embeddings as pickle
        df.to_pickle(output_path)
        logger.info(f"Saved full dataset with embeddings to {output_path}")


def main():
    """Generate sample datasets"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = DataGenerator()
    
    # Generate products dataset
    products_df = generator.generate_products(n_products=5000)  # Start with 5K for testing
    
    # Save dataset
    generator.save_dataset(
        products_df,
        'data/datasets/products.pkl'
    )
    
    # Display sample
    print("\n" + "=" * 60)
    print("Sample Products")
    print("=" * 60)
    print(products_df.head())
    print(f"\nDataset shape: {products_df.shape}")
    print(f"Embedding dimension: {generator.embedding_dim}")


if __name__ == "__main__":
    main()

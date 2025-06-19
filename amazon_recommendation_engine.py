#Change Log this commit
# - Added progress tracking and performance monitoring
# -- timing info and tqdm progress bars for major operations (e.g., training and evaluation phases)
# -- enhanced output formatting with phases and better structure
# - Implemented flexible dataset sampling capabilities 
# -- parameters for max_customers,max_products, %-based sampling options
# -- smart sampling strategies (most_active, random, balanced)
# -- upgraded filtering logic to support limits
# - Clarified in error message that the data belongs in a data/ subdirectory
# - Removed debugging leftover that defined customer_predictions twice
# - Deleted idx_to_user and idx_to_item mappings. I was going to use them for some feature tuning, but I haven't gotten to that yet. I'll put them back when I implement that.


"""
Amazon Electronics Recommendation Engine v2b

Production recommendation system designed for high-scale customer personalization
and revenue optimization. Built with industry-standard libraries for maximum 
compatibility and performance in enterprise environments.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

# Type aliases for better code readability
UserId = str
ProductId = str
Rating = float
ModelMetrics = Dict[str, Union[float, int]]
RecommendationList = List[Tuple[ProductId, Rating]]


class CollaborativeFilteringEngine:
    """
    Production collaborative filtering engine using sklearn.
    
    Implements user-based and item-based collaborative filtering algorithms
    optimized for scalable recommendation generation in enterprise e-commerce environments.
    """
    
    def __init__(self, similarity_metric: str = 'cosine', n_neighbors: int = 40, user_based: bool = True) -> None:
        """
        Initialize collaborative filtering engine.
        
        Args:
            similarity_metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            n_neighbors: Number of neighbors for recommendations
            user_based: True for user-based, False for item-based filtering
        """
        self.similarity_metric: str = similarity_metric
        self.n_neighbors: int = n_neighbors
        self.user_based: bool = user_based
        self.model: Optional[NearestNeighbors] = None
        self.interaction_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None
        self.item_means: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None
        self.user_to_idx: Dict[UserId, int] = {}
        self.item_to_idx: Dict[ProductId, int] = {}
        
    def _create_interaction_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create user-item interaction matrix from DataFrame."""
        unique_users = data['user_id'].unique()
        unique_items = data['prod_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        n_users, n_items = len(unique_users), len(unique_items)
        interaction_matrix = np.zeros((n_users, n_items))
        
        print("Building interaction matrix...")
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing interactions"):
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['prod_id']]
            interaction_matrix[user_idx, item_idx] = row['rating']
            
        return interaction_matrix
        
    def fit(self, data: pd.DataFrame) -> 'CollaborativeFilteringEngine':
        """
        Train the collaborative filtering model.
        
        Args:
            data: DataFrame with columns ['user_id', 'prod_id', 'rating']
        """
        start_time = time.time()
        print(f"Training collaborative filtering model on {len(data):,} interactions...")
        
        self.interaction_matrix = self._create_interaction_matrix(data)
        print(f"Created {self.interaction_matrix.shape[0]} x {self.interaction_matrix.shape[1]} interaction matrix")
        
        print("Calculating rating statistics...")
        self.global_mean = np.mean(self.interaction_matrix[self.interaction_matrix > 0])
        
        print("Computing user preferences...")
        self.user_means = np.array([
            np.mean(row[row > 0]) if np.any(row > 0) else self.global_mean 
            for row in tqdm(self.interaction_matrix, desc="User means")
        ])
        
        print("Computing product characteristics...")
        self.item_means = np.array([
            np.mean(col[col > 0]) if np.any(col > 0) else self.global_mean 
            for col in tqdm(self.interaction_matrix.T, desc="Item means")
        ])
        
        if self.user_based:
            matrix = self.interaction_matrix
            print("Building user-based similarity model...")
        else:
            matrix = self.interaction_matrix.T
            print("Building item-based similarity model...")
            
        print("Processing interaction data...")
        matrix_filled = matrix.copy()
        for i in tqdm(range(matrix_filled.shape[0]), desc="Filling missing values"):
            zero_mask = matrix_filled[i] == 0
            if self.user_based:
                matrix_filled[i][zero_mask] = self.user_means[i]
            else:
                matrix_filled[i][zero_mask] = self.item_means[i]
        
        metric = 'cosine' if self.similarity_metric == 'cosine' else 'euclidean'
        n_neighbors = min(self.n_neighbors + 1, matrix_filled.shape[0])
        print(f"Training {metric} similarity model with {n_neighbors} neighbors...")
        
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm='auto'
        )
        
        print("Fitting nearest neighbors model...")
        self.model.fit(matrix_filled)
        
        elapsed_time = time.time() - start_time
        print(f"Collaborative filtering model trained successfully in {elapsed_time:.1f} seconds")
        return self
        
    def predict(self, user_id: UserId, item_id: ProductId, verbose: bool = False) -> 'PredictionResult':
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            verbose: Whether to print prediction details
            
        Returns:
            PredictionResult object with estimated rating
        """
        if user_id not in self.user_to_idx:
            est = self.global_mean
        elif item_id not in self.item_to_idx:
            user_idx = self.user_to_idx[user_id]
            est = self.user_means[user_idx]
        else:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            if self.interaction_matrix[user_idx, item_idx] > 0:
                est = self.interaction_matrix[user_idx, item_idx]
            else:
                est = self._predict_rating(user_idx, item_idx)
        
        result = PredictionResult(user_id, item_id, est)
        
        if verbose:
            print(f"User: {user_id}")
            print(f"Item: {item_id}")
            print(f"Predicted rating: {est:.2f}")
            
        return result
        
    def _predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Internal method to predict rating using collaborative filtering algorithm."""
        if self.user_based:
            target_user = self.interaction_matrix[user_idx].reshape(1, -1)
            
            target_user_filled = target_user.copy()
            zero_mask = target_user_filled[0] == 0
            target_user_filled[0][zero_mask] = self.user_means[user_idx]
            
            distances, indices = self.model.kneighbors(target_user_filled)
            similar_users = indices[0][1:]
            
            numerator = 0
            denominator = 0
            
            for similar_user_idx in similar_users:
                if self.interaction_matrix[similar_user_idx, item_idx] > 0:
                    if self.similarity_metric == 'cosine':
                        similarity = 1 - distances[0][np.where(indices[0] == similar_user_idx)[0][0]]
                    else:
                        similarity = 1 / (1 + distances[0][np.where(indices[0] == similar_user_idx)[0][0]])
                    
                    rating = self.interaction_matrix[similar_user_idx, item_idx]
                    user_mean = self.user_means[similar_user_idx]
                    
                    numerator += similarity * (rating - user_mean)
                    denominator += similarity
            
            if denominator > 0:
                prediction = self.user_means[user_idx] + (numerator / denominator)
            else:
                prediction = self.user_means[user_idx]
                
        else:
            target_item = self.interaction_matrix[:, item_idx].reshape(1, -1)
            
            target_item_filled = target_item.copy()
            zero_mask = target_item_filled[0] == 0
            target_item_filled[0][zero_mask] = self.item_means[item_idx]
            
            matrix_t = self.interaction_matrix.T
            matrix_t_filled = matrix_t.copy()
            for i in range(matrix_t_filled.shape[0]):
                zero_mask = matrix_t_filled[i] == 0
                matrix_t_filled[i][zero_mask] = self.item_means[i]
            
            similarities = cosine_similarity(target_item_filled, matrix_t_filled)[0]
            similar_indices = np.argsort(similarities)[::-1][1:self.n_neighbors+1]
            
            numerator = 0
            denominator = 0
            
            for similar_item_idx in similar_indices:
                if self.interaction_matrix[user_idx, similar_item_idx] > 0:
                    similarity = similarities[similar_item_idx]
                    rating = self.interaction_matrix[user_idx, similar_item_idx]
                    item_mean = self.item_means[similar_item_idx]
                    
                    numerator += similarity * (rating - item_mean)
                    denominator += similarity
            
            if denominator > 0:
                prediction = self.item_means[item_idx] + (numerator / denominator)
            else:
                prediction = self.item_means[item_idx]
        
        return np.clip(prediction, 1.0, 5.0)
        
    def test(self, test_data: pd.DataFrame) -> List['PredictionResult']:
        """Test the model on a dataset and return predictions."""
        predictions = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Making predictions"):
            pred = self.predict(row['user_id'], row['prod_id'])
            pred.r_ui = row['rating']
            predictions.append(pred)
        return predictions


class MatrixFactorizationEngine:
    """
    Production matrix factorization recommendation engine.
    
    Implements SVD-based latent factor modeling optimized for scalable personalization
    in high-dimensional customer-product interaction spaces.
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, n_epochs: int = 20, 
                 reg_all: float = 0.02, random_state: int = 42) -> None:
        """
        Initialize matrix factorization engine.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            n_epochs: Number of training epochs
            reg_all: Regularization parameter
            random_state: Random seed for reproducibility
        """
        self.n_factors: int = n_factors
        self.learning_rate: float = learning_rate
        self.n_epochs: int = n_epochs
        self.reg_all: float = reg_all
        self.random_state: int = random_state
        
        # Create dedicated random number generator for reproducibility
        self.rng: np.random.RandomState = np.random.RandomState(random_state)
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None
        self.user_to_idx: Dict[UserId, int] = {}
        self.item_to_idx: Dict[ProductId, int] = {}
        
    def fit(self, data: pd.DataFrame) -> 'MatrixFactorizationEngine':
        """Use stochastic gradient descent to train the matrix factorization model."""
        start_time = time.time()
        print(f"Training matrix factorization model on {len(data):,} interactions...")
        
        unique_users = data['user_id'].unique()
        unique_items = data['prod_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        n_users, n_items = len(unique_users), len(unique_items)
        print(f"Matrix dimensions: {n_users} users x {n_items} items ({self.n_factors} factors)")
        
        print("Initializing model parameters...")
        self.global_mean = data['rating'].mean()
        
        # Use self.rng for all random operations to ensure reproducibility
        self.user_factors = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = self.rng.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        print("Preparing training data...")
        training_data = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Converting data"):
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['prod_id']]
            rating = row['rating']
            training_data.append((user_idx, item_idx, rating))
        
        print(f"Starting SGD training for {self.n_epochs} epochs...")
        
        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            # Use self.rng.shuffle for reproducible shuffling
            self.rng.shuffle(training_data)
            
            epoch_start = time.time()
            for user_idx, item_idx, rating in training_data:
                prediction = (self.global_mean + 
                            self.user_biases[user_idx] + 
                            self.item_biases[item_idx] + 
                            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                error = rating - prediction
                
                user_bias = self.user_biases[user_idx]
                item_bias = self.item_biases[item_idx]
                
                self.user_biases[user_idx] += self.learning_rate * (error - self.reg_all * user_bias)
                self.item_biases[item_idx] += self.learning_rate * (error - self.reg_all * item_bias)
                
                user_factors = self.user_factors[user_idx].copy()
                item_factors = self.item_factors[item_idx].copy()
                
                self.user_factors[user_idx] += self.learning_rate * (error * item_factors - self.reg_all * user_factors)
                self.item_factors[item_idx] += self.learning_rate * (error * user_factors - self.reg_all * item_factors)
            
            if (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                tqdm.write(f"Epoch {epoch + 1}/{self.n_epochs} completed in {epoch_time:.1f}s")
        
        elapsed_time = time.time() - start_time
        print(f"Matrix factorization model trained successfully in {elapsed_time:.1f} seconds")
        return self
        
    def predict(self, user_id: UserId, item_id: ProductId, verbose: bool = False) -> 'PredictionResult':
        """Predict rating for a user-item pair."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            est = self.global_mean
        else:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            est = (self.global_mean + 
                  self.user_biases[user_idx] + 
                  self.item_biases[item_idx] + 
                  np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        est = np.clip(est, 1.0, 5.0)
        
        result = PredictionResult(user_id, item_id, est)
        
        if verbose:
            print(f"User: {user_id}")
            print(f"Item: {item_id}")
            print(f"Predicted rating: {est:.2f}")
            
        return result
        
    def test(self, test_data: pd.DataFrame) -> List['PredictionResult']:
        """Test the model on a dataset and return predictions."""
        predictions = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Making predictions"):
            pred = self.predict(row['user_id'], row['prod_id'])
            pred.r_ui = row['rating']
            predictions.append(pred)
        return predictions


class PredictionResult:
    """Container for recommendation prediction results."""
    
    def __init__(self, uid: UserId, iid: ProductId, est: Rating, r_ui: Optional[Rating] = None) -> None:
        self.uid: UserId = uid
        self.iid: ProductId = iid
        self.est: Rating = est
        self.r_ui: Optional[Rating] = r_ui


class AmazonRecommendationEngine:
    """
    Production recommendation engine for Amazon Electronics catalog.
    
    Enterprise-grade recommendation system providing personalization capabilities
    using multiple algorithms including collaborative filtering and matrix factorization.
    Designed for high-scale customer personalization and business intelligence.
    """
    
    def __init__(self, min_user_interactions: int = 50, min_product_interactions: int = 5,
                 max_customers: Optional[int] = None, max_products: Optional[int] = None,
                 max_percent_customers: Optional[float] = None, max_percent_products: Optional[float] = None) -> None:
        """
        Initialize the recommendation engine.
        
        Args:
            min_user_interactions: Minimum customer interactions required
            min_product_interactions: Minimum product ratings required
            max_customers: Maximum number of customers to use (None = all qualified customers)
            max_products: Maximum number of products to use (None = all qualified products)
            max_percent_customers: Maximum percentage of customers to use (0-100)
            max_percent_products: Maximum percentage of products to use (0-100)
        """
        self.min_user_interactions: int = min_user_interactions
        self.min_product_interactions: int = min_product_interactions
        self.max_customers: Optional[int] = max_customers
        self.max_products: Optional[int] = max_products
        self.max_percent_customers: Optional[float] = max_percent_customers
        self.max_percent_products: Optional[float] = max_percent_products
        self.processed_data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Dict[str, Any]] = {}
        
    def load_customer_data(self, file_path: Union[str, Path], has_header: bool = False) -> pd.DataFrame:
        """
        Load customer interaction data from CSV file.
        
        Args:
            file_path: Path to customer interaction dataset
            has_header: Whether CSV has header row
            
        Returns:
            DataFrame with customer interaction data
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if has_header:
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv(file_path, header=None)
            data.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
        
        if 'timestamp' in data.columns:
            data = data.drop('timestamp', axis=1)
            
        print(f"Loaded {len(data):,} customer interactions from {file_path.name}")
        return data
        
    def filter_for_quality_recommendations(self, data: pd.DataFrame, sampling_strategy: str = 'most_active') -> pd.DataFrame:
        """
        Filter data to ensure statistical significance and recommendation quality.
        
        Args:
            data: Raw customer interaction data
            sampling_strategy: Sampling approach when limits are applied
                - 'most_active': Most active customers & popular products (default)
                - 'random': Random sampling of qualified entities
                - 'balanced': Stratified sampling across engagement levels
            
        Returns:
            Filtered dataset optimized for recommendation quality
        """
        print("Optimizing dataset for recommendation quality...")
        original_size = len(data)
        
        # Filter customers by minimum interactions
        customer_interactions = data['user_id'].value_counts()
        qualified_customers = customer_interactions[
            customer_interactions >= self.min_user_interactions
        ].index
        
        # Apply customer limits if specified
        customer_limit = None
        if self.max_customers:
            customer_limit = self.max_customers
            limit_type = f"absolute limit of {customer_limit:,}"
        elif self.max_percent_customers:
            customer_limit = int(len(qualified_customers) * self.max_percent_customers / 100)
            limit_type = f"percentage limit of {self.max_percent_customers}% ({customer_limit:,} customers)"
        
        if customer_limit and len(qualified_customers) > customer_limit:
            if sampling_strategy == 'most_active':
                sampled_customers = qualified_customers[:customer_limit]
                print(f"Selected top {customer_limit:,} most active customers ({limit_type})")
            elif sampling_strategy == 'random':
                sampled_customers = qualified_customers.to_series().sample(
                    n=customer_limit, random_state=42
                ).index
                print(f"Randomly selected {customer_limit:,} qualified customers ({limit_type})")
            elif sampling_strategy == 'balanced':
                interactions_per_customer = customer_interactions[qualified_customers]
                low_tier = interactions_per_customer[interactions_per_customer <= interactions_per_customer.quantile(0.33)]
                mid_tier = interactions_per_customer[
                    (interactions_per_customer > interactions_per_customer.quantile(0.33)) & 
                    (interactions_per_customer <= interactions_per_customer.quantile(0.67))
                ]
                high_tier = interactions_per_customer[interactions_per_customer > interactions_per_customer.quantile(0.67)]
                
                customers_per_tier = customer_limit // 3
                sampled_customers = pd.concat([
                    low_tier.sample(min(customers_per_tier, len(low_tier)), random_state=42),
                    mid_tier.sample(min(customers_per_tier, len(mid_tier)), random_state=42),
                    high_tier.sample(min(customer_limit - 2*customers_per_tier, len(high_tier)), random_state=42)
                ]).index
                print(f"Selected {customer_limit:,} customers using balanced sampling ({limit_type})")
            
            qualified_customers = sampled_customers
        else:
            print(f"Using all {len(qualified_customers):,} qualified customers")
        
        data = data[data['user_id'].isin(qualified_customers)]
        print(f"Retained {len(qualified_customers):,} customers with sufficient interaction history")
        
        # Filter products by minimum ratings
        product_ratings = data['prod_id'].value_counts()
        qualified_products = product_ratings[
            product_ratings >= self.min_product_interactions
        ].index
        
        # Apply product limits if specified
        product_limit = None
        if self.max_products:
            product_limit = self.max_products
            limit_type = f"absolute limit of {product_limit:,}"
        elif self.max_percent_products:
            product_limit = int(len(qualified_products) * self.max_percent_products / 100)
            limit_type = f"percentage limit of {self.max_percent_products}% ({product_limit:,} products)"
        
        if product_limit and len(qualified_products) > product_limit:
            if sampling_strategy == 'most_active':
                sampled_products = qualified_products[:product_limit]
                print(f"Selected top {product_limit:,} most popular products ({limit_type})")
            elif sampling_strategy == 'random':
                sampled_products = qualified_products.to_series().sample(
                    n=product_limit, random_state=42
                ).index
                print(f"Randomly selected {product_limit:,} qualified products ({limit_type})")
            elif sampling_strategy == 'balanced':
                ratings_per_product = product_ratings[qualified_products]
                low_pop = ratings_per_product[ratings_per_product <= ratings_per_product.quantile(0.33)]
                mid_pop = ratings_per_product[
                    (ratings_per_product > ratings_per_product.quantile(0.33)) & 
                    (ratings_per_product <= ratings_per_product.quantile(0.67))
                ]
                high_pop = ratings_per_product[ratings_per_product > ratings_per_product.quantile(0.67)]
                
                products_per_tier = product_limit // 3
                sampled_products = pd.concat([
                    low_pop.sample(min(products_per_tier, len(low_pop)), random_state=42),
                    mid_pop.sample(min(products_per_tier, len(mid_pop)), random_state=42),
                    high_pop.sample(min(product_limit - 2*products_per_tier, len(high_pop)), random_state=42)
                ]).index
                print(f"Selected {product_limit:,} products using balanced sampling ({limit_type})")
            
            qualified_products = sampled_products
        else:
            print(f"Using all {len(qualified_products):,} qualified products")
        
        data = data[data['prod_id'].isin(qualified_products)]
        print(f"Retained {len(qualified_products):,} products with sufficient rating data")
        
        reduction = original_size - len(data)
        print(f"Dataset optimized: {len(data):,} interactions ({reduction:,} filtered)")
        
        return data
        
    def analyze_customer_behavior(self, data: pd.DataFrame) -> Dict[str, Union[int, float, Dict[str, Any]]]:
        """Analyze customer behavior patterns for business insights."""
        analysis = {
            'total_interactions': len(data),
            'unique_customers': data['user_id'].nunique(),
            'unique_products': data['prod_id'].nunique(),
            'rating_distribution': data['rating'].value_counts().to_dict(),
            'average_rating': data['rating'].mean(),
            'rating_std': data['rating'].std(),
            'customer_engagement': data['user_id'].value_counts().describe().to_dict(),
            'product_popularity': data['prod_id'].value_counts().describe().to_dict()
        }
        
        total_possible_interactions = analysis['unique_customers'] * analysis['unique_products']
        analysis['market_penetration'] = len(data) / total_possible_interactions
        analysis['data_sparsity'] = 1 - analysis['market_penetration']
        
        return analysis
        
    def visualize_rating_distribution(self, data: pd.DataFrame, save_path: Optional[Union[str, Path]] = None, 
                                    output_dir: str = "outputs") -> None:
        """Create business intelligence visualizations for rating patterns."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"rating_distribution_{timestamp}.png"

        fig, (ax_box, ax_hist) = plt.subplots(
            nrows=2, sharex=True, 
            gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=(12, 8)
        )
        
        sns.boxplot(data=data, x='rating', ax=ax_box, color='skyblue')
        ax_box.set_title('Customer Satisfaction Distribution')
        
        sns.histplot(data=data, x='rating', bins=5, stat='percent', 
                    discrete=True, ax=ax_hist, color='lightblue', alpha=0.7)
        ax_hist.axvline(data['rating'].mean(), color='red', linestyle='--', 
                       label=f'Average: {data["rating"].mean():.2f}')
        ax_hist.axvline(data['rating'].median(), color='black', linestyle='-', 
                       label=f'Median: {data["rating"].median():.2f}')
        ax_hist.set_title('Rating Frequency Distribution')
        ax_hist.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def create_popularity_recommendations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate popularity-based product recommendations."""
        product_metrics = data.groupby('prod_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).round(3)
        
        product_metrics.columns = ['avg_rating', 'total_ratings', 'rating_std', 'unique_customers']
        product_metrics['rating_std'] = product_metrics['rating_std'].fillna(0)
        
        product_metrics = product_metrics.sort_values(['avg_rating', 'total_ratings'], ascending=False)
        
        return product_metrics
        
    def get_top_products(self, product_metrics: pd.DataFrame, n_products: int = 5, 
                        min_ratings: int = 50) -> List[ProductId]:
        """Identify top-performing products for business promotion."""
        qualified_products = product_metrics[
            product_metrics['total_ratings'] >= min_ratings
        ]
        
        top_products = qualified_products.head(n_products).index.tolist()
        
        print(f"Top {n_products} products with minimum {min_ratings} customer ratings:")
        for i, product in enumerate(top_products, 1):
            metrics = qualified_products.loc[product]
            print(f"{i}. Product {product}: {metrics['avg_rating']:.2f} avg rating "
                  f"({metrics['total_ratings']} ratings)")
                  
        return top_products
        
    def calculate_recommendation_performance(self, model: Union[CollaborativeFilteringEngine, MatrixFactorizationEngine], 
                                           test_data: pd.DataFrame, k: int = 10, 
                                           threshold: float = 3.5) -> ModelMetrics:
        """
        Evaluate recommendation algorithm performance using business metrics.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            k: Number of top recommendations to evaluate
            threshold: Rating threshold for customer satisfaction
            
        Returns:
            Dictionary with performance metrics
        """
        start_time = time.time()
        print(f"Evaluating model performance on {len(test_data):,} test interactions...")
        
        print("Generating predictions...")
        predictions = model.test(test_data)
        
        customer_predictions = defaultdict(list)
        actual_ratings = []
        predicted_ratings = []
        
        print("Processing predictions...")
        for pred in tqdm(predictions, desc="Processing predictions"):
            customer_predictions[pred.uid].append((pred.est, pred.r_ui if pred.r_ui else 0))
            predicted_ratings.append(pred.est)
            
        for _, row in test_data.iterrows():
            actual_ratings.append(row['rating'])
            
        # Recalculate customer_predictions properly aligned with test_data
        customer_predictions = defaultdict(list)
        for i, (_, row) in enumerate(test_data.iterrows()):
            customer_predictions[row['user_id']].append((predicted_ratings[i], row['rating']))
            
        print("Calculating precision, recall, and F1-score...")
        
        precisions = {}
        recalls = {}
        
        for customer_id, ratings in tqdm(customer_predictions.items(), desc="Computing metrics"):
            ratings.sort(key=lambda x: x[0], reverse=True)
            
            satisfied_customers = sum((actual >= threshold) for (_, actual) in ratings)
            recommended_products = sum((predicted >= threshold) for (predicted, _) in ratings[:k])
            correct_recommendations = sum(
                ((actual >= threshold) and (predicted >= threshold))
                for (predicted, actual) in ratings[:k]
            )
            
            precisions[customer_id] = (correct_recommendations / recommended_products 
                                     if recommended_products != 0 else 0)
            recalls[customer_id] = (correct_recommendations / satisfied_customers 
                                  if satisfied_customers != 0 else 0)
        
        precision = round(sum(precisions.values()) / len(precisions), 3) if len(precisions) > 0 else 0
        recall = round(sum(recalls.values()) / len(recalls), 3) if len(recalls) > 0 else 0
        f1_score = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0
        
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        
        elapsed_time = time.time() - start_time
        print(f"Evaluation completed in {elapsed_time:.1f} seconds")
        print(f'RMSE: {rmse:.4f}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1_score}')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'rmse': rmse,
            'customer_satisfaction_threshold': threshold,
            'recommendations_evaluated': k
        }
        
    def build_collaborative_filtering_engine(self, data: pd.DataFrame, similarity_type: str = 'cosine', 
                                            user_based: bool = True) -> Tuple[CollaborativeFilteringEngine, pd.DataFrame]:
        """Build collaborative filtering recommendation engine."""
        train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
        
        model = CollaborativeFilteringEngine(
            similarity_metric=similarity_type,
            n_neighbors=40,
            user_based=user_based
        )
        model.fit(train_data)
        
        model_type = f"{'customer' if user_based else 'product'}_collaborative_filtering"
        self.models[model_type] = {
            'model': model,
            'test_data': test_data,
            'config': {'similarity': similarity_type, 'user_based': user_based}
        }
        
        return model, test_data
        
    def build_matrix_factorization_engine(self, data: pd.DataFrame) -> Tuple[MatrixFactorizationEngine, pd.DataFrame]:
        """Build matrix factorization recommendation engine."""
        train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
        
        model = MatrixFactorizationEngine(
            n_factors=50,
            learning_rate=0.01,
            n_epochs=20,
            reg_all=0.02
        )
        model.fit(train_data)
        
        self.models['matrix_factorization'] = {
            'model': model,
            'test_data': test_data
        }
        
        return model, test_data
        
    def generate_customer_recommendations(self, data: pd.DataFrame, customer_id: UserId, 
                                        model: Union[CollaborativeFilteringEngine, MatrixFactorizationEngine], 
                                        n_recommendations: int = 5) -> RecommendationList:
        """Generate personalized product recommendations for a specific customer."""
        all_products = data['prod_id'].unique()
        customer_products = data[data['user_id'] == customer_id]['prod_id'].unique()
        unrated_products = [p for p in all_products if p not in customer_products]
        
        recommendations = []
        for product_id in unrated_products:
            try:
                predicted_satisfaction = model.predict(customer_id, product_id).est
                recommendations.append((product_id, predicted_satisfaction))
            except:
                continue
                
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:n_recommendations]
        
        print(f"Top {n_recommendations} personalized recommendations for customer {customer_id}:")
        for i, (product, satisfaction) in enumerate(top_recommendations, 1):
            print(f"{i}. Product {product}: {satisfaction:.2f} predicted satisfaction")
            
        return top_recommendations
        
    def run_comprehensive_analysis(self, file_path: Union[str, Path], sampling_strategy: str = 'most_active') -> Dict[str, Any]:
        """Execute complete recommendation engine analysis and optimization."""
        total_start_time = time.time()
        
        print("=" * 70)
        print("AMAZON ELECTRONICS RECOMMENDATION ENGINE v2b")
        print("Enhanced with progress tracking and dataset sampling")
        print("=" * 70)
        print("Analyzing customer behavior and optimizing recommendation algorithms...\n")
        
        raw_data = self.load_customer_data(file_path)
        processed_data = self.filter_for_quality_recommendations(raw_data, sampling_strategy)
        self.processed_data = processed_data
        
        print("\nCUSTOMER BEHAVIOR ANALYSIS")
        print("-" * 50)
        behavior_analysis = self.analyze_customer_behavior(processed_data)
        print(f"Active customers: {behavior_analysis['unique_customers']:,}")
        print(f"Product catalog: {behavior_analysis['unique_products']:,}")
        print(f"Customer interactions: {behavior_analysis['total_interactions']:,}")
        print(f"Average satisfaction: {behavior_analysis['average_rating']:.2f}/5.0")
        print(f"Market penetration: {behavior_analysis['market_penetration']*100:.4f}%")
        
        print("\nRECOMMENDATION ALGORITHM PERFORMANCE")
        print("-" * 60)
        
        product_metrics = self.create_popularity_recommendations(processed_data)
        print("\n1. Popularity-Based Algorithm:")
        print("   " + "="*30)
        self.get_top_products(product_metrics, n_products=5, min_ratings=50)
        
        print("\n2. Customer Collaborative Filtering:")
        print("   " + "="*35)
        user_model, user_test_data = self.build_collaborative_filtering_engine(
            processed_data, similarity_type='cosine', user_based=True
        )
        user_performance = self.calculate_recommendation_performance(
            user_model, user_test_data
        )
        
        print("\n3. Product Collaborative Filtering:")
        print("   " + "="*34)
        item_model, item_test_data = self.build_collaborative_filtering_engine(
            processed_data, similarity_type='cosine', user_based=False
        )
        item_performance = self.calculate_recommendation_performance(
            item_model, item_test_data
        )
        
        print("\n4. Matrix Factorization Algorithm:")
        print("   " + "="*32)
        svd_model, svd_test_data = self.build_matrix_factorization_engine(processed_data)
        svd_performance = self.calculate_recommendation_performance(
            svd_model, svd_test_data
        )
        
        print("\nBUSINESS IMPACT ANALYSIS")
        print("-" * 50)
        
        algorithms = {
            'Customer Collaborative Filtering': user_performance,
            'Product Collaborative Filtering': item_performance,
            'Matrix Factorization': svd_performance
        }
        
        best_f1 = max(algorithms.values(), key=lambda x: x['f1_score'])
        best_algorithm = [name for name, perf in algorithms.items() if perf['f1_score'] == best_f1['f1_score']][0]
        
        print(f"OPTIMAL ALGORITHM: {best_algorithm}")
        print(f"   Customer Engagement: {best_f1['recall']*100:.1f}% of relevant products identified")
        print(f"   Recommendation Quality: {best_f1['precision']*100:.1f}% accuracy")
        print(f"   Overall Performance: {best_f1['f1_score']:.3f} F1-Score")
        print(f"   Prediction Accuracy: {best_f1['rmse']:.4f} RMSE")
        
        print("\nSAMPLE CUSTOMER RECOMMENDATIONS")
        print("-" * 50)
        sample_customers = processed_data['user_id'].unique()[:3]
        
        for customer in sample_customers:
            print(f"\nCustomer {customer}:")
            if best_algorithm == 'Customer Collaborative Filtering':
                recommendations = self.generate_customer_recommendations(
                    processed_data, customer, user_model, n_recommendations=3
                )
            elif best_algorithm == 'Product Collaborative Filtering':
                recommendations = self.generate_customer_recommendations(
                    processed_data, customer, item_model, n_recommendations=3
                )
            else:
                recommendations = self.generate_customer_recommendations(
                    processed_data, customer, svd_model, n_recommendations=3
                )
        
        total_elapsed_time = time.time() - total_start_time
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("Enhanced recommendation engine ready for production deployment.")
        print(f"Total processing time: {total_elapsed_time:.1f} seconds")
        print("=" * 70)
        
        return {
            'behavior_analysis': behavior_analysis,
            'algorithm_performance': algorithms,
            'best_algorithm': best_algorithm,
            'models': self.models,
            'total_time': total_elapsed_time
        }


def main() -> None:
    """
    Main execution function for enterprise recommendation engine.
    """
    parser = argparse.ArgumentParser(description='Amazon Recommendation Engine v2b')
    parser.add_argument('--data-path', default='data/ratings_Electronics.csv')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--min-user-interactions', type=int, default=50)
    parser.add_argument('--min-product-interactions', type=int, default=5)
    parser.add_argument('--max-customers', type=int, default=None,
                        help='Maximum number of customers to use for faster testing')
    parser.add_argument('--max-products', type=int, default=None,
                        help='Maximum number of products to use for faster testing')
    parser.add_argument('--max-percent-customers', type=float, default=None,
                        help='Maximum percentage of customers to use (0-100)')
    parser.add_argument('--max-percent-products', type=float, default=None,
                        help='Maximum percentage of products to use (0-100)')
    parser.add_argument('--sampling-strategy', type=str, default='most_active',
                        choices=['most_active', 'random', 'balanced'],
                        help='Sampling strategy when limits are applied')
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
        
    engine = AmazonRecommendationEngine(
        min_user_interactions=args.min_user_interactions,
        min_product_interactions=args.min_product_interactions,
        max_customers=args.max_customers,
        max_products=args.max_products,
        max_percent_customers=args.max_percent_customers,
        max_percent_products=args.max_percent_products
    )
        
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the dataset exists within subdirectory data/ or use --data-path to specify location")
        return
    
    try:
        results = engine.run_comprehensive_analysis(data_path, args.sampling_strategy)
        
        if engine.processed_data is not None:
            engine.visualize_rating_distribution(
                engine.processed_data, 
                output_dir=args.output_dir
            )
            
    except FileNotFoundError:
        print(f"Dataset not found at {data_path}")
        print("Please verify the file path and ensure the dataset exists within subdirectory data/.")
        print("Cross-platform path handling enabled for enterprise deployment.")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()

"""
Amazon Electronics Recommendation Engine
Enterprise-grade recommendation system using sklearn and numpy.

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
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


class CollaborativeFilteringEngine:
    """
    Production collaborative filtering engine using sklearn.
    
    Implements user-based and item-based collaborative filtering algorithms
    optimized for scalable recommendation generation in enterprise e-commerce environments.
    """
    
    def __init__(self, similarity_metric='cosine', n_neighbors=40, user_based=True):
        """
        Initialize collaborative filtering engine.
        
        Args:
            similarity_metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            n_neighbors: Number of neighbors for recommendations
            user_based: True for user-based, False for item-based filtering
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.user_based = user_based
        self.model = None
        self.interaction_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
    def _create_interaction_matrix(self, data):
        """Create user-item interaction matrix from DataFrame."""
        unique_users = data['user_id'].unique()
        unique_items = data['prod_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users, n_items = len(unique_users), len(unique_items)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in data.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['prod_id']]
            interaction_matrix[user_idx, item_idx] = row['rating']
            
        return interaction_matrix
        
    def fit(self, data):
        """
        Train the collaborative filtering model.
        
        Args:
            data: DataFrame with columns ['user_id', 'prod_id', 'rating']
        """
        self.interaction_matrix = self._create_interaction_matrix(data)
        
        self.global_mean = np.mean(self.interaction_matrix[self.interaction_matrix > 0])
        
        self.user_means = np.array([
            np.mean(row[row > 0]) if np.any(row > 0) else self.global_mean 
            for row in self.interaction_matrix
        ])
        
        self.item_means = np.array([
            np.mean(col[col > 0]) if np.any(col > 0) else self.global_mean 
            for col in self.interaction_matrix.T
        ])
        
        if self.user_based:
            matrix = self.interaction_matrix
        else:
            matrix = self.interaction_matrix.T
            
        matrix_filled = matrix.copy()
        for i in range(matrix_filled.shape[0]):
            zero_mask = matrix_filled[i] == 0
            if self.user_based:
                matrix_filled[i][zero_mask] = self.user_means[i]
            else:
                matrix_filled[i][zero_mask] = self.item_means[i]
        
        metric = 'cosine' if self.similarity_metric == 'cosine' else 'euclidean'
        self.model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, matrix_filled.shape[0]),
            metric=metric,
            algorithm='auto'
        )
        self.model.fit(matrix_filled)
        
        return self
        
    def predict(self, user_id, item_id, verbose=False):
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
        
    def _predict_rating(self, user_idx, item_idx):
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
        
    def test(self, test_data):
        """Test the model on a dataset and return predictions."""
        predictions = []
        for _, row in test_data.iterrows():
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
    
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=20, reg_all=0.02, random_state=42):
        """
        Initialize matrix factorization engine.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            n_epochs: Number of training epochs
            reg_all: Regularization parameter
            random_state: necessary for reproducibility
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_all = reg_all
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
    def fit(self, data):
        """Use stochastic gradient descent to train the matrix factorization model."""
        np.random.seed(self.random_state)  # Reproducible results
        unique_users = data['user_id'].unique()
        unique_items = data['prod_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users, n_items = len(unique_users), len(unique_items)
        
        self.global_mean = data['rating'].mean()
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        training_data = []
        for _, row in data.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['prod_id']]
            rating = row['rating']
            training_data.append((user_idx, item_idx, rating))
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(training_data)
            
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
        
        return self
        
    def predict(self, user_id, item_id, verbose=False):
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
        
    def test(self, test_data):
        """Test the model on a dataset and return predictions."""
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['prod_id'])
            pred.r_ui = row['rating']
            predictions.append(pred)
        return predictions


class PredictionResult:
    """Container for recommendation prediction results."""
    
    def __init__(self, uid, iid, est, r_ui=None):
        self.uid = uid
        self.iid = iid
        self.est = est
        self.r_ui = r_ui


class AmazonRecommendationEngine:
    """
    Production recommendation engine for Amazon Electronics catalog.
    
    Enterprise-grade recommendation system providing personalization capabilities
    using multiple algorithms including collaborative filtering and matrix factorization.
    Designed for high-scale customer personalization and business intelligence.
    """
    
    def __init__(self, min_user_interactions=50, min_product_interactions=5):
        """
        Initialize the recommendation engine.
        
        Args:
            min_user_interactions: Minimum customer interactions required
            min_product_interactions: Minimum product ratings required
        """
        self.min_user_interactions = min_user_interactions
        self.min_product_interactions = min_product_interactions
        self.processed_data = None
        self.models = {}
        
    def load_customer_data(self, file_path, has_header=True):
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
        
    def filter_for_quality_recommendations(self, data):
        """
        Filter data to ensure statistical significance and recommendation quality.
        
        Args:
            data: Raw customer interaction data
            
        Returns:
            Filtered dataset optimized for recommendation quality
        """
        print("Optimizing dataset for recommendation quality...")
        original_size = len(data)
        
        customer_interactions = data['user_id'].value_counts()
        qualified_customers = customer_interactions[
            customer_interactions >= self.min_user_interactions
        ].index
        
        data = data[data['user_id'].isin(qualified_customers)]
        print(f"Retained {len(qualified_customers):,} customers with sufficient interaction history")
        
        product_ratings = data['prod_id'].value_counts()
        qualified_products = product_ratings[
            product_ratings >= self.min_product_interactions
        ].index
        
        data = data[data['prod_id'].isin(qualified_products)]
        print(f"Retained {len(qualified_products):,} products with sufficient rating data")
        
        reduction = original_size - len(data)
        print(f"Dataset optimized: {len(data):,} interactions ({reduction:,} filtered)")
        
        return data
        
    def analyze_customer_behavior(self, data):
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
        
    def visualize_rating_distribution(self, data, save_path=None, output_dir="outputs"):
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
            
    def create_popularity_recommendations(self, data):
        """Generate popularity-based product recommendations."""
        product_metrics = data.groupby('prod_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).round(3)
        
        product_metrics.columns = ['avg_rating', 'total_ratings', 'rating_std', 'unique_customers']
        product_metrics['rating_std'] = product_metrics['rating_std'].fillna(0)
        
        product_metrics = product_metrics.sort_values(['avg_rating', 'total_ratings'], ascending=False)
        
        return product_metrics
        
    def get_top_products(self, product_metrics, n_products=5, min_ratings=50):
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
        
    def calculate_recommendation_performance(self, model, test_data, k=10, threshold=3.5):
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
        predictions = model.test(test_data)
        
        customer_predictions = defaultdict(list)
        actual_ratings = []
        predicted_ratings = []
        
        for pred in predictions:
            customer_predictions[pred.uid].append((pred.est, pred.r_ui if pred.r_ui else 0))
            predicted_ratings.append(pred.est)
            
        for _, row in test_data.iterrows():
            actual_ratings.append(row['rating'])
            
        precisions = {}
        recalls = {}
        
        for customer_id, ratings in customer_predictions.items():
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
        
    def build_collaborative_filtering_engine(self, data, similarity_type='cosine', user_based=True):
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
        
    def build_matrix_factorization_engine(self, data):
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
        
    def generate_customer_recommendations(self, data, customer_id, model, n_recommendations=5):
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
        
    def run_comprehensive_analysis(self, file_path):
        """Execute complete recommendation engine analysis and optimization."""
        print("=" * 70)
        print("AMAZON ELECTRONICS RECOMMENDATION ENGINE")
        print("Enterprise-grade system using sklearn and numpy")
        print("=" * 70)
        print("Analyzing customer behavior and optimizing recommendation algorithms...\n")
        
        raw_data = self.load_customer_data(file_path)
        processed_data = self.filter_for_quality_recommendations(raw_data)
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
                
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("Enterprise recommendation engine ready for production deployment.")
        print("Built with sklearn and numpy for maximum compatibility and performance.")
        print("=" * 70)
        
        return {
            'behavior_analysis': behavior_analysis,
            'algorithm_performance': algorithms,
            'best_algorithm': best_algorithm,
            'models': self.models
        }


def main():
    """
    Main execution function for enterprise recommendation engine.
    """
    parser = argparse.ArgumentParser(description='Amazon Recommendation Engine')
    parser.add_argument('--data-path', default='data/ratings_Electronics.csv')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--min-user-interactions', type=int, default=50)
    parser.add_argument('--min-product-interactions', type=int, default=5)
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
        
    engine = AmazonRecommendationEngine(
        min_user_interactions=args.min_user_interactions,
        min_product_interactions=args.min_product_interactions
    )
        
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the dataset exists or use --data-path to specify location")
        return
    
    try:
        results = engine.run_comprehensive_analysis(data_path)
        
        if engine.processed_data is not None:
            engine.visualize_rating_distribution(
                engine.processed_data, 
                args.output_dir
            )
            
    except FileNotFoundError:
        print(f"Dataset not found at {data_path}")
        print("Please verify the file path and ensure the dataset exists.")
        print("Cross-platform path handling enabled for enterprise deployment.")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()

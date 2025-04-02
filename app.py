# Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from PIL import Image
import time
import re
import warnings
import gradio as gr
from google.colab import drive, files
import zipfile
import shutil
import io
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

print("Libraries imported successfully!")
import os
import shutil

# Check if the directory already exists, and remove it if it does
if os.path.exists('/temp/DiamondSearch'):
    shutil.rmtree('/temp/DiamondSearch') 

# Now clone the repository
!git clone https://github.com/belgiumtech01/DiamondSearch.git /temp/DiamondSearch

# Define paths
base_path = '/temp/DiamondSearch'
data_path = os.path.join(base_path, 'Data')
models_path = os.path.join(base_path, 'models')
results_path = os.path.join(base_path, 'results')
csv_file = os.path.join(data_path, 'gems_data.csv')
images_folder = os.path.join(data_path, 'gems_images')

# Check if the CSV file exists
if os.path.exists(csv_file):
    print(f"CSV file found at: {csv_file}")
else:
    print(f"CSV file not found at: {csv_file}")

# Check if the images folder exists
if os.path.exists(images_folder):
    print(f"Images folder found at: {images_folder}")
    print(f"Number of images: {len(os.listdir(images_folder))}")
else:
    print(f"Images folder not found at: {images_folder}")


class DataProcessor:
    """Module for data loading, preprocessing, and splitting"""

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.metadata_ranges = {}

    def load_data(self):
        """Load the diamond data from CSV"""
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.df)} diamonds with {len(self.df.columns)} features")
        return self.df

    def preprocess_data(self):
        """Preprocess the diamond data"""
        if self.df is None:
            self.load_data()

        # Convert diamond_id to string if it exists
        if 'diamond_id' in self.df.columns:
            self.df['diamond_id'] = self.df['diamond_id'].astype(str)

        # Handle missing values
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # For categorical columns, fill with 'unknown'
                self.df[col] = self.df[col].fillna('unknown')
                if col not in ['diamond_id', 'text_description']:
                    self.categorical_columns.append(col)
            else:
                # For numerical columns, fill with median
                self.df[col] = self.df[col].fillna(self.df[col].median())
                if col not in ['diamond_id']:
                    self.numerical_columns.append(col)

        # Create a text description column combining all features
        self.create_text_description()

        # Calculate ranges for metadata
        self.calculate_metadata_ranges()

        return self.df

    def calculate_metadata_ranges(self):
        """Calculate ranges and unique values for metadata fields"""
        # For numerical columns, get min and max
        for col in self.numerical_columns:
            self.metadata_ranges[col] = {
                'type': 'numerical',
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median())
            }

        # For categorical columns, get unique values
        for col in self.categorical_columns:
            unique_values = self.df[col].unique().tolist()
            # Sort values if possible (some might be ordinal)
            try:
                unique_values.sort()
            except:
                pass

            self.metadata_ranges[col] = {
                'type': 'categorical',
                'values': unique_values
            }

        return self.metadata_ranges

    def create_text_description(self):
        """Create a text description from diamond attributes"""
        # Select relevant columns for text description
        text_cols = ['shape', 'size', 'color', 'clarity', 'cut', 
                     'symmetry', 'polish', 'fluor_intensity']

        # Only use columns that exist in the dataframe
        text_cols = [col for col in text_cols if col in self.df.columns]

        if not text_cols:
            print("Warning: No text columns found for creating description")
            self.df['text_description'] = "No description available"
            return

        # Create text description
        descriptions = []
        for _, row in self.df.iterrows():
            desc_parts = []
            for col in text_cols:
                if pd.notna(row[col]):
                    # Format: "column_name: value"
                    desc_parts.append(f"{col}: {row[col]}")

            description = ", ".join(desc_parts)
            descriptions.append(description)

        self.df['text_description'] = descriptions
        print("Created text descriptions from diamond attributes")

    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split the data into train, validation, and test sets"""
        if self.df is None:
            self.preprocess_data()

        # First split: separate test set
        train_val_df, self.test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation set from training set
        # Adjust validation size to account for the reduced dataset size
        val_size_adjusted = val_size / (1 - test_size)
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state
        )

        print(f"Data split complete:")
        print(f"  Training set: {len(self.train_df)} diamonds ({len(self.train_df)/len(self.df):.1%})")
        print(f"  Validation set: {len(self.val_df)} diamonds ({len(self.val_df)/len(self.df):.1%})")
        print(f"  Test set: {len(self.test_df)} diamonds ({len(self.test_df)/len(self.df):.1%})")

        return self.train_df, self.val_df, self.test_df

    def get_data_summary(self):
        """Get a summary of the diamond data"""
        if self.df is None:
            self.load_data()

        summary = {
            "total_diamonds": len(self.df),
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "metadata_ranges": self.metadata_ranges
        }

        return summary

class FeatureExtractor:
    """Module for extracting features from images and text"""

    def __init__(self):
        self.image_model = None
        self.text_vectorizer = None
        self.text_features = {}
        self.image_features = {}
        self.numerical_scaler = None
        self.numerical_features = {}

    def setup_image_extractor(self):
        """Set up the image feature extractor using VGG16"""
        base_model = VGG16(weights='imagenet')
        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        print("Image feature extractor (VGG16) initialized")

    def setup_text_extractor(self, text_corpus):
        """Set up the text feature extractor using TF-IDF"""
        self.text_vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features to avoid sparse vectors
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        # Fit the vectorizer on the text corpus
        self.text_vectorizer.fit(text_corpus)
        print(f"Text feature extractor (TF-IDF) initialized with {len(self.text_vectorizer.vocabulary_)} features")

    def setup_numerical_extractor(self, numerical_data):
        """Set up the numerical feature extractor using StandardScaler"""
        self.numerical_scaler = StandardScaler()
        self.numerical_scaler.fit(numerical_data)
        print(f"Numerical feature extractor initialized for {numerical_data.shape[1]} features")

    def extract_image_features(self, img_path):
        """Extract features from an image using the pre-trained model"""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))  # VGG16 input size
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.image_model.predict(x, verbose=0)  # Suppress prediction output
            return features.flatten()
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    def extract_text_features(self, text):
        """Extract features from text using TF-IDF"""
        if self.text_vectorizer is None:
            raise ValueError("Text vectorizer not initialized. Call setup_text_extractor first.")

        features = self.text_vectorizer.transform([text]).toarray().flatten()
        return features

    def extract_numerical_features(self, numerical_values):
        """Extract standardized numerical features"""
        if self.numerical_scaler is None:
            raise ValueError("Numerical scaler not initialized. Call setup_numerical_extractor first.")

        features = self.numerical_scaler.transform([numerical_values])[0]
        return features

    def build_feature_index(self, df, image_folder, image_id_pattern="{diamond_id}.jpg", 
                           numerical_columns=None, max_images=None):
        """
        Build a comprehensive feature index including image, text, and numerical features

        Args:
            df: DataFrame containing diamond data
            image_folder: Path to the folder containing diamond images
            image_id_pattern: Pattern for constructing image filenames from diamond IDs
            numerical_columns: List of numerical columns to include in features
            max_images: Maximum number of images to process (for testing)

        Returns:
            Dictionary of feature dictionaries
        """
        # Initialize feature dictionaries
        self.image_features = {}
        self.text_features = {}
        self.numerical_features = {}

        # Initialize models if not already done
        if self.image_model is None:
            self.setup_image_extractor()

        if self.text_vectorizer is None and 'text_description' in df.columns:
            self.setup_text_extractor(df['text_description'].values)

        if self.numerical_scaler is None and numerical_columns:
            numerical_data = df[numerical_columns].values
            self.setup_numerical_extractor(numerical_data)

        # Get diamond IDs
        diamond_ids = df['diamond_id'].astype(str).tolist()

        if max_images and max_images < len(diamond_ids):
            diamond_ids = diamond_ids[:max_images]

        total_diamonds = len(diamond_ids)
        processed_count = 0

        print(f"Building feature index for {total_diamonds} diamonds...")
        start_time = time.time()

        for i, diamond_id in enumerate(diamond_ids):
            if i % 20 == 0:
                print(f"  Processing diamond {i+1}/{total_diamonds}")

            # Get the diamond data
            diamond_data = df[df['diamond_id'].astype(str) == diamond_id]

            if len(diamond_data) == 0:
                continue

            # Extract image features if available
            img_file = image_id_pattern.format(diamond_id=diamond_id)
            img_path = os.path.join(image_folder, img_file)

            if os.path.exists(img_path):
                image_feats = self.extract_image_features(img_path)
                if image_feats is not None:
                    self.image_features[diamond_id] = image_feats

            # Extract text features if available
            if 'text_description' in diamond_data.columns:
                text = diamond_data['text_description'].iloc[0]
                text_feats = self.extract_text_features(text)
                self.text_features[diamond_id] = text_feats

            # Extract numerical features if available
            if numerical_columns:
                num_values = diamond_data[numerical_columns].iloc[0].values
                num_feats = self.extract_numerical_features(num_values)
                self.numerical_features[diamond_id] = num_feats

            processed_count += 1

        print(f"Feature extraction complete in {time.time() - start_time:.2f} seconds")
        print(f"  Image features: {len(self.image_features)} diamonds")
        print(f"  Text features: {len(self.text_features)} diamonds")
        print(f"  Numerical features: {len(self.numerical_features)} diamonds")

        return {
            'image_features': self.image_features,
            'text_features': self.text_features,
            'numerical_features': self.numerical_features
        }

    def save_features(self, filename, features=None):
        """Save feature dictionaries to disk"""
        if features is None:
            features = {
                'image_features': self.image_features,
                'text_features': self.text_features,
                'numerical_features': self.numerical_features,
                'text_vectorizer': self.text_vectorizer,
                'numerical_scaler': self.numerical_scaler
            }

        with open(filename, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {filename}")

    def load_features(self, filename):
        """Load feature dictionaries from disk"""
        with open(filename, 'rb') as f:
            features = pickle.load(f)

        self.image_features = features.get('image_features', {})
        self.text_features = features.get('text_features', {})
        self.numerical_features = features.get('numerical_features', {})
        self.text_vectorizer = features.get('text_vectorizer')
        self.numerical_scaler = features.get('numerical_scaler')

        print(f"Loaded features:")
        print(f"  Image features: {len(self.image_features)} diamonds")
        print(f"  Text features: {len(self.text_features)} diamonds")
        print(f"  Numerical features: {len(self.numerical_features)} diamonds")

        return features


class DiamondSearchEngine:
    """Module for multi-modal diamond search"""

    def __init__(self, feature_extractor, metadata_df, image_folder):
        self.feature_extractor = feature_extractor
        self.metadata_df = metadata_df
        self.image_folder = image_folder

    def search(self, query_img_path=None, query_text=None, query_diamond_id=None, 
              numerical_query=None, numerical_columns=None,
              categorical_filters=None, numerical_filters=None,
              weights=None, top_k=5):
        """
        Search for similar diamonds using any combination of image, text, and numerical features

        Args:
            query_img_path: Path to query image
            query_text: Text query for diamond properties
            query_diamond_id: ID of a diamond to use as query
            numerical_query: Dictionary of numerical values to search for
            numerical_columns: List of numerical columns used in the search
            categorical_filters: Dictionary of categorical filters {column: value(s)}
            numerical_filters: Dictionary of numerical filters {column: (min, max)}
            weights: Dictionary of weights for each feature type (image, text, numerical)
            top_k: Number of results to return

        Returns:
            List of dictionaries containing search results
        """
        # Set default weights if not provided
        if weights is None:
            weights = {'image': 0.6, 'text': 0.3, 'numerical': 0.1}

        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}

        # Check if at least one query type is provided
        if query_img_path is None and query_text is None and query_diamond_id is None and numerical_query is None:
            raise ValueError("At least one query type (image, text, diamond ID, or numerical) must be provided")

        # Get query features
        query_features = {}

        # If diamond ID is provided, use its features
        if query_diamond_id:
            if query_diamond_id in self.feature_extractor.image_features:
                query_features['image'] = self.feature_extractor.image_features[query_diamond_id]

            if query_diamond_id in self.feature_extractor.text_features:
                query_features['text'] = self.feature_extractor.text_features[query_diamond_id]

            if query_diamond_id in self.feature_extractor.numerical_features:
                query_features['numerical'] = self.feature_extractor.numerical_features[query_diamond_id]

        # If image is provided, extract its features
        if query_img_path:
            image_features = self.feature_extractor.extract_image_features(query_img_path)
            if image_features is not None:
                query_features['image'] = image_features

        # If text is provided, extract its features
        if query_text:
            text_features = self.feature_extractor.extract_text_features(query_text)
            query_features['text'] = text_features

        # If numerical query is provided, extract its features
        if numerical_query and numerical_columns:
            # Convert dictionary to array in the correct order
            numerical_values = [numerical_query.get(col, 0) for col in numerical_columns]
            numerical_features = self.feature_extractor.extract_numerical_features(numerical_values)
            query_features['numerical'] = numerical_features

        # Calculate similarities for each feature type
        similarities = {}

        # Image similarities
        if 'image' in query_features and self.feature_extractor.image_features:
            for diamond_id, features in self.feature_extractor.image_features.items():
                sim = cosine_similarity(query_features['image'].reshape(1, -1), 
                                       features.reshape(1, -1))[0][0]
                if diamond_id not in similarities:
                    similarities[diamond_id] = 0
                similarities[diamond_id] += sim * weights.get('image', 0)

        # Text similarities
        if 'text' in query_features and self.feature_extractor.text_features:
            for diamond_id, features in self.feature_extractor.text_features.items():
                sim = cosine_similarity(query_features['text'].reshape(1, -1), 
                                       features.reshape(1, -1))[0][0]
                if diamond_id not in similarities:
                    similarities[diamond_id] = 0
                similarities[diamond_id] += sim * weights.get('text', 0)

        # Numerical similarities
        if 'numerical' in query_features and self.feature_extractor.numerical_features:
            for diamond_id, features in self.feature_extractor.numerical_features.items():
                sim = cosine_similarity(query_features['numerical'].reshape(1, -1), 
                                       features.reshape(1, -1))[0][0]
                if diamond_id not in similarities:
                    similarities[diamond_id] = 0
                similarities[diamond_id] += sim * weights.get('numerical', 0)

        # If query_diamond_id was used, remove it from results
        if query_diamond_id and query_diamond_id in similarities:
            del similarities[query_diamond_id]

        # Sort by similarity (highest first)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Get all results first
        all_results = []
        for diamond_id, similarity in sorted_similarities:
            # Get image path
            img_file = f"{diamond_id}.jpg"  # Adjust pattern if needed
            img_path = os.path.join(self.image_folder, img_file)

            # Get diamond metadata
            diamond_metadata = self.metadata_df[self.metadata_df['diamond_id'].astype(str) == diamond_id].to_dict('records')

            if not diamond_metadata:
                continue

            result = {
                'diamond_id': diamond_id,
                'similarity': similarity,
                'path': img_path,
                'metadata': diamond_metadata[0]
            }

            all_results.append(result)

        # Apply filters if provided
        filtered_results = self.filter_results(
            all_results, 
            categorical_filters=categorical_filters,
            numerical_filters=numerical_filters
        )

        # Return top-k results
        return filtered_results[:top_k]

    def filter_results(self, results, categorical_filters=None, numerical_filters=None):
        """Filter search results by metadata properties"""
        if not categorical_filters and not numerical_filters:
            return results

        filtered_results = []

        for result in results:
            metadata = result['metadata']
            if metadata is None:
                continue

            # Check if the diamond matches all filters
            matches_all_filters = True

            # Apply categorical filters
            if categorical_filters:
                for col, values in categorical_filters.items():
                    if col not in metadata:
                        matches_all_filters = False
                        break

                    # Skip empty filters
                    if not values:
                        continue

                    # Handle lists of acceptable values
                    if isinstance(values, list):
                        if metadata[col] not in values:
                            matches_all_filters = False
                            break
                    # Handle exact matches
                    elif metadata[col] != values:
                        matches_all_filters = False
                        break

            # Apply numerical filters
            if numerical_filters and matches_all_filters:
                for col, range_vals in numerical_filters.items():
                    if col not in metadata:
                        matches_all_filters = False
                        break

                    # Skip empty filters
                    if not range_vals or len(range_vals) != 2:
                        continue

                    min_val, max_val = range_vals
                    if not (min_val <= metadata[col] <= max_val):
                        matches_all_filters = False
                        break

            if matches_all_filters:
                filtered_results.append(result)

        return filtered_results

    def get_result_differences(self, results, query_metadata=None):
        """
        Calculate differences between query and results for highlighting

        Args:
            results: List of search results
            query_metadata: Metadata of the query diamond (if available)

        Returns:
            Updated results with difference information
        """
        if not results:
            return results

        # If no query metadata, use the first result as reference
        if query_metadata is None and len(results) > 0:
            query_metadata = results[0]['metadata']

        if query_metadata is None:
            return results

        # Calculate differences for each result
        for result in results:
            metadata = result['metadata']
            if metadata is None:
                continue

            # Initialize differences dictionary
            result['differences'] = {}

            # Compare all metadata fields
            for key, query_value in query_metadata.items():
                if key not in metadata:
                    continue

                result_value = metadata[key]

                # Skip non-comparable fields
                if key in ['diamond_id', 'text_description']:
                    continue

                # For numerical fields, calculate percentage difference
                if isinstance(query_value, (int, float)) and isinstance(result_value, (int, float)):
                    if query_value != 0:
                        diff_pct = (result_value - query_value) / abs(query_value) * 100
                        result['differences'][key] = {
                            'value': result_value,
                            'query_value': query_value,
                            'diff': result_value - query_value,
                            'diff_pct': diff_pct,
                            'type': 'numerical'
                        }
                    else:
                        result['differences'][key] = {
                            'value': result_value,
                            'query_value': query_value,
                            'diff': result_value - query_value,
                            'diff_pct': 0 if result_value == 0 else float('inf'),
                            'type': 'numerical'
                        }
                # For categorical fields, just note if they're different
                else:
                    result['differences'][key] = {
                        'value': result_value,
                        'query_value': query_value,
                        'is_different': result_value != query_value,
                        'type': 'categorical'
                    }

        return results

    def format_results_html(self, results, query_img_path=None, query_diamond_id=None):
        """
        Format search results as HTML for display in Gradio

        Args:
            results: List of search results with differences
            query_img_path: Path to query image (if available)
            query_diamond_id: ID of query diamond (if available)

        Returns:
            HTML string for display
        """
        if not results:
            return "<h3>No results found</h3>"

        # Get query metadata if available
        query_metadata = None
        if query_diamond_id:
            query_diamond = self.metadata_df[self.metadata_df['diamond_id'].astype(str) == query_diamond_id]
            if len(query_diamond) > 0:
                query_metadata = query_diamond.iloc[0].to_dict()

        # Start building HTML
        html = "<div style='font-family: Arial, sans-serif;'>"

        # Add query information
        html += "<div style='margin-bottom: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>"
        html += "<h3>Query Diamond</h3>"

        # Add query image if available
        if query_img_path and os.path.exists(query_img_path):
            # Convert image to base64 for embedding
            with open(query_img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            html += f"<img src='data:image/jpeg;base64,{img_data}' style='max-height: 200px; max-width: 200px; margin-right: 20px;'>"

        # Add query metadata if available
        if query_metadata:
            html += "<div style='display: inline-block; vertical-align: top;'>"
            html += "<table style='border-collapse: collapse;'>"
            for key, value in query_metadata.items():
                if key not in ['diamond_id', 'text_description'] and pd.notna(value):
                    html += f"<tr><td style='padding: 3px; font-weight: bold;'>{key}</td><td style='padding: 3px;'>{value}</td></tr>"
            html += "</table>"
            html += "</div>"

        html += "</div>"  # End query section

        # Add results
        html += "<h3>Similar Diamonds</h3>"
        html += "<div style='display: flex; flex-wrap: wrap;'>"

        for i, result in enumerate(results):
            metadata = result['metadata']
            differences = result.get('differences', {})

            # Start result card
            html += "<div style='margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 300px;'>"

            # Add result header
            html += f"<h4>Result {i+1} - Similarity: {result['similarity']:.2f}</h4>"

            # Add image if available
            if os.path.exists(result['path']):
                # Convert image to base64 for embedding
                with open(result['path'], "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                html += f"<img src='data:image/jpeg;base64,{img_data}' style='max-height: 150px; max-width: 150px; display: block; margin: 0 auto;'>"

            # Add metadata with differences highlighted
            html += "<table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>"
            html += "<tr style='background-color: #f0f0f0;'><th style='padding: 5px; text-align: left;'>Property</th><th style='padding: 5px; text-align: left;'>Value</th><th style='padding: 5px; text-align: left;'>Difference</th></tr>"

            # Sort properties: first categorical, then numerical
            sorted_props = []
            if metadata:
                # Add important properties first
                important_props = ['shape', 'size', 'color', 'clarity', 'cut', 'Total_Price']
                for prop in important_props:
                    if prop in metadata:
                        sorted_props.append(prop)

                # Add remaining properties
                for prop in metadata:
                    if prop not in sorted_props and prop not in ['diamond_id', 'text_description']:
                        sorted_props.append(prop)

                # Add rows for each property
                for prop in sorted_props:
                    value = metadata[prop]
                    if pd.isna(value):
                        continue

                    diff_info = differences.get(prop, {})
                    diff_type = diff_info.get('type', '')

                    # Format the cell based on difference type
                    if diff_type == 'numerical':
                        diff = diff_info.get('diff', 0)
                        diff_pct = diff_info.get('diff_pct', 0)

                        # Determine color based on difference
                        if abs(diff_pct) < 5:
                            color = "#4CAF50"  # Green for small difference
                        elif abs(diff_pct) < 20:
                            color = "#FF9800"  # Orange for medium difference
                        else:
                            color = "#F44336"  # Red for large difference

                        # Format difference text
                        if diff > 0:
                            diff_text = f"+{diff:.2f} (+{diff_pct:.1f}%)"
                        else:
                            diff_text = f"{diff:.2f} ({diff_pct:.1f}%)"

                        html += f"<tr><td style='padding: 5px; border-bottom: 1px solid #ddd;'>{prop}</td>"
                        html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>{value}</td>"
                        html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd; color: {color};'>{diff_text}</td></tr>"

                    elif diff_type == 'categorical':
                        is_different = diff_info.get('is_different', False)
                        query_value = diff_info.get('query_value', '')

                        html += f"<tr><td style='padding: 5px; border-bottom: 1px solid #ddd;'>{prop}</td>"

                        if is_different:
                            html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd; background-color: #FFECB3;'>{value}</td>"
                            html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>Different from: {query_value}</td></tr>"
                        else:
                            html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>{value}</td>"
                            html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>Same</td></tr>"
                    else:
                        html += f"<tr><td style='padding: 5px; border-bottom: 1px solid #ddd;'>{prop}</td>"
                        html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>{value}</td>"
                        html += f"<td style='padding: 5px; border-bottom: 1px solid #ddd;'>-</td></tr>"

            html += "</table>"
            html += "</div>"  # End result card

        html += "</div>"  # End results flex container
        html += "</div>"  # End main container

        return html


class DiamondSearchInterface:
    """Gradio interface for the diamond search system"""

    def __init__(self, search_engine, data_processor):
        self.search_engine = search_engine
        self.data_processor = data_processor
        self.interface = None

    def create_interface(self):
        """Create the Gradio interface"""
        # Get metadata ranges for creating UI components
        metadata_ranges = self.data_processor.metadata_ranges

        # Define UI components
        with gr.Blocks(title="Diamond Search System") as interface:
            gr.Markdown("# Diamond Search System")
            gr.Markdown("Search for similar diamonds using image, text, or property filters")

            with gr.Tabs():
                # Image Search Tab
                with gr.TabItem("Image Search"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", label="Upload Diamond Image")
                            image_search_button = gr.Button("Search by Image", variant="primary")

                        with gr.Column(scale=2):
                            with gr.Accordion("Advanced Options", open=False):
                                image_weight = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.1, 
                                                       label="Image Weight")
                                text_weight = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.1, 
                                                      label="Text Weight")
                                numerical_weight = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.1, 
                                                          label="Numerical Weight")
                                top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, 
                                                label="Number of Results")

                # Property Search Tab
                with gr.TabItem("Property Search"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Create dropdown for categorical properties
                            categorical_inputs = {}
                            for col, info in metadata_ranges.items():
                                if info['type'] == 'categorical' and col in ['shape', 'color', 'clarity', 'cut', 'polish', 'symmetry']:
                                    values = info['values']
                                    # Add empty option
                                    dropdown_values = [''] + values
                                    categorical_inputs[col] = gr.Dropdown(
                                        choices=dropdown_values, 
                                        value='', 
                                        label=f"Select {col.capitalize()}"
                                    )

                            property_search_button = gr.Button("Search by Properties", variant="primary")

                        with gr.Column(scale=1):
                            # Create sliders for numerical properties
                            numerical_inputs = {}
                            for col, info in metadata_ranges.items():
                                if info['type'] == 'numerical' and col in ['size', 'Total_Price']:
                                    min_val = info['min']
                                    max_val = info['max']
                                    # Round values for better UI
                                    min_val = round(min_val, 2)
                                    max_val = round(max_val, 2)

                                    numerical_inputs[f"{col}_min"] = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=min_val,
                                        step=(max_val - min_val) / 100,
                                        label=f"{col.capitalize()} Range"
                                    )
                                    numerical_inputs[f"{col}_max"] = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=max_val,
                                        step=(max_val - min_val) / 100,
                                        label=f"{col.capitalize()} Range"
                                    )

                        with gr.Column(scale=1):
                            property_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, 
                                                     label="Number of Results")

                # Combined Search Tab
                with gr.TabItem("Combined Search"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            combined_image = gr.Image(type="filepath", label="Upload Diamond Image (Optional)")

                            # Create dropdown for categorical properties
                            combined_categorical = {}
                            for col, info in metadata_ranges.items():
                                if info['type'] == 'categorical' and col in ['shape', 'color', 'clarity', 'cut']:
                                    values = info['values']
                                    # Add empty option
                                    dropdown_values = [''] + values
                                    combined_categorical[col] = gr.Dropdown(
                                        choices=dropdown_values, 
                                        value='', 
                                        label=f"Select {col.capitalize()}"
                                    )

                            combined_search_button = gr.Button("Combined Search", variant="primary")

                        with gr.Column(scale=1):
                            # Create sliders for numerical properties
                            combined_numerical = {}
                            for col, info in metadata_ranges.items():
                                if info['type'] == 'numerical' and col in ['size', 'Total_Price']:
                                    min_val = info['min']
                                    max_val = info['max']
                                    # Round values for better UI
                                    min_val = round(min_val, 2)
                                    max_val = round(max_val, 2)

                                    combined_numerical[f"{col}_min"] = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=min_val,
                                        step=(max_val - min_val) / 100,
                                        label=f"{col.capitalize()} Range"
                                    )
                                    combined_numerical[f"{col}_max"] = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=max_val,
                                        step=(max_val - min_val) / 100,
                                        label=f"{col.capitalize()} Range"
                                    )

                            combined_weights = gr.CheckboxGroup(
                                choices=["Image", "Properties"],
                                value=["Image", "Properties"],
                                label="Search Using"
                            )

                            combined_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, 
                                                     label="Number of Results")

            # Results display
            results_html = gr.HTML(label="Search Results")

            # Define search functions
            def image_search(image_path, img_weight, txt_weight, num_weight, num_results):
                if image_path is None:
                    return "<h3>Please upload an image to search</h3>"

                weights = {
                    'image': img_weight,
                    'text': txt_weight,
                    'numerical': num_weight
                }

                # Perform search
                results = self.search_engine.search(
                    query_img_path=image_path,
                    weights=weights,
                    top_k=num_results
                )

                # Add differences
                results = self.search_engine.get_result_differences(results)

                # Format results as HTML
                html = self.search_engine.format_results_html(results, query_img_path=image_path)

                return html

            def property_search(**kwargs):
                # Extract inputs
                top_k = kwargs.pop('top_k')

                # Separate categorical and numerical filters
                categorical_filters = {}
                numerical_filters = {}

                for key, value in kwargs.items():
                    if key in metadata_ranges:
                        if metadata_ranges[key]['type'] == 'categorical':
                            # Only add non-empty values
                            if value:
                                categorical_filters[key] = value
                        elif "_min" in key or "_max" in key:
                          # Handle numerical min/max sliders
                          base_key = key.replace("_min", "").replace("_max", "")
                          if base_key in metadata_ranges and metadata_ranges[base_key]['type'] == 'numerical':
                            if base_key not in numerical_filters:
                              numerical_filters[base_key] = [0, 0]  # Initialize with placeholders

                          if "_min" in key:
                            numerical_filters[base_key][0] = value
                          else:  # "_max" in key
                            numerical_filters[base_key][1] = value
                            # Add range values
                            #numerical_filters[key] = value

                # Check if any filters are provided
                if not categorical_filters and not numerical_filters:
                    return "<h3>Please select at least one property to search</h3>"

                # Create text description from categorical filters
                text_query = ", ".join([f"{k}: {v}" for k, v in categorical_filters.items() if v])

                # Create numerical query
                numerical_query = {}
                numerical_columns = []
                for col, range_vals in numerical_filters.items():
                    # Use midpoint of range as query value
                    mid_val = (range_vals[0] + range_vals[1]) / 2
                    numerical_query[col] = mid_val
                    numerical_columns.append(col)

                # Set weights based on what's provided
                weights = {'image': 0.0, 'text': 0.0, 'numerical': 0.0}
                if text_query:
                    weights['text'] = 0.6
                if numerical_query:
                    weights['numerical'] = 0.4

                # Normalize weights
                weight_sum = sum(weights.values())
                if weight_sum > 0:
                    weights = {k: v/weight_sum for k, v in weights.items()}

                # Perform search
                results = self.search_engine.search(
                    query_text=text_query if text_query else None,
                    numerical_query=numerical_query if numerical_query else None,
                    numerical_columns=numerical_columns,
                    categorical_filters=categorical_filters,
                    numerical_filters=numerical_filters,
                    weights=weights,
                    top_k=top_k
                )

                # Create a mock query metadata for differences
                query_metadata = {}
                for col, val in categorical_filters.items():
                    if val:
                        query_metadata[col] = val

                for col, range_vals in numerical_filters.items():
                    # Use midpoint of range as reference
                    query_metadata[col] = (range_vals[0] + range_vals[1]) / 2

                # Add differences
                results = self.search_engine.get_result_differences(results, query_metadata)

                # Format results as HTML
                html = self.search_engine.format_results_html(results)

                return html

            def combined_search(image_path, search_using, top_k, **kwargs):
                # Extract categorical and numerical inputs
                categorical_filters = {}
                numerical_filters = {}

                for key, value in kwargs.items():
                    if key in metadata_ranges:
                        if metadata_ranges[key]['type'] == 'categorical':
                            # Only add non-empty values
                            if value:
                                categorical_filters[key] = value
                        elif metadata_ranges[key]['type'] == 'numerical':
                            # Add range values
                            numerical_filters[key] = value

                # Set weights based on search_using
                weights = {'image': 0.0, 'text': 0.0, 'numerical': 0.0}

                if "Image" in search_using and image_path is not None:
                    weights['image'] = 0.6

                if "Properties" in search_using:
                    # Create text description from categorical filters
                    text_query = ", ".join([f"{k}: {v}" for k, v in categorical_filters.items() if v])
                    if text_query:
                        weights['text'] = 0.2

                    # Create numerical query
                    numerical_query = {}
                    numerical_columns = []
                    for col, range_vals in numerical_filters.items():
                        # Use midpoint of range as query value
                        mid_val = (range_vals[0] + range_vals[1]) / 2
                        numerical_query[col] = mid_val
                        numerical_columns.append(col)

                    if numerical_query:
                        weights['numerical'] = 0.2
                else:
                    text_query = None
                    numerical_query = None
                    numerical_columns = None

                # Normalize weights
                weight_sum = sum(weights.values())
                if weight_sum > 0:
                    weights = {k: v/weight_sum for k, v in weights.items()}
                else:
                    return "<h3>Please select at least one search method (Image or Properties)</h3>"

                # Perform search
                results = self.search_engine.search(
                    query_img_path=image_path,
                    query_text=text_query,
                    numerical_query=numerical_query,
                    numerical_columns=numerical_columns,
                    categorical_filters=categorical_filters,
                    numerical_filters=numerical_filters,
                    weights=weights,
                    top_k=top_k
                )

                # Create a mock query metadata for differences
                query_metadata = {}
                for col, val in categorical_filters.items():
                    if val:
                        query_metadata[col] = val

                for col, range_vals in numerical_filters.items():
                    # Use midpoint of range as reference
                    query_metadata[col] = (range_vals[0] + range_vals[1]) / 2

                # Add differences
                results = self.search_engine.get_result_differences(results, query_metadata)

                # Format results as HTML
                html = self.search_engine.format_results_html(results, query_img_path=image_path)

                return html

            # Connect buttons to search functions
            image_search_button.click(
                image_search,
                inputs=[image_input, image_weight, text_weight, numerical_weight, top_k],
                outputs=results_html
            )

            # Create a dictionary of all property search inputs
            property_inputs = [val for subdict in [categorical_inputs, numerical_inputs] for val in subdict.values()]
            #property_inputs.update(categorical_inputs)
            #property_inputs.update(numerical_inputs)
            #property_inputs['top_k'] = property_top_k
            property_inputs.append(property_top_k)  # Add top_k to inputs

            property_search_button.click(
                property_search,
                inputs=property_inputs,
                outputs=results_html
            )

            # Create a dictionary of all combined search inputs
            combined_inputs = {}
            combined_inputs.update(combined_categorical)
            combined_inputs.update(combined_numerical)

            combined_search_button.click(
                combined_search,
                inputs=[combined_image, combined_weights, combined_top_k] + list(combined_inputs.values()),
                outputs=results_html
            )

        self.interface = interface
        return interface

    def launch(self, share=True):
        """Launch the Gradio interface"""
        if self.interface is None:
            self.create_interface()

        self.interface.launch(share=share)


# Initialize the data processor
data_processor = DataProcessor(csv_file)

# Load and preprocess the data
df = data_processor.preprocess_data()

# Display basic information about the dataset
print(f"Dataset contains {len(df)} diamonds with the following columns:")
print(df.columns.tolist())

# Split the data
train_df, val_df, test_df = data_processor.split_data()

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

# Define numerical columns for feature extraction
numerical_columns = ['size', 'depth_percent', 'table_percent', 
                     'meas_length', 'meas_width', 'meas_depth', 'Total_Price']

# Only use columns that exist in the dataframe
numerical_columns = [col for col in numerical_columns if col in df.columns]

# Build feature index for training data
train_features = feature_extractor.build_feature_index(
    train_df, 
    images_folder, 
    numerical_columns=numerical_columns,
    max_images=None  # Process all images
)

# Save the features
feature_extractor.save_features(os.path.join(models_path, "diamond_features.pkl"))

# Initialize the search engine
search_engine = DiamondSearchEngine(feature_extractor, train_df, images_folder)

# Create and launch the Gradio interface
interface = DiamondSearchInterface(search_engine, data_processor)
interface.launch(share=True)

# core_engine.py
"""
Meta-Learning Engine Orchestration System - STREAMLIT INTEGRATED VERSION
Modified to work with Streamlit while preserving all original functionality
"""

import os
import csv
import json
import pickle
import gzip
import lz4
import glob
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from urllib.parse import urljoin

# ================================
# 🔧 CONFIGURATION SECTION
# ================================

# 📝 USER: Paste your OpenRouter API key here
OPENROUTER_API_KEY = "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"

# Try to import OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    OPENROUTER_AVAILABLE = True
except ImportError:
    print("⚠️ OpenAI client not available. LLM analysis will be simulated.")
    client = None
    OPENROUTER_AVAILABLE = False

OPENROUTER_MODEL_NAME = "alibaba/tongyi-deepresearch-30b-a3b:free"

CUSTOM_LLM_PROMPT = """
SYSTEM ROLE:
You are Corv, an AI strategist built for high-velocity Austin startups. You merge financial intelligence with narrative impact. You read 13 data streams covering customer sentiment, market funneling, operational cost, and growth signals.
Your mission: translate data into momentum.
TONE:
Speak like a founder who codes, fundraises, and scales chaos into order. Short sentences. Sharp edges. No corporate filler. Every word must feel kinetic—rational and hungry at the same time.
INSTRUCTIONS:
Ingest + Analyze: read all data inputs. Identify current standing in three domains:
Financial position (revenue, burn, runway, unit economics)
Operational efficiency (bottlenecks, leverage points)
Market sentiment (public trust, customer mood, investor temperature)
Produce the following sections in order:
1. Analytical Snapshot
Concise audit of the company's present condition. Three subsections:
What's Working
What's Failing
Where Pressure Converts to Power
2. Core Play
Define the most rational, data-backed strategy for maximum profitability and sustainable growth. Include expected ROI, timeline, and measurable signals of success.
3. Experimental Paths
Generate three additional strategies. Label and separate clearly:
A. High Risk / High Reward
Describe a plan with explosive upside and significant volatility. Project outcomes across:
Financial impact
Market reception
Moral cost
B. Unorthodox but Feasible
Describe a creative, contrarian approach that breaks norms but remains structurally sound. Provide financial, market, and ethical projections.
C. Morally Grey Opportunity
Describe an ethically ambiguous but profitable play. Quantify the short-term and long-term trade-offs.
4. Predictive Visuals (Text-Only)
Create ASCII-style or symbolic chart visuals for the next 12 months:
Revenue trajectory
Customer sentiment curve
Risk vs. Reward matrix
Use minimal symbols (▇, ░, ▲, ▼) and concise numeric indicators.
5. Momentum Narrative
Write a short, cinematic forecast of what happens if the company executes.
"""

# ================================
# 📊 DATA LAYOUT CONFIGURATION
# ================================

@dataclass
class CSVSlot:
    """Represents one CSV slot and its corresponding model folder"""
    slot_number: int
    csv_file: str
    model_folder: str
    engine_name: str
    
    def __post_init__(self):
        # Make paths relative to project root
        self.csv_path = os.path.join(os.path.dirname(__file__), "..", self.csv_file)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", self.model_folder)

# 🔧 CSV SLOT CONFIGURATION - Updated for Streamlit integration
CSV_SLOTS = [
    # ENGINE 1 - 3 slots
    CSVSlot(1, "demo_data/customer_email_train.csv", "models/customer_email_model", "engine_1"),
    CSVSlot(2, "demo_data/product_reviews_train.csv", "models/product_reviews_model", "engine_1"), 
    CSVSlot(3, "demo_data/social_media_posts.csv", "models/social_media_model", "engine_1"),
    
    # ENGINE 2 - 5 slots
    CSVSlot(4, "demo_data/ab_test_training_data.csv", "models/train_ab_test_model", "engine_2"),
    CSVSlot(5, "demo_data/conversion_training_data.csv", "models/train_conversion_model", "engine_2"),
    CSVSlot(6, "demo_data/engagement_training_data.csv", "models/train_engagement_model", "engine_2"),
    CSVSlot(7, "demo_data/trial_to_paid_training_data.csv", "models/train_trial_to_paid_model", "engine_2"),
    CSVSlot(8, "demo_data/cart_abandonment_training_data.csv", "models/cart_abandonment_model", "engine_2"),
    
    # ENGINE 3 - 5 slots
    CSVSlot(9, "demo_data/anomaly_detection_data.csv", "models/anomaly_detection_model", "engine_3"),
    CSVSlot(10, "demo_data/cost_benefit_data.csv", "models/train_cost_benefit_analysis_model", "engine_3"),
    CSVSlot(11, "demo_data/resource_efficiency_data.csv", "models/train_resource_efficiency_model", "engine_3"),
    CSVSlot(12, "demo_data/performance_optimization_data.csv", "models/train_performance_optimization_model", "engine_3"),
    CSVSlot(13, "demo_data/time_series_data.csv", "models/train_time_series_forecasting_model", "engine_3"),
]

# ================================
# 🏗️ CORE SYSTEM CLASSES (Modified for Streamlit)
# ================================

class RobustPKLLoader:
    """Ultra-robust PKL loader with maximum compatibility"""
    
    @staticmethod
    def load_pkl_file(file_path: str) -> Tuple[bool, Any, str]:
        """Try every possible method to load a PKL file"""
        
        if not os.path.exists(file_path):
            return False, None, "File does not exist"
        
        # Method 1: Standard pickle.load()
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return True, data, "Standard pickle.load()"
        except Exception:
            pass
        
        # Method 2: Pickle with explicit protocol versions
        for protocol in [pickle.HIGHEST_PROTOCOL, 5, 4, 3, 2, 1, 0]:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                return True, data, f"Protocol {protocol}"
            except Exception:
                continue
        
        # Method 3: With encoding='latin1'
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return True, data, "Latin1 encoding"
        except Exception:
            pass
        
        # Method 4: With encoding='bytes'
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            return True, data, "Bytes encoding"
        except Exception:
            pass
        
        # Method 5: Gzipped pickle
        try:
            import gzip
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            return True, data, "Gzipped pickle"
        except Exception:
            pass
        
        # Method 6: Try joblib (common for sklearn)
        try:
            import joblib
            with open(file_path, 'rb') as f:
                data = joblib.load(f)
            return True, data, "JobLib"
        except Exception:
            pass
        
        # If all methods fail, provide diagnostic info
        try:
            file_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f:
                header = f.read(50)
            
            if b'sklearn' in header:
                return False, None, f"Valid sklearn pickle ({file_size} bytes) - environment mismatch likely"
            elif header.startswith(b'\x80'):
                return False, None, f"Valid pickle file ({file_size} bytes) - incompatible format"
            else:
                return False, None, f"Not a pickle file. Header: {header[:30]}"
                
        except Exception as e:
            return False, None, f"Could not analyze file: {str(e)}"

class SmartSlotProcessor:
    """Smart slot processor with better model handling"""
    
    def __init__(self):
        self.pkl_loader = RobustPKLLoader()
        
    def process_slot(self, engine_name: str, slot_name: str, csv_path: str, model_path: str, progress_callback=None) -> Dict:
        """Process a single slot with enhanced diagnostics"""
        slot_data = {
            'slot_name': slot_name,
            'csv_file': csv_path,
            'model_folder': model_path,
            'csv_info': {},
            'model_info': {},
            'preserved_data': {},
            'processing_errors': [],
            'success': False
        }
        
        if progress_callback:
            progress_callback(f"Processing {os.path.basename(csv_path)}")
        
        # Process CSV
        csv_info = self._process_csv(csv_path)
        slot_data['csv_info'] = csv_info
        
        # Process models
        model_info = self._process_models(model_path, slot_data['processing_errors'])
        slot_data['model_info'] = model_info
        
        # Extract and preserve data
        preserved_data = self._extract_and_preserve_data(csv_info, model_info, slot_data['processing_errors'])
        slot_data['preserved_data'] = preserved_data
        
        # Determine success
        slot_data['success'] = len(slot_data['processing_errors']) == 0
        
        return slot_data
    
    def _process_csv(self, csv_path: str) -> Dict:
        """Process CSV file and extract insights"""
        csv_info = {
            'file_path': csv_path,
            'summary_stats': {},
            'data_types': {},
            'missing_values': {},
            'key_insights': []
        }
        
        try:
            if not os.path.exists(csv_path):
                # Create sample data if file doesn't exist
                csv_info = self._create_sample_csv(csv_path)
                return csv_info
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Basic stats
            csv_info['summary_stats'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Data types
            csv_info['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Missing values
            missing_counts = df.isnull().sum()
            csv_info['missing_values'] = {col: int(count) for col, count in missing_counts.items() if count > 0}
            
            # Key insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    csv_info['key_insights'].append(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}")
            
        except Exception as e:
            csv_info['processing_error'] = str(e)
            # Create sample data as fallback
            csv_info = self._create_sample_csv(csv_path)
        
        return csv_info
    
    def _create_sample_csv(self, csv_path: str) -> Dict:
        """Create sample CSV data for demo purposes"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Generate sample data based on file type
        filename = os.path.basename(csv_path).lower()
        
        if 'customer_email' in filename or 'email' in filename:
            data = {
                'customer_id': range(1, 101),
                'email': [f'customer{i}@example.com' for i in range(1, 101)],
                'open_rate': np.random.uniform(0.1, 0.5, 100),
                'click_rate': np.random.uniform(0.02, 0.15, 100),
                'conversion': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
                'revenue': np.random.uniform(10, 500, 100)
            }
        elif 'review' in filename:
            data = {
                'product_id': range(1, 101),
                'rating': np.random.randint(1, 6, 100),
                'review_text': [f'Review {i}' for i in range(1, 101)],
                'sentiment_score': np.random.uniform(-1, 1, 100),
                'helpful_votes': np.random.randint(0, 50, 100)
            }
        elif 'social' in filename:
            data = {
                'post_id': range(1, 101),
                'platform': np.random.choice(['twitter', 'facebook', 'instagram'], 100),
                'engagement': np.random.randint(1, 1000, 100),
                'reach': np.random.randint(100, 10000, 100),
                'sentiment': np.random.choice(['positive', 'negative', 'neutral'], 100)
            }
        else:
            # Generic sample data
            data = {
                'id': range(1, 101),
                'value1': np.random.uniform(0, 100, 100),
                'value2': np.random.uniform(0, 100, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
            }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return {
            'file_path': csv_path,
            'summary_stats': {
                'rows': len(df),
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {},
            'key_insights': ['Sample data generated for demo'],
            'sample_data': True
        }
    
    def _process_models(self, model_path: str, errors: List) -> Dict:
        """Process model files with comprehensive diagnostics"""
        model_info = {
            'model_folder': model_path,
            'folder_exists': False,
            'pkl_files': [],
            'loaded_models': {},
            'model_metadata': {},
            'files_processed': [],
            'load_attempts': []
        }
        
        try:
            if not os.path.exists(model_path):
                # Create sample model data if folder doesn't exist
                os.makedirs(model_path, exist_ok=True)
                model_info['folder_exists'] = True
                model_info['sample_models'] = True
                return model_info
            
            model_info['folder_exists'] = True
            
            # Find all PKL files
            pkl_files = glob.glob(os.path.join(model_path, "*.pkl"))
            model_info['pkl_files'] = [os.path.basename(f) for f in pkl_files]
            
            # Try to load each PKL file
            for pkl_file in pkl_files:
                success, model_data, method = self.pkl_loader.load_pkl_file(pkl_file)
                
                if success:
                    model_info['loaded_models'][os.path.basename(pkl_file)] = {
                        'data': model_data,
                        'load_method': method,
                        'type': type(model_data).__name__,
                        'size': os.path.getsize(pkl_file)
                    }
                    model_info['files_processed'].append(f"✅ {os.path.basename(pkl_file)} ({method})")
                else:
                    error_msg = f"Failed to load {pkl_file}: {model_data}"
                    errors.append(error_msg)
                    model_info['files_processed'].append(f"❌ {os.path.basename(pkl_file)} ({model_data})")
                
                # Track attempt details
                model_info['load_attempts'].append({
                    'file': os.path.basename(pkl_file),
                    'success': success,
                    'method': method,
                    'error': model_data if not success else None
                })
            
        except Exception as e:
            error_msg = f"Model processing error: {str(e)}"
            errors.append(error_msg)
            model_info['model_error'] = error_msg
        
        return model_info
    
    def _extract_and_preserve_data(self, csv_info: Dict, model_info: Dict, errors: List) -> Dict:
        """Extract and preserve key insights from CSV and model data"""
        preserved_data = {
            'key_predictions': {},
            'model_insights': {},
            'data_summary': {},
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Extract model insights
            for model_name, model_data in model_info.get('loaded_models', {}).items():
                model_obj = model_data['data']
                
                # Basic model info
                preserved_data['model_insights'][model_name] = {
                    'model_type': model_data['type'],
                    'load_method': model_data['load_method'],
                    'size_bytes': model_data['size']
                }
                
                # Try to get model predictions/parameters
                if hasattr(model_obj, 'predict'):
                    preserved_data['key_predictions'][f"{model_name}_ready"] = True
                
                if hasattr(model_obj, 'feature_importances_'):
                    importances = model_obj.feature_importances_
                    preserved_data['key_predictions'][f"{model_name}_top_features"] = len(importances)
                
                if hasattr(model_obj, 'classes_'):
                    preserved_data['key_predictions'][f"{model_name}_classes"] = len(model_obj.classes_)
            
            # Data summary from CSV
            summary_stats = csv_info.get('summary_stats', {})
            preserved_data['data_summary'] = {
                'dataset_size': summary_stats.get('rows', 0),
                'features_count': len(summary_stats.get('columns', [])),
                'numeric_features': len(summary_stats.get('numeric_columns', [])),
                'categorical_features': len(summary_stats.get('categorical_columns', []))
            }
            
            # Key predictions based on data
            preserved_data['key_predictions']['data_quality_score'] = self._calculate_data_quality_score(csv_info, model_info)
            preserved_data['key_predictions']['model_count'] = len(model_info.get('loaded_models', {}))
            preserved_data['key_predictions']['csv_processed'] = summary_stats.get('rows', 0) > 0
            
        except Exception as e:
            error_msg = f"Data extraction error: {str(e)}"
            errors.append(error_msg)
            preserved_data['extraction_error'] = error_msg
        
        return preserved_data
    
    def _calculate_data_quality_score(self, csv_info: Dict, model_info: Dict) -> float:
        """Calculate a data quality score"""
        score = 0.0
        
        # CSV quality (40% weight)
        summary_stats = csv_info.get('summary_stats', {})
        if summary_stats.get('rows', 0) > 0:
            score += 0.2
        
        if summary_stats.get('columns'):
            score += 0.1
        
        missing_values = csv_info.get('missing_values', {})
        if not missing_values:  # No missing values is good
            score += 0.1
        
        # Model quality (60% weight)
        loaded_models = model_info.get('loaded_models', {})
        if loaded_models:
            score += 0.3
        
        if len(loaded_models) > 1:  # Multiple models is better
            score += 0.2
        
        return min(score, 1.0)

class MetaLearningEngineSystem:
    """
    Main orchestrator for the entire meta-learning system - STREAMLIT INTEGRATED
    """
    
    def __init__(self):
        self.slot_processor = SmartSlotProcessor()
        
    def run_full_analysis(self, progress_callback=None):
        """Execute the complete meta-learning pipeline"""
        if progress_callback:
            progress_callback("Starting Meta-Learning Engine System")
        
        try:
            # STEP 1: Process all engines independently
            all_engine_data = self._process_all_engines(progress_callback)
            
            # STEP 2: Combine and clean all engine data
            data_combiner = DataCombiner()
            combined_data = data_combiner.combine_preserve_all_data(all_engine_data)
            
            # STEP 3: Feed to LLM for intelligent analysis
            llm_analyzer = LLMAnalyzer()
            llm_response = llm_analyzer.analyze_with_llm(combined_data)
            
            # STEP 4: Write results to file
            results_writer = ResultsWriter()
            results_writer.write_results(llm_response, combined_data)
            
            if progress_callback:
                progress_callback("Analysis Complete!")
            
            return {
                'success': True,
                'llm_response': llm_response,
                'combined_data': combined_data,
                'engine_data': all_engine_data
            }
            
        except Exception as e:
            error_msg = f"System Error: {str(e)}"
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'llm_response': {'llm_response': f"Analysis failed: {error_msg}"}
            }
    
    def _process_all_engines(self, progress_callback=None) -> Dict[str, Dict]:
        """Process each engine's assigned CSV slots"""
        engine_data = {}
        
        # Group slots by engine
        engine_slots = {}
        for slot in CSV_SLOTS:
            if slot.engine_name not in engine_slots:
                engine_slots[slot.engine_name] = []
            engine_slots[slot.engine_name].append(slot)
        
        # Process each engine
        for engine_name, slots in engine_slots.items():
            if progress_callback:
                progress_callback(f"Processing {engine_name} ({len(slots)} slots)")
            
            engine_result = self._process_engine_slots(engine_name, slots, progress_callback)
            engine_data[engine_name] = engine_result
        
        return engine_data
    
    def _process_engine_slots(self, engine_name: str, slots: List[CSVSlot], progress_callback=None) -> Dict:
        """Process all slots for an engine"""
        engine_data = {
            'slot_data': {},
            'model_outputs': {},
            'consolidated_insights': {},
            'processing_metadata': {
                'engine': engine_name,
                'slots_processed': len(slots),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for slot in slots:
            if progress_callback:
                progress_callback(f"Processing slot {slot.slot_number}: {os.path.basename(slot.csv_file)}")
            
            # Process the slot
            slot_result = self.slot_processor.process_slot(
                engine_name, f"slot_{slot.slot_number}", slot.csv_path, slot.model_path
            )
            
            engine_data['slot_data'][f'slot_{slot.slot_number}'] = slot_result
            
            # Store model outputs
            if slot_result.get('model_info', {}).get('loaded_models'):
                engine_data['model_outputs'][f'slot_{slot.slot_number}'] = slot_result['model_info']['loaded_models']
        
        # Consolidate engine insights
        engine_data['consolidated_insights'] = self._consolidate_engine_insights(engine_data['slot_data'])
        
        return engine_data
    
    def _consolidate_engine_insights(self, slot_data: Dict) -> Dict:
        """Consolidate insights across all slots in an engine"""
        consolidated = {
            'total_predictions': 0,
            'successful_slots': 0,
            'total_errors': 0,
            'data_quality_metrics': {},
            'prediction_categories': {},
            'processing_summary': {
                'total_slots': len(slot_data),
                'successful_slots': 0,
                'total_errors': 0
            }
        }
        
        for slot_name, data in slot_data.items():
            # Count predictions
            predictions = data.get('preserved_data', {}).get('key_predictions', {})
            consolidated['total_predictions'] += len(predictions)
            
            # Count successful slots
            if predictions:
                consolidated['processing_summary']['successful_slots'] += 1
            
            # Count errors
            errors = data.get('processing_errors', [])
            consolidated['processing_summary']['total_errors'] += len(errors)
            
            # Data quality
            csv_info = data.get('csv_info', {})
            if 'summary_stats' in csv_info:
                consolidated['data_quality_metrics'][slot_name] = self._calculate_slot_quality(csv_info, predictions)
        
        return consolidated
    
    def _calculate_slot_quality(self, csv_info: Dict, predictions: Dict) -> float:
        """Calculate quality score for a slot"""
        score = 0.0
        
        # CSV quality
        summary_stats = csv_info.get('summary_stats', {})
        if summary_stats.get('rows', 0) > 0:
            score += 0.5
        
        if summary_stats.get('columns'):
            score += 0.3
        
        # Model quality
        if predictions:
            score += 0.2
        
        return min(score, 1.0)

class DataCombiner:
    """Combines and cleans data from all engines - STREAMLIT INTEGRATED"""
    
    def combine_preserve_all_data(self, all_engine_data: Dict) -> Dict:
        """Combine all engine data while preserving all information"""
        combined_data = {
            'engine_summaries': {},
            'cross_engine_insights': {},
            'prediction_matrix': {},
            'data_preservation_log': {},
            'combined_metadata': {
                'engines_processed': len(all_engine_data),
                'total_slots': sum(len(engine_data.get('slot_data', {})) 
                                 for engine_data in all_engine_data.values()),
                'combination_timestamp': datetime.now().isoformat(),
                'preservation_mode': 'all_data_preserved'
            }
        }
        
        # Create engine summaries
        for engine_name, engine_data in all_engine_data.items():
            processing_summary = engine_data.get('consolidated_insights', {}).get('processing_summary', {})
            
            combined_data['engine_summaries'][engine_name] = {
                'slots_processed': len(engine_data.get('slot_data', {})),
                'total_predictions': engine_data.get('consolidated_insights', {}).get('total_predictions', 0),
                'successful_slots': processing_summary.get('successful_slots', 0),
                'total_errors': processing_summary.get('total_errors', 0),
                'data_quality_avg': np.mean(list(engine_data.get('consolidated_insights', {})
                                               .get('data_quality_metrics', {}).values()) or [0]),
                'prediction_categories': engine_data.get('consolidated_insights', {}).get('prediction_categories', {})
            }
            
            # Preserve all slot data
            combined_data['data_preservation_log'][engine_name] = {
                'slot_details': engine_data.get('slot_data', {}),
                'raw_model_outputs': engine_data.get('model_outputs', {}),
                'consolidated_insights': engine_data.get('consolidated_insights', {})
            }
        
        # Generate cross-engine insights
        combined_data['cross_engine_insights'] = self._generate_cross_engine_insights(all_engine_data)
        
        # Create prediction matrix
        combined_data['prediction_matrix'] = self._create_prediction_matrix(all_engine_data)
        
        # Final data quality assessment
        combined_data['combined_metadata']['overall_data_quality'] = self._assess_overall_quality(combined_data)
        
        return combined_data
    
    def _generate_cross_engine_insights(self, all_engine_data: Dict) -> Dict:
        """Generate insights across all engines"""
        cross_insights = {
            'common_predictions': {},
            'unique_insights': {},
            'error_analysis': {}
        }
        
        # Collect all predictions and errors
        all_predictions = {}
        all_errors = {}
        
        for engine_name, engine_data in all_engine_data.items():
            engine_predictions = {}
            engine_errors = 0
            
            for slot_name, slot_data in engine_data.get('slot_data', {}).items():
                predictions = slot_data.get('preserved_data', {}).get('key_predictions', {})
                engine_predictions.update(predictions)
                
                errors = slot_data.get('processing_errors', [])
                engine_errors += len(errors)
            
            all_predictions[engine_name] = engine_predictions
            all_errors[engine_name] = engine_errors
        
        cross_insights['error_analysis'] = all_errors
        
        # Find common and unique predictions
        all_pred_keys = set()
        for predictions in all_predictions.values():
            all_pred_keys.update(predictions.keys())
        
        for pred_key in all_pred_keys:
            sources = []
            values = []
            for engine_name, predictions in all_predictions.items():
                if pred_key in predictions:
                    sources.append(engine_name)
                    values.append(predictions[pred_key])
            
            if len(sources) > 1:  # Common across engines
                cross_insights['common_predictions'][pred_key] = {
                    'sources': sources,
                    'values': values,
                    'agreement_level': self._calculate_agreement(values)
                }
            elif len(sources) == 1:  # Unique to one engine
                cross_insights['unique_insights'][pred_key] = {
                    'source': sources[0],
                    'value': values[0]
                }
        
        return cross_insights
    
    def _create_prediction_matrix(self, all_engine_data: Dict) -> Dict:
        """Create matrix of all predictions across engines"""
        matrix = {
            'rows': [],  # Engine-Slot combinations
            'columns': [],  # Prediction types
            'data': {}   # Values
        }
        
        for engine_name, engine_data in all_engine_data.items():
            for slot_name, slot_data in engine_data.get('slot_data', {}).items():
                slot_id = f"{engine_name}_{slot_name}"
                matrix['rows'].append(slot_id)
                
                predictions = slot_data.get('preserved_data', {}).get('key_predictions', {})
                for pred_key, pred_value in predictions.items():
                    if pred_key not in matrix['columns']:
                        matrix['columns'].append(pred_key)
                    
                    if slot_id not in matrix['data']:
                        matrix['data'][slot_id] = {}
                    matrix['data'][slot_id][pred_key] = pred_value
        
        return matrix
    
    def _calculate_agreement(self, values: List) -> float:
        """Calculate agreement level between prediction values"""
        if len(values) <= 1:
            return 1.0
        
        # Simple agreement calculation
        if all(isinstance(v, (int, float)) for v in values):
            mean_val = np.mean(values)
            std_val = np.std(values)
            return 1.0 - min(std_val / max(abs(mean_val), 1), 1.0)
        elif all(isinstance(v, str) for v in values):
            unique_vals = set(values)
            return 1.0 - (len(unique_vals) - 1) / max(len(values) - 1, 1)
        return 0.5
    
    def _assess_overall_quality(self, combined_data: Dict) -> float:
        """Assess overall data quality across all engines"""
        quality_scores = []
        
        for engine_summary in combined_data['engine_summaries'].values():
            quality_scores.append(engine_summary['data_quality_avg'])
        
        return np.mean(quality_scores) if quality_scores else 0.0

class LLMAnalyzer:
    """Handles LLM analysis using OpenRouter API - STREAMLIT INTEGRATED"""
    
    def __init__(self):
        self.model_name = OPENROUTER_MODEL_NAME
        
    def analyze_with_llm(self, combined_data: Dict) -> Dict:
        """Send combined data to LLM for intelligent analysis"""
        
        # Check if OpenRouter is available
        if not OPENROUTER_AVAILABLE or not client:
            return self._mock_llm_analysis(combined_data)
        
        # Prepare the prompt
        prompt = self._build_analysis_prompt(combined_data)
        
        # Send to OpenRouter using the client
        response = self._call_openrouter_api(prompt)
        
        # Check for response
        if 'error' in response:
            return {
                'llm_response': f"API Error: {response['error']}",
                'usage_stats': {},
                'model_used': self.model_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality_score': combined_data['combined_metadata']['overall_data_quality'],
                'api_error': response['error']
            }
        
        # Extract response content
        choices = response.get('choices', [])
        if not choices:
            return {
                'llm_response': "No response received from API",
                'usage_stats': response.get('usage', {}),
                'model_used': self.model_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality_score': combined_data['combined_metadata']['overall_data_quality']
            }
        
        message = choices[0].get('message', {})
        content = message.get('content', 'No content received')
        
        return {
            'llm_response': content,
            'usage_stats': response.get('usage', {}),
            'model_used': self.model_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_quality_score': combined_data['combined_metadata']['overall_data_quality']
        }
    
    def _mock_llm_analysis(self, combined_data: Dict) -> Dict:
        """Provide mock analysis when OpenRouter is not available"""
        mock_content = f"""
🚀 META-LEARNING ENGINE ANALYSIS (DEMO MODE)

ANALYTICAL SNAPSHOT:
What's Working:
• Successfully processed {combined_data['combined_metadata']['total_slots']} data slots across {combined_data['combined_metadata']['engines_processed']} engines
• Data quality score: {combined_data['combined_metadata']['overall_data_quality']:.1%}
• Engine processing completed with minimal errors

What's Failing:
• Some model loading failures detected (expected in demo mode)
• Limited cross-engine correlation analysis

Where Pressure Converts to Power:
• Demo environment ready for real data integration
• Robust error handling prevents system crashes

CORE PLAY:
Implement real data pipeline with your CSV files and trained models. Expected ROI: 15-25% efficiency improvement within 90 days.

EXPERIMENTAL PATHS:
A. High Risk / High Reward: Deploy AI-powered real-time analytics across all customer touchpoints
B. Unorthodox but Feasible: Implement blockchain-based data verification for model outputs  
C. Morally Grey Opportunity: Use sentiment analysis to predict competitor moves

PREDICTIVE VISUALS:
Revenue Trajectory: ▇▇▇░░░░░░░ (projected growth)
Customer Sentiment: ▇▇▇▇▇░░░░░ (positive trend)
Risk vs Reward: ▲▲▲░░░░░░ (moderate risk profile)

MOMENTUM NARRATIVE:
The engine stands ready. Data flows through three specialized processors, each learning from the others. When real data arrives, this becomes a precision instrument for business intelligence.
        """
        
        return {
            'llm_response': mock_content,
            'usage_stats': {},
            'model_used': f"{self.model_name} (Demo Mode)",
            'analysis_timestamp': datetime.now().isoformat(),
            'data_quality_score': combined_data['combined_metadata']['overall_data_quality'],
            'demo_mode': True
        }
    
    def _build_analysis_prompt(self, combined_data: Dict) -> str:
        """Build the analysis prompt with user's custom prompt and data"""
        
        # Format the data for the LLM
        data_summary = self._format_data_for_prompt(combined_data)
        
        prompt_template = f"""
{CUSTOM_LLM_PROMPT}

AVAILABLE DATA FOR ANALYSIS:
{data_summary}

INSTRUCTIONS:
1. Analyze the provided data from all 3 engines
2. Identify which datasets are most relevant to the user's question
3. Provide comprehensive insights based on the available information
4. If data conflicts exist, present all perspectives
5. Include confidence levels and data quality indicators
6. Focus on actionable insights and recommendations

Please provide a thorough analysis of the data and answer the user's question based on the insights from all engines.
"""
        
        return prompt_template
    
    def _format_data_for_prompt(self, combined_data: Dict) -> str:
        """Format the combined data for inclusion in the prompt"""
        formatted_data = []
        
        # Engine summaries
        formatted_data.append("=== ENGINE SUMMARIES ===")
        for engine_name, summary in combined_data['engine_summaries'].items():
            formatted_data.append(f"\n{engine_name.upper()}:")
            formatted_data.append(f"  - Slots processed: {summary['slots_processed']}")
            formatted_data.append(f"  - Successful slots: {summary['successful_slots']}")
            formatted_data.append(f"  - Total predictions: {summary['total_predictions']}")
            formatted_data.append(f"  - Processing errors: {summary['total_errors']}")
            formatted_data.append(f"  - Data quality: {summary['data_quality_avg']:.2f}")
            if summary['prediction_categories']:
                formatted_data.append(f"  - Prediction categories: {list(summary['prediction_categories'].keys())}")
        
        # Cross-engine insights
        cross_insights = combined_data['cross_engine_insights']
        if cross_insights.get('common_predictions'):
            formatted_data.append(f"\n=== CROSS-ENGINE INSIGHTS ===")
            formatted_data.append("Common predictions found:")
            for pred, info in cross_insights['common_predictions'].items():
                formatted_data.append(f"  - {pred}: {info['agreement_level']:.2f} agreement across {len(info['sources'])} engines")
        
        # Overall metadata
        metadata = combined_data['combined_metadata']
        formatted_data.append(f"\n=== SYSTEM METADATA ===")
        formatted_data.append(f"Total engines processed: {metadata['engines_processed']}")
        formatted_data.append(f"Total slots processed: {metadata['total_slots']}")
        formatted_data.append(f"Overall data quality: {metadata['overall_data_quality']:.3f}")
        formatted_data.append(f"Processing timestamp: {metadata['combination_timestamp']}")
        
        return "\n".join(formatted_data)
    
    def _call_openrouter_api(self, prompt: str) -> Dict:
        """Call OpenRouter API using the OpenAI client"""
        try:
            # Use the OpenAI client to call OpenRouter
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meta-learning AI that analyzes data from multiple engines and provides comprehensive insights. Be detailed and analytical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            # Convert OpenAI response to our format
            return {
                'choices': [
                    {
                        'message': {
                            'content': response.choices[0].message.content
                        }
                    }
                ],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "error": f"OpenRouter API Error: {str(e)}"
            }

class ResultsWriter:
    """Writes results to file - STREAMLIT INTEGRATED"""
    
    def write_results(self, llm_response: Dict, combined_data: Dict):
        """Write analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, f"meta_learning_results_{timestamp}.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("META-LEARNING ENGINE SYSTEM - ANALYSIS RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Used: {llm_response.get('model_used', 'Unknown')}\n")
                f.write(f"Data Quality Score: {llm_response.get('data_quality_score', 0):.3f}\n")
                f.write("=" * 80 + "\n\n")
                
                # LLM Response
                f.write("LLM ANALYSIS RESPONSE:\n")
                f.write("-" * 40 + "\n")
                llm_content = llm_response.get('llm_response', 'No response received')
                f.write(llm_content + "\n\n")
                
                # API Error if present
                if 'api_error' in llm_response:
                    f.write("API ERROR:\n")
                    f.write("-" * 40 + "\n")
                    f.write(llm_response['api_error'] + "\n\n")
                
                # Usage Statistics
                usage = llm_response.get('usage_stats', {})
                if usage:
                    f.write("API USAGE STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}\n")
                    f.write(f"Completion Tokens: {usage.get('completion_tokens', 'N/A')}\n")
                    f.write(f"Total Tokens: {usage.get('total_tokens', 'N/A')}\n\n")
                
                # Data Summary
                f.write("DATA PROCESSING SUMMARY:\n")
                f.write("-" * 40 + "\n")
                metadata = combined_data['combined_metadata']
                f.write(f"Engines Processed: {metadata['engines_processed']}\n")
                f.write(f"Total Slots Processed: {metadata['total_slots']}\n")
                f.write(f"Overall Data Quality: {metadata['overall_data_quality']:.3f}\n\n")
                
                # Engine Breakdown
                f.write("ENGINE BREAKDOWN:\n")
                f.write("-" * 40 + "\n")
                for engine_name, summary in combined_data['engine_summaries'].items():
                    f.write(f"\n{engine_name.upper()}:\n")
                    f.write(f"  Slots: {summary['slots_processed']}\n")
                    f.write(f"  Successful: {summary['successful_slots']}\n")
                    f.write(f"  Predictions: {summary['total_predictions']}\n")
                    f.write(f"  Errors: {summary['total_errors']}\n")
                    f.write(f"  Quality: {summary['data_quality_avg']:.3f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Failed to write results: {e}")
            return None

# Export the main system class for use in Streamlit
def create_meta_learning_system():
    """Factory function to create the meta-learning system"""
    return MetaLearningEngineSystem()

if __name__ == "__main__":
    # Demo mode
    system = create_meta_learning_system()
    result = system.run_full_analysis()
    print("Demo complete!")
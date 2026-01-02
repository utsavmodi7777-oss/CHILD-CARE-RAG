#!/usr/bin/env python3
"""
Optimized RAG Retrieval Evaluator - Clean version with detailed logging
GPT-4o-mini with core metrics and dual Cohere keys
"""

import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from dataclasses import dataclass, asdict
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class EvaluationConfig:
    name: str
    description: str
    use_query_expansion: bool = True
    expansion_count: int = 4
    use_hyde: bool = True
    retrieval_k: int = 20
    rrf_k: int = 60
    use_reranking: bool = True
    reranking_top_k: int = 5

@dataclass
class QueryResult:
    query_id: int
    config_name: str
    query_text: str
    ground_truth_chunks: List[str]
    retrieved_chunks: List[str]
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    execution_time: float
    openai_calls: int
    cohere_calls: int

@dataclass
class ConfigurationSummary:
    config_name: str
    total_queries: int
    avg_hit_at_1: float
    avg_hit_at_3: float
    avg_hit_at_5: float
    avg_mrr: float
    avg_precision_at_1: float
    avg_precision_at_3: float
    avg_precision_at_5: float
    total_execution_time: float
    total_openai_calls: int
    total_cohere_calls: int
    estimated_cost_gpt4o_mini: float

class VectorRetriever:
    """Milvus vector database retrieval wrapper"""
    
    def __init__(self):
        self.client = None
        self.collection_name = None
        self.setup_connection()
        
    def setup_connection(self):
        """Setup connection to Milvus"""
        try:
            print("Setting up vector database connection...")
            
            from pymilvus import MilvusClient
            from src.config.vector_config import COLLECTION_CONFIG
            from src.config.settings import Settings
            
            settings = Settings()
            print(f"Connecting to Zilliz Cloud: {settings.zilliz_uri}")
            
            self.client = MilvusClient(
                uri=settings.zilliz_uri,
                token=settings.zilliz_token
            )
            self.collection_name = COLLECTION_CONFIG["name"]
            
            print(f"Loading collection: {self.collection_name}")
            self.client.load_collection(self.collection_name)
            print("Vector database connection established successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to connect to vector database: {e}")
            raise
            
    def similarity_search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            # Import the client manager
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, parent_dir)
            from src.utils.client_manager import client_manager
            from src.config.settings import Settings
            settings = Settings()
            
            # Generate embedding for query using managed client
            embedding_client = client_manager.get_embedding_client()
            query_embedding = embedding_client.embed_query(query)
            
            # Search in Milvus
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["id", "content", "source_file"],
                search_params={"metric_type": "IP"}
            )
            
            # Convert to standardized format
            documents = []
            if search_results and len(search_results) > 0:
                for result in search_results[0]:
                    doc = {
                        'page_content': result["entity"]["content"],
                        'metadata': {
                            'id': result["entity"]["id"],
                            'source': result["entity"].get("source_file", ""),
                            'similarity_score': float(result["distance"])
                        }
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"ERROR in similarity search: {e}")
            return []

class CostTracker:
    """Track API costs for GPT-4o-mini"""
    
    def __init__(self):
        self.openai_calls = 0
        self.cohere_calls = 0
        # GPT-4o-mini pricing per 1K tokens
        self.gpt4o_mini_input_price = 0.00015
        self.gpt4o_mini_output_price = 0.0006
        self.avg_input_tokens = 150
        self.avg_output_tokens = 350
    
    def add_openai_call(self):
        self.openai_calls += 1
    
    def add_cohere_call(self):
        self.cohere_calls += 1
    
    def get_estimated_cost(self) -> float:
        cost_per_call = (
            (self.avg_input_tokens / 1000 * self.gpt4o_mini_input_price) +
            (self.avg_output_tokens / 1000 * self.gpt4o_mini_output_price)
        )
        return self.openai_calls * cost_per_call

class DualCohereRateLimiter:
    """Handle dual Cohere keys with rate limiting"""
    
    def __init__(self):
        self.requests_per_minute = 10
        self.min_interval = 60.0 / self.requests_per_minute
        self.key1_last_request = 0
        self.key2_last_request = 0
        self.key1_count = 0
        self.key2_count = 0
        self.current_key = 1
    
    def wait_and_use_key(self) -> int:
        """Wait if necessary and return the key to use"""
        current_time = time.time()
        
        # Check which keys are available
        key1_available = (current_time - self.key1_last_request) >= self.min_interval
        key2_available = (current_time - self.key2_last_request) >= self.min_interval
        
        if key1_available and key2_available:
            key_to_use = 1 if self.current_key == 2 else 2
        elif key1_available:
            key_to_use = 1
        elif key2_available:
            key_to_use = 2
        else:
            # Both unavailable, use the one that will be available sooner
            key1_wait = self.min_interval - (current_time - self.key1_last_request)
            key2_wait = self.min_interval - (current_time - self.key2_last_request)
            key_to_use = 1 if key1_wait <= key2_wait else 2
        
        # Wait if necessary
        if key_to_use == 1:
            time_since_last = current_time - self.key1_last_request
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"Rate limiting key 1: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
            self.key1_last_request = time.time()
            self.key1_count += 1
        else:
            time_since_last = current_time - self.key2_last_request
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"Rate limiting key 2: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
            self.key2_last_request = time.time()
            self.key2_count += 1
        
        self.current_key = key_to_use
        return key_to_use

class OptimizedRetrievalEvaluator:
    """Main evaluation class with core metrics"""
    
    def __init__(self, eval_data_path: str):
        self.eval_data_path = Path(eval_data_path)
        self.results_dir = Path("src/evaluation/results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("Initializing evaluator components...")
        
        # Load evaluation data
        with open(self.eval_data_path, 'r', encoding='utf-8') as f:
            self.eval_data = json.load(f)
        print(f"Loaded {len(self.eval_data)} evaluation questions")
        
        # Initialize components
        self.retriever = VectorRetriever()
        self.cost_tracker = CostTracker()
        self.rate_limiter = DualCohereRateLimiter()
        
        # Results storage
        self.query_results: List[QueryResult] = []
        self.config_summaries: List[ConfigurationSummary] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        print("Evaluator initialized successfully")
    
    def get_test_configurations(self) -> List[EvaluationConfig]:
        """Define test configurations"""
        configs = [
            EvaluationConfig(
                name="baseline_current",
                description="Current production configuration",
                use_query_expansion=True,
                expansion_count=4,
                use_hyde=True,
                retrieval_k=20,
                rrf_k=60,
                use_reranking=True,
                reranking_top_k=5
            ),
            EvaluationConfig(
                name="fast_baseline",
                description="Fast baseline without reranking",
                use_query_expansion=True,
                expansion_count=4,
                use_hyde=True,
                retrieval_k=20,
                rrf_k=60,
                use_reranking=False,
                reranking_top_k=20
            ),
            EvaluationConfig(
                name="minimal_processing",
                description="Minimal processing for speed",
                use_query_expansion=False,
                expansion_count=1,
                use_hyde=False,
                retrieval_k=15,
                rrf_k=60,
                use_reranking=False,
                reranking_top_k=15
            ),
            EvaluationConfig(
                name="expansion_only",
                description="Query expansion without HyDE",
                use_query_expansion=True,
                expansion_count=3,
                use_hyde=False,
                retrieval_k=20,
                rrf_k=60,
                use_reranking=True,
                reranking_top_k=5
            ),
            EvaluationConfig(
                name="hyde_only",
                description="HyDE without query expansion",
                use_query_expansion=False,
                expansion_count=1,
                use_hyde=True,
                retrieval_k=20,
                rrf_k=60,
                use_reranking=True,
                reranking_top_k=5
            )
        ]
        return configs
    
    def calculate_core_metrics(self, ground_truth: List[str], retrieved: List[str]) -> Dict[str, float]:
        """Calculate Hit@K, MRR, Precision@K"""
        metrics = {}
        
        # Hit@K
        for k in [1, 3, 5]:
            hit = 1.0 if any(chunk in ground_truth for chunk in retrieved[:k]) else 0.0
            metrics[f'hit_at_{k}'] = hit
        
        # Precision@K
        for k in [1, 3, 5]:
            relevant_in_k = sum(1 for chunk in retrieved[:k] if chunk in ground_truth)
            metrics[f'precision_at_{k}'] = relevant_in_k / k if k > 0 else 0.0
        
        # MRR
        mrr = 0.0
        for i, chunk in enumerate(retrieved):
            if chunk in ground_truth:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        return metrics
    
    def evaluate_single_query(self, query_data: Dict, config: EvaluationConfig, query_id: int) -> QueryResult:
        """Evaluate a single query"""
        start_time = time.time()
        openai_calls = 0
        cohere_calls = 0
        
        try:
            query = query_data['query']
            ground_truth = query_data['chunk_ids']
            
            # Track API calls based on configuration
            if config.use_query_expansion:
                openai_calls += 1
                self.cost_tracker.add_openai_call()
            
            if config.use_hyde:
                openai_calls += 1
                self.cost_tracker.add_openai_call()
            
            # Perform retrieval
            retrieved_docs = self.retriever.similarity_search(query, k=config.retrieval_k)
            retrieved_chunks = [doc['metadata']['id'] for doc in retrieved_docs]
            
            # Simulate reranking with dual key rate limiting
            if config.use_reranking and retrieved_docs:
                cohere_key = self.rate_limiter.wait_and_use_key()
                cohere_calls += 1
                self.cost_tracker.add_cohere_call()
                retrieved_chunks = retrieved_chunks[:config.reranking_top_k]
            
            # Calculate metrics
            metrics = self.calculate_core_metrics(ground_truth, retrieved_chunks)
            execution_time = time.time() - start_time
            
            result = QueryResult(
                query_id=query_id,
                config_name=config.name,
                query_text=query,
                ground_truth_chunks=ground_truth,
                retrieved_chunks=retrieved_chunks[:5],
                hit_at_1=metrics['hit_at_1'],
                hit_at_3=metrics['hit_at_3'],
                hit_at_5=metrics['hit_at_5'],
                mrr=metrics['mrr'],
                precision_at_1=metrics['precision_at_1'],
                precision_at_3=metrics['precision_at_3'],
                precision_at_5=metrics['precision_at_5'],
                execution_time=execution_time,
                openai_calls=openai_calls,
                cohere_calls=cohere_calls
            )
            
            return result
            
        except Exception as e:
            print(f"ERROR evaluating query {query_id}: {e}")
            return QueryResult(
                query_id=query_id,
                config_name=config.name,
                query_text=query_data['query'],
                ground_truth_chunks=query_data['chunk_ids'],
                retrieved_chunks=[],
                hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0,
                mrr=0.0,
                precision_at_1=0.0, precision_at_3=0.0, precision_at_5=0.0,
                execution_time=time.time() - start_time,
                openai_calls=openai_calls,
                cohere_calls=cohere_calls
            )
    
    def evaluate_configuration(self, config: EvaluationConfig) -> List[QueryResult]:
        """Evaluate all queries for a configuration"""
        print(f"\\nEvaluating configuration: {config.name}")
        print(f"Description: {config.description}")
        
        config_results = []
        
        for i, query_data in enumerate(self.eval_data):
            print(f"Query {i+1}/{len(self.eval_data)}: {query_data['query'][:50]}...")
            
            result = self.evaluate_single_query(query_data, config, i+1)
            config_results.append(result)
            
            if (i + 1) % 10 == 0:
                current_hit5 = sum(r.hit_at_5 for r in config_results) / len(config_results)
                current_mrr = sum(r.mrr for r in config_results) / len(config_results)
                print(f"Progress: {i+1}/{len(self.eval_data)} | Hit@5: {current_hit5:.3f} | MRR: {current_mrr:.3f}")
        
        return config_results
    
    def calculate_configuration_summary(self, results: List[QueryResult]) -> ConfigurationSummary:
        """Calculate summary statistics"""
        if not results:
            return None
        
        config_name = results[0].config_name
        total_queries = len(results)
        
        avg_metrics = {
            'hit_at_1': sum(r.hit_at_1 for r in results) / total_queries,
            'hit_at_3': sum(r.hit_at_3 for r in results) / total_queries,
            'hit_at_5': sum(r.hit_at_5 for r in results) / total_queries,
            'mrr': sum(r.mrr for r in results) / total_queries,
            'precision_at_1': sum(r.precision_at_1 for r in results) / total_queries,
            'precision_at_3': sum(r.precision_at_3 for r in results) / total_queries,
            'precision_at_5': sum(r.precision_at_5 for r in results) / total_queries,
        }
        
        total_execution_time = sum(r.execution_time for r in results)
        total_openai_calls = sum(r.openai_calls for r in results)
        total_cohere_calls = sum(r.cohere_calls for r in results)
        
        return ConfigurationSummary(
            config_name=config_name,
            total_queries=total_queries,
            avg_hit_at_1=avg_metrics['hit_at_1'],
            avg_hit_at_3=avg_metrics['hit_at_3'],
            avg_hit_at_5=avg_metrics['hit_at_5'],
            avg_mrr=avg_metrics['mrr'],
            avg_precision_at_1=avg_metrics['precision_at_1'],
            avg_precision_at_3=avg_metrics['precision_at_3'],
            avg_precision_at_5=avg_metrics['precision_at_5'],
            total_execution_time=total_execution_time,
            total_openai_calls=total_openai_calls,
            total_cohere_calls=total_cohere_calls,
            estimated_cost_gpt4o_mini=self.cost_tracker.get_estimated_cost()
        )
    
    def save_results_to_csv(self):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config_summaries:
            summary_df = pd.DataFrame([asdict(s) for s in self.config_summaries])
            summary_df.to_csv(self.results_dir / f"summary_results_{timestamp}.csv", index=False)
            print(f"Saved summary results: summary_results_{timestamp}.csv")
        
        if self.query_results:
            detailed_df = pd.DataFrame([asdict(r) for r in self.query_results])
            detailed_df.to_csv(self.results_dir / f"detailed_results_{timestamp}.csv", index=False)
            print(f"Saved detailed results: detailed_results_{timestamp}.csv")
        
        configs = self.get_test_configurations()
        config_df = pd.DataFrame([asdict(c) for c in configs])
        config_df.to_csv(self.results_dir / f"configurations_{timestamp}.csv", index=False)
        print(f"Saved configurations: configurations_{timestamp}.csv")
        
        return timestamp
    
    def generate_report(self, timestamp: str):
        """Generate markdown report"""
        sorted_configs = sorted(self.config_summaries, key=lambda x: x.avg_hit_at_5, reverse=True)
        
        report_content = f"""# RAG Retrieval Evaluation Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- Model: GPT-4o-mini
- Dataset: {len(self.eval_data)} questions
- Configurations: {len(self.config_summaries)}
- Total Cost: ${self.cost_tracker.get_estimated_cost():.3f}

## Performance Ranking

| Rank | Configuration | Hit@5 | Hit@3 | Hit@1 | MRR | Precision@5 |
|------|---------------|-------|-------|-------|-----|-------------|
"""
        
        for i, config in enumerate(sorted_configs, 1):
            report_content += f"| {i} | {config.config_name} | {config.avg_hit_at_5:.3f} | {config.avg_hit_at_3:.3f} | {config.avg_hit_at_1:.3f} | {config.avg_mrr:.3f} | {config.avg_precision_at_5:.3f} |\\n"
        
        report_content += f"""

## Best Configuration: {sorted_configs[0].config_name}
- Hit@5: {sorted_configs[0].avg_hit_at_5:.3f}
- MRR: {sorted_configs[0].avg_mrr:.3f}
- Cost: ${sorted_configs[0].estimated_cost_gpt4o_mini:.3f}

## Data Files
- summary_results_{timestamp}.csv
- detailed_results_{timestamp}.csv
- configurations_{timestamp}.csv
"""
        
        report_path = self.results_dir / f"EVALUATION_REPORT_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Generated report: EVALUATION_REPORT_{timestamp}.md")
        return report_path
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("Starting RAG Retrieval Evaluation")
        
        configs = self.get_test_configurations()
        fast_configs = [c for c in configs if not c.use_reranking]
        rerank_configs = [c for c in configs if c.use_reranking]
        
        print(f"Execution plan: {len(fast_configs)} fast + {len(rerank_configs)} reranking configs")
        
        start_time = time.time()
        
        try:
            # Phase 1: Fast configurations
            if fast_configs:
                print("\\nPhase 1: Fast configurations (no reranking)")
                for config in fast_configs:
                    config_results = self.evaluate_configuration(config)
                    self.query_results.extend(config_results)
                    
                    summary = self.calculate_configuration_summary(config_results)
                    if summary:
                        self.config_summaries.append(summary)
                    
                    print(f"Completed {config.name}: Hit@5={summary.avg_hit_at_5:.3f}, MRR={summary.avg_mrr:.3f}")
            
            # Phase 2: Reranking configurations
            if rerank_configs:
                print("\\nPhase 2: Reranking configurations (rate-limited)")
                for i, config in enumerate(rerank_configs, 1):
                    print(f"Configuration {i}/{len(rerank_configs)}: {config.name}")
                    
                    config_results = self.evaluate_configuration(config)
                    self.query_results.extend(config_results)
                    
                    summary = self.calculate_configuration_summary(config_results)
                    if summary:
                        self.config_summaries.append(summary)
                    
                    print(f"Completed {config.name}: Hit@5={summary.avg_hit_at_5:.3f}, MRR={summary.avg_mrr:.3f}")
                    print(f"Key usage: Key1={self.rate_limiter.key1_count}, Key2={self.rate_limiter.key2_count}")
            
            # Save results
            timestamp = self.save_results_to_csv()
            report_path = self.generate_report(timestamp)
            
            total_time = time.time() - start_time
            best_config = sorted(self.config_summaries, key=lambda x: x.avg_hit_at_5, reverse=True)[0]
            
            print(f"\\nEvaluation Complete!")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Total cost: ${self.cost_tracker.get_estimated_cost():.3f}")
            print(f"Best config: {best_config.config_name} (Hit@5: {best_config.avg_hit_at_5:.3f})")
            print(f"Report: {report_path}")
            
            return True
            
        except KeyboardInterrupt:
            print("\\nInterrupted - saving partial results...")
            if self.query_results:
                timestamp = self.save_results_to_csv()
                print(f"Partial results saved: {timestamp}")
            return False
            
        except Exception as e:
            print(f"ERROR: Evaluation failed: {e}")
            if self.query_results:
                timestamp = self.save_results_to_csv()
                print(f"Partial results saved: {timestamp}")
            return False

def main():
    """Main execution function"""
    print("RAG Retrieval Evaluator - Clean Version")
    print("="*50)
    
    eval_path = "src/evaluation/eval_set50.json"
    if not Path(eval_path).exists():
        print(f"ERROR: Evaluation data not found: {eval_path}")
        return False
    
    try:
        evaluator = OptimizedRetrievalEvaluator(eval_path)
        
        # Show estimates
        configs = evaluator.get_test_configurations()
        total_queries = len(evaluator.eval_data) * len(configs)
        estimated_openai_calls = sum(
            len(evaluator.eval_data) * (
                (1 if c.use_query_expansion else 0) + 
                (1 if c.use_hyde else 0)
            ) for c in configs
        )
        estimated_cohere_calls = sum(
            len(evaluator.eval_data) for c in configs if c.use_reranking
        )
        estimated_cost = estimated_openai_calls * 0.000233
        estimated_time_minutes = (estimated_cohere_calls * 3) / 60
        
        print(f"\\nEstimates:")
        print(f"Configurations: {len(configs)}")
        print(f"Total queries: {total_queries}")
        print(f"OpenAI calls: {estimated_openai_calls}")
        print(f"Cohere calls: {estimated_cohere_calls}")
        print(f"Estimated cost: ${estimated_cost:.3f}")
        print(f"Estimated time: {estimated_time_minutes:.0f} minutes")
        
        print(f"\\nStarting evaluation...")
        success = evaluator.run_evaluation()
        
        return success
        
    except Exception as e:
        print(f"ERROR: Failed to initialize evaluator: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

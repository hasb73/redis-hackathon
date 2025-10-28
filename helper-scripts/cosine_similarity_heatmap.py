#!/usr/bin/env python3
"""
Cosine Similarity Distribution Generator for HDFS Anomaly Detection

This script generates only the distribution of cosine similarities between log entries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class CosineSimilarityDistribution:
    def __init__(self, dataset_path=None, output_dir="similarity_analysis"):
        """
        Initialize the similarity distribution generator
        
        Args:
            dataset_path: Path to HDFS structured dataset
            output_dir: Directory to save output visualizations
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default dataset path if not provided
        if dataset_path is None:
            self.dataset_path = os.path.join(
                self.script_dir, 
                "./HDFS_dataset/parsed/HDFS.log_structured.csv"
            )
        else:
            self.dataset_path = dataset_path
            
        self.output_dir = os.path.join(self.script_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize embedding model
        print("Loading Sentence-BERT model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_data(self, sample_size=1000):
        """
        Load and prepare HDFS data for similarity analysis
        
        Args:
            sample_size: Number of log entries to analyze (for performance)
        """
        print(f"Loading HDFS data from {self.dataset_path}...")
        
        # Load structured logs
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(df)} log entries")
        else:
            print(f"Dataset not found at {self.dataset_path}")
            print("Please ensure the HDFS dataset is parsed and available.")
            return None, None
        
        # Load labels if available
        labels_df = None
        if os.path.exists(self.labels_path):
            labels_df = pd.read_csv(self.labels_path)
            print(f"Loaded {len(labels_df)} labels")
        
        # Sample data for analysis (full dataset may be too large)
        if len(df) > sample_size:
            print(f"Sampling {sample_size} entries for analysis...")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        # Prepare text for embedding
        if 'Content' in df_sample.columns:
            texts = df_sample['Content'].astype(str).tolist()
        elif 'Template' in df_sample.columns:
            texts = df_sample['Template'].astype(str).tolist()
        else:
            # Use first text column found
            text_cols = df_sample.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                texts = df_sample[text_cols[0]].astype(str).tolist()
            else:
                print("No text columns found in dataset")
                return None, None
        
        return df_sample, texts
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for log entries
        
        Args:
            texts: List of log text entries
            
        Returns:
            numpy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} log entries...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_cosine_similarity_heatmap(self, embeddings, labels=None, title="Cosine Similarity Heatmap"):
        """
        Create a cosine similarity heatmap
        
        Args:
            embeddings: Embedding vectors
            labels: Optional labels for anomaly indication
            title: Plot title
        """
        print("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        
        # Use seaborn for better aesthetics
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            similarity_matrix,
            mask=mask,
            annot=False,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            cbar_kws={"shrink": .8},
            vmin=-1,
            vmax=1
        )
        
        plt.title(f'{title}\n({embeddings.shape[0]} samples)', fontsize=16, pad=20)
        plt.xlabel('Log Entry Index', fontsize=12)
        plt.ylabel('Log Entry Index', fontsize=12)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Heatmap saved to: {output_path}")
        return similarity_matrix
    
    def create_interactive_heatmap(self, embeddings, texts, labels=None):
        """
        Create an interactive heatmap using Plotly
        
        Args:
            embeddings: Embedding vectors
            texts: Original log texts for hover information
            labels: Optional anomaly labels
        """
        print("Creating interactive cosine similarity heatmap...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Prepare hover text
        hover_text = []
        for i in range(len(texts)):
            row = []
            for j in range(len(texts)):
                hover_info = f"Entry {i} vs Entry {j}<br>"
                hover_info += f"Similarity: {similarity_matrix[i][j]:.3f}<br>"
                hover_info += f"Text {i}: {texts[i][:100]}...<br>"
                hover_info += f"Text {j}: {texts[j][:100]}..."
                row.append(hover_info)
            hover_text.append(row)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='RdYlBu_r',
            zmid=0,
            colorbar=dict(title="Cosine Similarity")
        ))
        
        fig.update_layout(
            title=f'Interactive Cosine Similarity Heatmap<br>({embeddings.shape[0]} HDFS Log Entries)',
            xaxis_title='Log Entry Index',
            yaxis_title='Log Entry Index',
            width=800,
            height=800
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, "interactive_similarity_heatmap.html")
        fig.write_html(output_path)
        print(f"Interactive heatmap saved to: {output_path}")
        
        return fig
    
    def analyze_anomaly_patterns(self, embeddings, labels_df=None):
        """
        Analyze similarity patterns between normal and anomalous log entries
        
        Args:
            embeddings: Embedding vectors
            labels_df: DataFrame with anomaly labels
        """
        if labels_df is None:
            print("No labels provided for anomaly pattern analysis")
            return
        
        print("Analyzing anomaly patterns in similarity space...")
        
        # Assume we can match some entries with labels (this is a simplified approach)
        # In practice, you'd need proper Block ID matching
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Full similarity matrix
        im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
        ax1.set_title('Full Cosine Similarity Matrix')
        ax1.set_xlabel('Log Entry Index')
        ax1.set_ylabel('Log Entry Index')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Similarity distribution
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        ax2.hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of Cosine Similarities')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Frequency')
        ax2.axvline(similarities.mean(), color='red', linestyle='--', 
                   label=f'Mean: {similarities.mean():.3f}')
        ax2.legend()
        
        # 3. PCA visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             alpha=0.6, c=range(len(embeddings_2d)), cmap='viridis')
        ax3.set_title(f'PCA Visualization of Embeddings\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.colorbar(scatter, ax=ax3)
        
        # 4. Similarity statistics
        stats_text = f"""Similarity Statistics:
        Mean: {similarities.mean():.4f}
        Std:  {similarities.std():.4f}
        Min:  {similarities.min():.4f}
        Max:  {similarities.max():.4f}
        
        High Similarity (>0.8): {(similarities > 0.8).sum()} pairs
        Low Similarity (<0.2): {(similarities < 0.2).sum()} pairs
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Similarity Statistics')
        
        plt.tight_layout()
        
        # Save analysis plot
        output_path = os.path.join(self.output_dir, "anomaly_pattern_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Anomaly pattern analysis saved to: {output_path}")
    
    def generate_similarity_report(self, embeddings, texts, similarity_matrix):
        """
        Generate a comprehensive similarity analysis report
        
        Args:
            embeddings: Embedding vectors
            texts: Original log texts
            similarity_matrix: Computed similarity matrix
        """
        print("Generating similarity analysis report...")
        
        # Find most and least similar pairs
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]
        indices = np.triu_indices_from(similarity_matrix, k=1)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)
        
        # Most similar pairs
        most_similar_pairs = []
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[-(i+1)]
            row_idx, col_idx = indices[0][idx], indices[1][idx]
            similarity = similarities[idx]
            most_similar_pairs.append({
                'similarity': similarity,
                'text1': texts[row_idx][:200],
                'text2': texts[col_idx][:200],
                'index1': row_idx,
                'index2': col_idx
            })
        
        # Least similar pairs
        least_similar_pairs = []
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            row_idx, col_idx = indices[0][idx], indices[1][idx]
            similarity = similarities[idx]
            least_similar_pairs.append({
                'similarity': similarity,
                'text1': texts[row_idx][:200],
                'text2': texts[col_idx][:200],
                'index1': row_idx,
                'index2': col_idx
            })
        
        # Generate report
        report = f"""
# Cosine Similarity Analysis Report
Generated on: {pd.Timestamp.now()}

## Dataset Statistics
- Total log entries analyzed: {len(texts)}
- Embedding dimensions: {embeddings.shape[1]}
- Total pairwise comparisons: {len(similarities)}

## Similarity Statistics
- Mean similarity: {similarities.mean():.4f}
- Standard deviation: {similarities.std():.4f}
- Minimum similarity: {similarities.min():.4f}
- Maximum similarity: {similarities.max():.4f}
- Median similarity: {np.median(similarities):.4f}

## Distribution Analysis
- High similarity pairs (>0.8): {(similarities > 0.8).sum()} ({(similarities > 0.8).mean()*100:.1f}%)
- Medium similarity pairs (0.3-0.8): {((similarities >= 0.3) & (similarities <= 0.8)).sum()} ({((similarities >= 0.3) & (similarities <= 0.8)).mean()*100:.1f}%)
- Low similarity pairs (<0.3): {(similarities < 0.3).sum()} ({(similarities < 0.3).mean()*100:.1f}%)

## Most Similar Log Entry Pairs
"""
        
        for i, pair in enumerate(most_similar_pairs):
            report += f"""
### Pair {i+1} (Similarity: {pair['similarity']:.4f})
**Entry {pair['index1']}:** {pair['text1']}
**Entry {pair['index2']}:** {pair['text2']}
"""
        
        report += "\n## Least Similar Log Entry Pairs\n"
        
        for i, pair in enumerate(least_similar_pairs):
            report += f"""
### Pair {i+1} (Similarity: {pair['similarity']:.4f})
**Entry {pair['index1']}:** {pair['text1']}
**Entry {pair['index2']}:** {pair['text2']}
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, "similarity_analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Similarity analysis report saved to: {report_path}")
    
    def run_complete_analysis(self, sample_size=100000):
        """
        Run complete cosine similarity analysis
        
        Args:
            sample_size: Number of log entries to analyze
        """
        print("=" * 60)
        print("HDFS Anomaly Detection - Cosine Similarity Analysis")
        print("=" * 60)
        
        # Load data
        df, texts = self.load_data(sample_size=sample_size)
        if df is None or texts is None:
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Load labels
        labels_df = None
        if os.path.exists(self.labels_path):
            labels_df = pd.read_csv(self.labels_path)
        
        # Create visualizations
        print("\n" + "="*40)
        print("Generating Visualizations")
        print("="*40)
        
        # 1. Static heatmap
        similarity_matrix = self.create_cosine_similarity_heatmap(
            embeddings, 
            title="HDFS Log Cosine Similarity"
        )
        
        # 2. Interactive heatmap
        self.create_interactive_heatmap(embeddings, texts)
        
        # 3. Anomaly pattern analysis
        self.analyze_anomaly_patterns(embeddings, labels_df)
        
        # 4. Generate report
        self.generate_similarity_report(embeddings, texts, similarity_matrix)
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Generated files:")
        print(f"   - Static heatmap: hdfs_log_cosine_similarity.png")
        print(f"   - Interactive heatmap: interactive_similarity_heatmap.html")
        print(f"   - Pattern analysis: anomaly_pattern_analysis.png")
        print(f"   - Detailed report: similarity_analysis_report.md")

def main():
    """Main execution function"""
    # Initialize generator
    generator = CosineSimilarityHeatmapGenerator()
    
    # Run complete analysis
    generator.run_complete_analysis(sample_size=10000)

if __name__ == "__main__":
    main()

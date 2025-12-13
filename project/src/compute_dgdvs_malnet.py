#!/usr/bin/env python3
"""
Compute Directed Graphlet Degree Vectors (DGDVs) for MalNet-Tiny dataset
Production pipeline using UCL Directed Graphlet Counter
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import networkx as nx
from tqdm import tqdm

from load_malnet import MalNetTinyLoader
from ucl_integration import compute_directed_gdvs_with_ucl

# Import GPU acceleration
try:
    from helpers.gpu_acceleration import corrcoef as gpu_corrcoef, print_gpu_status, get_gpu_info
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    gpu_corrcoef = None
    print_gpu_status = lambda: None
    get_gpu_info = lambda: {'available': False}

class DGDVProcessor:
    """
    Process MalNet-Tiny graphs to compute Directed Graphlet Degree Vectors (DGDVs)
    """
    
    def __init__(self, data_dir: str, output_dir: str, use_gpu: bool = True):
        """
        Initialize DGDV processor
        
        Args:
            data_dir: Directory containing MalNet-Tiny dataset
            output_dir: Directory to save computed DGDVs
            use_gpu: Whether to use GPU acceleration if available
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
        # Create subdirectories
        (self.output_dir / "gdvs").mkdir(exist_ok=True)
        (self.output_dir / "gcms").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Print GPU status if available
        if GPU_AVAILABLE and use_gpu:
            print_gpu_status()
    
    def process_graphs(self, 
                      graphs: List[nx.DiGraph],
                      labels: Optional[List[str]] = None,
                      max_graphlet_size: int = 5,
                      min_graphlet_size: int = 3,
                      limit: Optional[int] = None,
                      save_individual: bool = False,
                      batch_size: int = 100,
                      resume: bool = True) -> Dict:
        """
        Compute DGDVs for all graphs with incremental saving to prevent memory exhaustion
        
        Args:
            graphs: List of directed NetworkX graphs
            labels: Optional list of graph labels
            max_graphlet_size: Maximum graphlet size (3, 4, or 5)
            min_graphlet_size: Minimum graphlet size to compute (default: 3)
            limit: Limit number of graphs to process (for testing)
            save_individual: Save individual GDV files (default: False)
            batch_size: Number of graphs to process before saving checkpoint (default: 100)
            resume: If True, check for existing checkpoint and resume from there
        
        Returns:
            Dictionary with processing results and metadata
        """
        if limit:
            graphs = graphs[:limit]
            if labels:
                labels = labels[:limit]
        
        print(f"Processing {len(graphs)} graphs...")
        print(f"Graphlet size range: {min_graphlet_size} to {max_graphlet_size}")
        print(f"Output directory: {self.output_dir}")
        print(f"Batch size: {batch_size} (saving incrementally to prevent OOM)")
        
        # Validate graphlet size range
        if min_graphlet_size < 2 or max_graphlet_size > 4:
            print(f"\nWarning: UCL counter supports 2-4 node graphlets")
            print(f"  Requested: {min_graphlet_size}-{max_graphlet_size} node")
            min_graphlet_size = max(2, min(min_graphlet_size, 4))
            max_graphlet_size = min(4, max(max_graphlet_size, 2))
            print(f"  Using: {min_graphlet_size}-{max_graphlet_size} node")
        
        # Determine filenames
        if min_graphlet_size == max_graphlet_size:
            dgdvs_filename = f"all_dgdvs_{min_graphlet_size}node.pkl"
            gcms_filename = f"all_gcms_{min_graphlet_size}node.pkl"
            metadata_filename = f"processing_metadata_{min_graphlet_size}node.json"
        else:
            dgdvs_filename = f"all_dgdvs_{min_graphlet_size}_{max_graphlet_size}node.pkl"
            gcms_filename = f"all_gcms_{min_graphlet_size}_{max_graphlet_size}node.pkl"
            metadata_filename = f"processing_metadata_{min_graphlet_size}_{max_graphlet_size}node.json"
        
        dgdvs_file = self.output_dir / "gdvs" / dgdvs_filename
        gcms_file = self.output_dir / "gcms" / gcms_filename
        metadata_file = self.output_dir / "metadata" / metadata_filename
        
        # Use individual file storage to avoid memory accumulation
        individual_dir = self.output_dir / "gdvs" / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual GCM directory
        individual_gcm_dir = self.output_dir / "gcms" / "individual"
        individual_gcm_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing checkpoint if resuming
        start_idx = 0
        successful = []
        failed = []
        dgdv_shape = None
        
        if resume and metadata_file.exists():
            try:
                print(f"\nFound existing checkpoint: {metadata_file}")
                with open(metadata_file, 'r') as f:
                    checkpoint_metadata = json.load(f)
                    successful = checkpoint_metadata.get('successful_indices', [])
                    failed = checkpoint_metadata.get('failed_indices', [])
                    start_idx = len(successful) + len(failed)
                    dgdv_shape = checkpoint_metadata.get('dgdv_shape', None)
                
                print(f"  Resuming from graph {start_idx}/{len(graphs)}")
                print(f"  Already processed: {len(successful)} successful, {len(failed)} failed")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}")
                print("  Starting from beginning...")
                successful = []
                failed = []
                start_idx = 0
        
        # Process each graph - save immediately to disk, don't keep in memory
        for i in range(start_idx, len(graphs)):
            graph = graphs[i]
            try:
                # Compute DGDV using UCL counter
                dgdv = compute_directed_gdvs_with_ucl(
                    graph,
                    min_graphlet_size=min_graphlet_size,
                    max_graphlet_size=max_graphlet_size
                )
                
                if dgdv is None:
                    failed.append(i)
                    # Save metadata checkpoint
                    if (i + 1) % batch_size == 0:
                        self._save_metadata_checkpoint(successful, failed, metadata_file,
                                                      min_graphlet_size, max_graphlet_size,
                                                      len(graphs), labels, dgdv_shape)
                    continue
                
                # Save DGDV immediately to individual file (don't keep in memory)
                individual_file = individual_dir / f"graph_{i:05d}.pkl"
                with open(individual_file, 'wb') as f:
                    pickle.dump(dgdv, f)
                
                # Store shape from first successful graph
                if dgdv_shape is None:
                    dgdv_shape = dgdv.shape
                
                successful.append(i)
                
                # Clear from memory immediately
                del dgdv
                
                # Save metadata checkpoint periodically
                if (i + 1) % batch_size == 0 or (i + 1) == len(graphs):
                    self._save_metadata_checkpoint(successful, failed, metadata_file,
                                                  min_graphlet_size, max_graphlet_size,
                                                  len(graphs), labels, dgdv_shape)
                    print(f"\n  Checkpoint saved at graph {i+1}/{len(graphs)}")
                
            except MemoryError as e:
                print(f"\n[ERROR] Out of memory at graph {i}: {e}")
                print(f"  Saving checkpoint before terminating...")
                self._save_metadata_checkpoint(successful, failed, metadata_file,
                                              min_graphlet_size, max_graphlet_size,
                                              len(graphs), labels, dgdv_shape)
                raise
            except Exception as e:
                print(f"\nError processing graph {i}: {e}")
                failed.append(i)
                # Save checkpoint on error
                if (i + 1) % batch_size == 0:
                    self._save_metadata_checkpoint(successful, failed, metadata_file,
                                                  min_graphlet_size, max_graphlet_size,
                                                  len(graphs), labels, dgdv_shape)
                continue
        
        # Now combine individual files into final output (in batches to avoid OOM)
        # Only combine if we have new graphs to process
        if len(successful) > 0:
            print(f"\nCombining individual DGDV files into final output...")
            # Check if we need to combine (if output file doesn't exist or is incomplete)
            need_combine = not dgdvs_file.exists()
            if not need_combine:
                try:
                    with open(dgdvs_file, 'rb') as f:
                        existing_dgdvs = pickle.load(f)
                    need_combine = len(existing_dgdvs) < len(successful)
                except:
                    need_combine = True
            
            if need_combine:
                self._combine_individual_files(individual_dir, dgdvs_file, successful, batch_size=100)
            else:
                print(f"  Output file already contains all {len(successful)} DGDVs, skipping combine")
        
        # Compute and save GCMs incrementally from individual files
        print(f"\nComputing GCMs from saved DGDVs...")
        all_gcms = self._compute_gcms_from_files(individual_dir, gcms_file, individual_gcm_dir, successful, batch_size=50)
        
        # Final save
        print(f"\nFinalizing results...")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        # Save final metadata
        self._save_metadata_checkpoint(successful, failed, metadata_file,
                                      min_graphlet_size, max_graphlet_size,
                                      len(graphs), labels, dgdv_shape)
        
        if all_gcms:
            with open(gcms_file, 'wb') as f:
                pickle.dump(all_gcms, f)
            print(f"  Saved GCMs: {gcms_file}")
        
        print(f"  Saved DGDVs: {dgdvs_file}")
        print(f"  Saved metadata: {metadata_file}")
        
        # Return metadata only (don't load all DGDVs into memory)
        return {
            'dgdvs': None,  # Not loaded to save memory
            'gcms': all_gcms,
            'metadata': {
                'total_graphs': len(graphs),
                'successful': len(successful),
                'failed': len(failed),
                'failed_indices': failed,
                'successful_indices': successful,
                'min_graphlet_size': min_graphlet_size,
                'max_graphlet_size': max_graphlet_size,
                'dgdv_shape': dgdv_shape,
                'labels': labels if labels else None
            }
        }
    
    def _save_metadata_checkpoint(self, successful, failed, metadata_file,
                                  min_graphlet_size, max_graphlet_size, total_graphs, labels, dgdv_shape=None):
        """Save metadata checkpoint (without DGDVs to save memory)"""
        metadata = {
            'total_graphs': total_graphs,
            'successful': len(successful),
            'failed': len(failed),
            'successful_indices': successful,
            'failed_indices': failed,
            'min_graphlet_size': min_graphlet_size,
            'max_graphlet_size': max_graphlet_size,
            'dgdv_shape': dgdv_shape,
            'labels': labels if labels else None
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _combine_individual_files(self, individual_dir: Path, output_file: Path, 
                                  successful_indices: List[int], batch_size: int = 100):
        """Combine individual DGDV files into final output file using chunked approach"""
        import shutil
        
        # Use chunked approach: save to temporary chunk files, then merge without loading all at once
        temp_chunks_dir = output_file.parent / "temp_chunks"
        temp_chunks_dir.mkdir(exist_ok=True)
        chunk_files = []
        checkpoint_interval = 5  # Save chunk every 5 batches (500 DGDVs) to reduce memory
        
        try:
            # Process in batches and save to chunk files
            for batch_start in range(0, len(successful_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(successful_indices))
                batch_indices = successful_indices[batch_start:batch_end]
                
                batch_dgdvs = []
                for idx in batch_indices:
                    individual_file = individual_dir / f"graph_{idx:05d}.pkl"
                    if individual_file.exists():
                        with open(individual_file, 'rb') as f:
                            batch_dgdvs.append(pickle.load(f))
                
                # Progress update
                if batch_end % (batch_size * 5) == 0 or batch_end == len(successful_indices):
                    print(f"    Combined {batch_end}/{len(successful_indices)} DGDVs")
                
                # Save to chunk file periodically
                batch_num = batch_start // batch_size
                chunk_num = batch_num // checkpoint_interval
                chunk_file = temp_chunks_dir / f"chunk_{chunk_num:04d}.pkl"
                
                # Load existing chunk if it exists, otherwise start new
                if chunk_file.exists():
                    with open(chunk_file, 'rb') as f:
                        existing_chunk = pickle.load(f)
                    batch_dgdvs = existing_chunk + batch_dgdvs
                
                # Save chunk (overwrite or create)
                with open(chunk_file, 'wb') as f:
                    pickle.dump(batch_dgdvs, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Track chunk file if not already tracked
                if chunk_file not in chunk_files:
                    chunk_files.append(chunk_file)
                    print(f"    Saved chunk {len(chunk_files)} at {batch_end}/{len(successful_indices)} DGDVs")
                
                # Clear batch from memory immediately
                del batch_dgdvs
            
            # Merge chunks in stages to avoid loading all at once
            print(f"    Merging {len(chunk_files)} chunks into final file...")
            merge_group_size = 3  # Merge 3 chunks at a time
            
            # Stage 1: Merge chunks into intermediate groups
            intermediate_files = []
            sorted_chunks = sorted(chunk_files)
            
            for group_start in range(0, len(sorted_chunks), merge_group_size):
                group_end = min(group_start + merge_group_size, len(sorted_chunks))
                group_chunks = sorted_chunks[group_start:group_end]
                
                # Merge this group
                group_dgdvs = []
                for chunk_file in group_chunks:
                    with open(chunk_file, 'rb') as f:
                        chunk_dgdvs = pickle.load(f)
                        group_dgdvs.extend(chunk_dgdvs)
                        del chunk_dgdvs
                
                # Save intermediate merged file
                intermediate_file = temp_chunks_dir / f"merged_group_{group_start // merge_group_size:04d}.pkl"
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(group_dgdvs, f, protocol=pickle.HIGHEST_PROTOCOL)
                intermediate_files.append(intermediate_file)
                del group_dgdvs
                
                print(f"    Merged group {len(intermediate_files)} ({group_end}/{len(sorted_chunks)} chunks)")
            
            # Stage 2: Merge intermediate files into final file
            print(f"    Merging {len(intermediate_files)} groups into final file...")
            all_dgdvs = []
            for i, inter_file in enumerate(intermediate_files):
                with open(inter_file, 'rb') as f:
                    group_dgdvs = pickle.load(f)
                    all_dgdvs.extend(group_dgdvs)
                    del group_dgdvs
                
                if (i + 1) % 2 == 0 or (i + 1) == len(intermediate_files):
                    print(f"    Merged {i + 1}/{len(intermediate_files)} groups ({len(all_dgdvs)} DGDVs so far)...")
            
            # Write final combined file
            print(f"    Writing final file with {len(all_dgdvs)} DGDVs...")
            temp_file = output_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(all_dgdvs, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic move
            temp_file.replace(output_file)
            file_size_mb = output_file.stat().st_size / (1024**2)
            print(f"    Final file saved ({file_size_mb:.1f} MB)")
            
        finally:
            # Clean up chunk files
            if temp_chunks_dir.exists():
                shutil.rmtree(temp_chunks_dir)
                print(f"    Cleaned up temporary chunk files")
    
    def _compute_gcms_from_files(self, individual_dir: Path, gcms_file: Path,
                                individual_gcm_dir: Path, successful_indices: List[int], 
                                batch_size: int = 50) -> List[np.ndarray]:
        """Compute GCMs from individual DGDV files, saving both individual and combined files"""
        # Check for existing GCMs
        existing_gcms = []
        start_idx = 0
        
        if gcms_file.exists():
            try:
                with open(gcms_file, 'rb') as f:
                    existing_gcms = pickle.load(f)
                start_idx = len(existing_gcms)
                print(f"  Found {start_idx} existing GCMs, resuming from index {start_idx}")
            except Exception as e:
                print(f"  Warning: Could not load existing GCMs: {e}")
                existing_gcms = []
                start_idx = 0
        
        gcms = existing_gcms.copy()
        
        # Process remaining graphs in batches
        for batch_start in range(start_idx, len(successful_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(successful_indices))
            batch_indices = successful_indices[batch_start:batch_end]
            
            # Load batch of DGDVs
            batch_dgdvs = []
            for idx in batch_indices:
                individual_file = individual_dir / f"graph_{idx:05d}.pkl"
                if individual_file.exists():
                    with open(individual_file, 'rb') as f:
                        batch_dgdvs.append(pickle.load(f))
            
            # Compute GCMs for this batch using Spearman correlation
            from compute_gcms import compute_gcm_from_dgdv
            for i, (idx, dgdv) in enumerate(zip(batch_indices, batch_dgdvs)):
                # Use Spearman correlation implementation
                gcm = compute_gcm_from_dgdv(dgdv, use_gpu=False)
                gcms.append(gcm)
                
                # Save individual GCM file immediately
                individual_gcm_file = individual_gcm_dir / f"graph_{idx:05d}.pkl"
                with open(individual_gcm_file, 'wb') as f:
                    pickle.dump(gcm, f)
            
            # Clear batch from memory
            del batch_dgdvs
            
            # Save checkpoint periodically
            if batch_end % (batch_size * 2) == 0 or batch_end == len(successful_indices):
                with open(gcms_file, 'wb') as f:
                    pickle.dump(gcms, f)
                if batch_end % (batch_size * 5) == 0:
                    print(f"    GCM checkpoint: {batch_end}/{len(successful_indices)} (saved {batch_end} individual files)")
        
        print(f"  Saved {len(gcms)} individual GCM files to {individual_gcm_dir}")
        return gcms
    
    def _compute_gcms(self, dgdvs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute Graphlet Correlation Matrices (GCMs) from DGDVs using Spearman correlation
        Based on Yaveroglu et al. methodology
        
        Includes both 3-node and 4-node orbits in GCM computation
        """
        from compute_gcms import compute_gcm_from_dgdv
        
        gcms = []
        
        for dgdv in tqdm(dgdvs, desc="Computing GCMs", leave=False):
            # Use the Spearman correlation implementation from compute_gcms module
            gcm = compute_gcm_from_dgdv(dgdv, use_gpu=False)  # Spearman doesn't use GPU
            gcms.append(gcm)
        
        return gcms
    
    def _compute_gcms_incremental(self, dgdvs: List[np.ndarray], gcms_file: Path, batch_size: int = 50) -> List[np.ndarray]:
        """
        Compute GCMs incrementally, saving checkpoints to prevent data loss.
        If gcms_file exists, loads existing GCMs and continues from there.
        """
        # Check for existing GCMs
        existing_gcms = []
        start_idx = 0
        
        if gcms_file.exists():
            try:
                with open(gcms_file, 'rb') as f:
                    existing_gcms = pickle.load(f)
                start_idx = len(existing_gcms)
                print(f"  Found {start_idx} existing GCMs, resuming from index {start_idx}")
            except Exception as e:
                print(f"  Warning: Could not load existing GCMs: {e}")
                existing_gcms = []
                start_idx = 0
        
        # Compute remaining GCMs
        gcms = existing_gcms.copy()
        
        for i in range(start_idx, len(dgdvs)):
            dgdv = dgdvs[i]
            
            # Compute GCM for this DGDV
            if GPU_AVAILABLE and self.use_gpu and gpu_corrcoef is not None:
                try:
                    corr_matrix = gpu_corrcoef(dgdv, use_gpu=self.use_gpu)
                    if corr_matrix.size == 0:
                        gcms.append(np.array([]))
                    else:
                        n = corr_matrix.shape[0]
                        gcm = corr_matrix[np.triu_indices(n, k=1)]
                        gcms.append(gcm)
                except Exception as e:
                    # Fall back to CPU
                    non_zero_orbits = np.var(dgdv, axis=0) > 0
                    if non_zero_orbits.sum() == 0:
                        gcms.append(np.array([]))
                    else:
                        dgdv_filtered = dgdv[:, non_zero_orbits]
                        try:
                            corr_matrix = np.corrcoef(dgdv_filtered.T)
                            n = corr_matrix.shape[0]
                            gcm = corr_matrix[np.triu_indices(n, k=1)]
                            gcms.append(gcm)
                        except:
                            gcms.append(np.array([]))
            else:
                # CPU fallback
                non_zero_orbits = np.var(dgdv, axis=0) > 0
                if non_zero_orbits.sum() == 0:
                    gcms.append(np.array([]))
                else:
                    dgdv_filtered = dgdv[:, non_zero_orbits]
                    try:
                        corr_matrix = np.corrcoef(dgdv_filtered.T)
                        n = corr_matrix.shape[0]
                        gcm = corr_matrix[np.triu_indices(n, k=1)]
                        gcms.append(gcm)
                    except:
                        gcms.append(np.array([]))
            
            # Save checkpoint periodically
            if (i + 1) % batch_size == 0 or (i + 1) == len(dgdvs):
                with open(gcms_file, 'wb') as f:
                    pickle.dump(gcms, f)
                if (i + 1) % (batch_size * 2) == 0:
                    print(f"    GCM checkpoint: {i+1}/{len(dgdvs)}")
        
        return gcms


def main():
    """Main processing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute DGDVs for MalNet-Tiny')
    parser.add_argument('--data-dir', type=str, default='data/malnet_tiny',
                       help='Directory containing MalNet-Tiny dataset')
    parser.add_argument('--output-dir', type=str, default='data/gdvs',
                       help='Directory to save computed DGDVs')
    parser.add_argument('--max-size', type=int, default=3,
                       help='Maximum graphlet size (3 or 4)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of graphs to process (for testing)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual GDV files')
    parser.add_argument('--recompute-individual', action='store_true',
                       help='Recompute individual DGDV files from combined file')
    parser.add_argument('--split-gcm', action='store_true',
                       help='Split combined GCM file into individual GCM files')
    parser.add_argument('--combined-file', type=str, default='data/gdvs/gcms/all_gcms_3_4node.pkl',
                       help='Path to combined file (for --recompute-individual or --split-gcm)')
    parser.add_argument('--individual-dir', type=str, default=None,
                       help='Directory for individual files (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Handle split GCM option
    if args.split_gcm:
        if args.individual_dir is None:
            # Auto-detect individual GCM directory
            gcm_file_path = Path(args.combined_file)
            args.individual_dir = str(gcm_file_path.parent / "individual")
        split_gcm_file_into_individual(
            combined_file=args.combined_file,
            individual_dir=args.individual_dir,
            delete_existing=True
        )
        return
    
    # Handle recompute option
    if args.recompute_individual:
        if args.individual_dir is None:
            # Auto-detect individual DGDV directory
            dgdv_file_path = Path(args.combined_file)
            args.individual_dir = str(dgdv_file_path.parent / "individual")
        recompute_individual_dgdvs_from_file(
            combined_file=args.combined_file,
            individual_dir=args.individual_dir,
            delete_existing=True
        )
        return
    
    # Load dataset
    print("Loading MalNet-Tiny dataset...")
    loader = MalNetTinyLoader(args.data_dir)
    graphs = loader.load_graphs()
    
    if not graphs:
        print("[ERROR] No graphs loaded. Please check dataset directory.")
        return
    
    if args.limit:
        graphs = graphs[:args.limit]
        print(f"Limited to {len(graphs)} graphs for processing")
    
    # Process
    processor = DGDVProcessor(args.data_dir, args.output_dir)
    results = processor.process_graphs(
        graphs, 
        labels=loader.labels,
        max_graphlet_size=args.max_size,
        limit=args.limit,
        save_individual=args.save_individual
    )
    
    print("\n[SUCCESS] DGDV computation complete!")
    print(f"Results saved to: {args.output_dir}")


def split_gcm_file_into_individual(combined_file: str,
                                    individual_dir: str,
                                    delete_existing: bool = True) -> None:
    """
    Split a combined GCM file into individual GCM files
    
    Loads the combined GCM file and saves each GCM as an individual file.
    Optionally deletes existing individual files first.
    
    Args:
        combined_file: Path to combined GCM pickle file (list of GCM arrays)
        individual_dir: Directory to save individual GCM files
        delete_existing: If True, delete all existing individual files before saving new ones
    """
    print("="*60)
    print("Splitting Combined GCM File into Individual Files")
    print("="*60)
    
    combined_path = Path(combined_file)
    individual_path = Path(individual_dir)
    
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined GCM file not found: {combined_file}")
    
    # Create individual directory if it doesn't exist
    individual_path.mkdir(parents=True, exist_ok=True)
    
    # Delete existing individual files if requested
    if delete_existing:
        print(f"\n[1/3] Deleting existing individual GCM files in {individual_dir}...")
        existing_files = list(individual_path.glob("graph_*.pkl"))
        if existing_files:
            for file in existing_files:
                file.unlink()
            print(f"  Deleted {len(existing_files)} existing files")
        else:
            print("  No existing files to delete")
    else:
        print(f"\n[1/3] Keeping existing individual GCM files")
    
    # Load combined GCM file
    print(f"\n[2/3] Loading combined GCM file: {combined_file}")
    with open(combined_path, 'rb') as f:
        gcms = pickle.load(f)
    
    if not isinstance(gcms, list):
        raise ValueError(f"Expected list of GCMs, got {type(gcms)}")
    
    print(f"  Loaded {len(gcms)} GCMs")
    if len(gcms) > 0:
        valid_gcms = [gcm for gcm in gcms if gcm.size > 0]
        if valid_gcms:
            print(f"  Valid GCMs: {len(valid_gcms)}/{len(gcms)}")
            print(f"  Average GCM size: {np.mean([gcm.size for gcm in valid_gcms]):.1f}")
    
    # Save each GCM as an individual file
    print(f"\n[3/3] Saving individual GCM files to {individual_dir}...")
    saved_count = 0
    
    for i, gcm in enumerate(tqdm(gcms, desc="Saving individual files")):
        if gcm is not None and gcm.size > 0:
            individual_file = individual_path / f"graph_{i:05d}.pkl"
            with open(individual_file, 'wb') as f:
                pickle.dump(gcm, f)
            saved_count += 1
    
    print(f"\n  Saved {saved_count}/{len(gcms)} individual GCM files")
    print("="*60)
    print("GCM File Splitting Complete!")
    print("="*60)


def recompute_individual_dgdvs_from_file(combined_file: str,
                                         individual_dir: str,
                                         delete_existing: bool = True) -> None:
    """
    Recompute individual DGDV files from a combined DGDV file
    
    Loads the combined DGDV file and saves each DGDV as an individual file.
    Optionally deletes existing individual files first.
    
    Args:
        combined_file: Path to combined DGDV pickle file (list of DGDV matrices)
        individual_dir: Directory to save individual DGDV files
        delete_existing: If True, delete all existing individual files before saving new ones
    """
    print("="*60)
    print("Recomputing Individual DGDV Files")
    print("="*60)
    
    combined_path = Path(combined_file)
    individual_path = Path(individual_dir)
    
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined DGDV file not found: {combined_file}")
    
    # Create individual directory if it doesn't exist
    individual_path.mkdir(parents=True, exist_ok=True)
    
    # Delete existing individual files if requested
    if delete_existing:
        print(f"\n[1/3] Deleting existing individual DGDV files in {individual_dir}...")
        existing_files = list(individual_path.glob("graph_*.pkl"))
        if existing_files:
            for file in existing_files:
                file.unlink()
            print(f"  Deleted {len(existing_files)} existing files")
        else:
            print("  No existing files to delete")
    else:
        print(f"\n[1/3] Keeping existing individual DGDV files")
    
    # Load combined DGDV file
    print(f"\n[2/3] Loading combined DGDV file: {combined_file}")
    with open(combined_path, 'rb') as f:
        dgdvs = pickle.load(f)
    
    if not isinstance(dgdvs, list):
        raise ValueError(f"Expected list of DGDVs, got {type(dgdvs)}")
    
    print(f"  Loaded {len(dgdvs)} DGDVs")
    if len(dgdvs) > 0:
        print(f"  DGDV shape: {dgdvs[0].shape}")
    
    # Save each DGDV as an individual file
    print(f"\n[3/3] Saving individual DGDV files to {individual_dir}...")
    saved_count = 0
    
    for i, dgdv in enumerate(tqdm(dgdvs, desc="Saving individual files")):
        if dgdv is not None and dgdv.size > 0:
            individual_file = individual_path / f"graph_{i:05d}.pkl"
            with open(individual_file, 'wb') as f:
                pickle.dump(dgdv, f)
            saved_count += 1
    
    print(f"\n  Saved {saved_count}/{len(dgdvs)} individual DGDV files")
    print("="*60)
    print("Recomputation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


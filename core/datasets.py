#!/usr/bin/env python3
"""
Universal Dataset Manager for AutoML System
Creates and manages various datasets for testing and demonstration
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from sklearn.datasets import (
    load_iris, load_wine, load_diabetes, 
    fetch_california_housing, fetch_openml
)


# ========================================
#     Dataset Creation Functions
# ========================================

def create_iris_dataset(save_path: str = "data/iris.csv") -> pd.DataFrame:
    """
    Create Iris dataset for classification/regression
    
    Returns:
        DataFrame with Iris data
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Add a regression target (petal area)
    df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Iris dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(iris.feature_names)}")
    print(f"   Targets: species (classification), petal_area (regression)")
    
    return df


def create_wine_dataset(save_path: str = "data/wine.csv") -> pd.DataFrame:
    """
    Create Wine dataset
    
    Returns:
        DataFrame with Wine data
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['quality'] = wine.target
    
    # Add a regression target (alcohol content normalized)
    df['alcohol_score'] = df['alcohol'] / df['alcohol'].max() * 100
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Wine dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {len(wine.feature_names)} chemical properties")
    print(f"   Targets: quality (classification), alcohol_score (regression)")
    
    return df


def create_housing_dataset(save_path: str = "data/housing.csv") -> pd.DataFrame:
    """
    Create California Housing dataset
    
    Returns:
        DataFrame with housing data
    """
    housing = fetch_california_housing()
    
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['price'] = housing.target
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Housing dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(housing.feature_names)}")
    print(f"   Target: price (regression)")
    
    return df


def create_diabetes_dataset(save_path: str = "data/diabetes.csv") -> pd.DataFrame:
    """
    Create Diabetes dataset
    
    Returns:
        DataFrame with diabetes data
    """
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['progression'] = diabetes.target
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Diabetes dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(diabetes.feature_names)}")
    print(f"   Target: progression (regression)")
    
    return df


def create_synthetic_molecular_dataset(save_path: str = "data/synthetic_molecules.csv", 
                                      n_samples: int = 100) -> pd.DataFrame:
    """
    Create synthetic molecular dataset with fake SMILES and properties
    
    Returns:
        DataFrame with synthetic molecular data
    """
    # Generate fake SMILES (these are not real molecules!)
    np.random.seed(42)
    
    # Simple SMILES patterns
    patterns = [
        'CC', 'CCO', 'CC(C)O', 'CCC', 'CCCO', 'CC(C)CO',
        'c1ccccc1', 'c1ccc(O)cc1', 'c1ccc(C)cc1', 'c1ccc(CC)cc1',
        'CC(=O)O', 'CC(=O)C', 'CCC(=O)O', 'CC(C)(C)O',
        'C1CCCCC1', 'C1CCC(O)CC1', 'C1CCC(C)CC1'
    ]
    
    data = []
    for i in range(n_samples):
        # Random combinations of patterns
        smiles = np.random.choice(patterns, size=3, replace=True)
        
        # Generate synthetic properties based on "structure"
        base_wavelength = 400 + len(smiles[0]) * 10 + np.random.randn() * 20
        base_plqy = 0.5 + len(smiles[1]) * 0.02 + np.random.randn() * 0.1
        base_lifetime = 1.0 + len(smiles[2]) * 0.1 + np.random.randn() * 0.2
        
        data.append({
            'molecule_id': f'MOL_{i:04d}',
            'L1': smiles[0],
            'L2': smiles[1],
            'L3': smiles[2],
            'wavelength': max(350, min(700, base_wavelength)),
            'plqy': max(0, min(1, base_plqy)),
            'lifetime': max(0.1, base_lifetime),
            'temperature': np.random.choice([25, 50, 75, 100]),
            'solvent': np.random.choice(['water', 'ethanol', 'dmso', 'thf'])
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Synthetic molecular dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   SMILES columns: L1, L2, L3")
    print(f"   Targets: wavelength, plqy, lifetime")
    print(f"   Additional features: temperature, solvent")
    
    return df


def create_mixed_dataset(save_path: str = "data/mixed_features.csv", 
                        n_samples: int = 100) -> pd.DataFrame:
    """
    Create dataset with both molecular and tabular features
    
    Returns:
        DataFrame with mixed features
    """
    # Create base molecular data
    mol_df = create_synthetic_molecular_dataset("temp_mol.csv", n_samples)
    
    # Add tabular features
    mol_df['pressure'] = np.random.uniform(0.5, 2.0, n_samples)
    mol_df['ph'] = np.random.uniform(4, 10, n_samples)
    mol_df['concentration'] = np.random.uniform(0.001, 1.0, n_samples)
    mol_df['reaction_time'] = np.random.uniform(1, 24, n_samples)
    
    # Add categorical features
    mol_df['catalyst'] = np.random.choice(['Pd', 'Pt', 'Ru', 'None'], n_samples)
    mol_df['method'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Create composite target
    mol_df['yield'] = (
        mol_df['plqy'] * 50 + 
        mol_df['concentration'] * 20 + 
        mol_df['reaction_time'] * 0.5 + 
        np.random.randn(n_samples) * 5
    )
    mol_df['yield'] = mol_df['yield'].clip(0, 100)
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    mol_df.to_csv(save_path, index=False)
    
    print(f"✅ Mixed dataset saved to {save_path}")
    print(f"   Shape: {mol_df.shape}")
    print(f"   Molecular features: L1, L2, L3 (SMILES)")
    print(f"   Numerical features: pressure, ph, concentration, reaction_time")
    print(f"   Categorical features: catalyst, method, solvent")
    print(f"   Targets: wavelength, plqy, lifetime, yield")
    
    # Clean up temp file
    Path("temp_mol.csv").unlink(missing_ok=True)
    
    return mol_df


def create_mnist_dataset(save_path: str = "data/mnist.csv", 
                        n_samples: Optional[int] = None,
                        subset_for_demo: bool = False) -> pd.DataFrame:
    """
    Create MNIST dataset - full or subset
    
    Args:
        save_path: Path to save the CSV file
        n_samples: Number of samples (None = full dataset of 70,000)
        subset_for_demo: If True, creates small subset (default 1000) for quick demos
    
    Returns:
        DataFrame with MNIST data
    """
    # Determine sample size
    if subset_for_demo:
        n_samples = n_samples or 1000
        save_path = save_path.replace('mnist.csv', 'mnist_small.csv')
        print(f"📥 Creating MNIST subset ({n_samples} samples) for quick demo...")
    elif n_samples:
        print(f"📥 Creating MNIST dataset with {n_samples} samples...")
    else:
        print(f"📥 Downloading full MNIST dataset (70,000 samples)...")
        print("   ⏱️ This may take a few minutes on first download...")
    
    # Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=True)
    
    # Get data and target
    if n_samples and n_samples < len(mnist.data):
        # Take a subset
        indices = np.random.choice(len(mnist.data), n_samples, replace=False)
        X = mnist.data.iloc[indices]
        y = mnist.target.iloc[indices]
        dataset_type = "subset"
    else:
        # Use full dataset
        X = mnist.data
        y = mnist.target
        n_samples = len(X)
        dataset_type = "full"
    
    # Convert to DataFrame
    feature_names = [f'pixel_{i}' for i in range(784)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add target
    df['digit'] = y.astype(int)
    
    # Normalize pixel values to 0-1
    pixel_cols = [col for col in df.columns if col.startswith('pixel_')]
    df[pixel_cols] = df[pixel_cols] / 255.0
    
    # Add some derived features for regression tasks
    df['mean_intensity'] = df[pixel_cols].mean(axis=1)
    df['std_intensity'] = df[pixel_cols].std(axis=1)
    df['max_intensity'] = df[pixel_cols].max(axis=1)
    df['min_intensity'] = df[pixel_cols].min(axis=1)
    
    # Create a regression target (digit value * mean intensity)
    # Ensure no NaN values
    df['complexity_score'] = df['digit'].astype(float) * df['mean_intensity'].fillna(0.1) * 10
    df['complexity_score'] = df['complexity_score'].fillna(0)
    
    # Save to CSV
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ MNIST {dataset_type} dataset saved to {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: 784 pixels (28x28 image flattened)")
    print(f"   Classification target: digit (0-9)")
    print(f"   Regression target: complexity_score")
    print(f"   Dataset size: {n_samples:,} images")
    if dataset_type == "full":
        print(f"   💡 This is the complete MNIST dataset!")
    
    return df


# ========================================
#     Dataset Manager
# ========================================

class DatasetManager:
    """Manages dataset downloads with interactive menu"""
    
    def __init__(self):
        self.datasets = {
            '1': {
                'name': 'Iris',
                'description': 'Classic 4-feature dataset (150 samples)',
                'type': 'Tabular',
                'size': 'Small',
                'function': create_iris_dataset,
                'args': {}
            },
            '2': {
                'name': 'Wine',
                'description': 'Wine quality with 13 features (178 samples)',
                'type': 'Tabular',
                'size': 'Small',
                'function': create_wine_dataset,
                'args': {}
            },
            '3': {
                'name': 'Diabetes',
                'description': 'Diabetes progression (442 samples)',
                'type': 'Tabular',
                'size': 'Small',
                'function': create_diabetes_dataset,
                'args': {}
            },
            '4': {
                'name': 'Housing',
                'description': 'California housing prices (20,640 samples)',
                'type': 'Tabular',
                'size': 'Medium',
                'function': create_housing_dataset,
                'args': {}
            },
            '5': {
                'name': 'MNIST (Demo)',
                'description': 'Handwritten digits subset (1,000 samples)',
                'type': 'Image/Tabular',
                'size': 'Small',
                'function': create_mnist_dataset,
                'args': {'subset_for_demo': True, 'n_samples': 1000}
            },
            '6': {
                'name': 'MNIST (Full)',
                'description': 'Complete MNIST dataset (70,000 samples)',
                'type': 'Image/Tabular',
                'size': 'Large (~500MB)',
                'function': create_mnist_dataset,
                'args': {'n_samples': None, 'save_path': 'data/mnist_full.csv'}
            },
            '7': {
                'name': 'Synthetic Molecules',
                'description': 'Synthetic SMILES data (100 samples)',
                'type': 'Molecular',
                'size': 'Small',
                'function': create_synthetic_molecular_dataset,
                'args': {'n_samples': 100}
            },
            '8': {
                'name': 'Mixed Features',
                'description': 'Molecular + Tabular features (100 samples)',
                'type': 'Mixed',
                'size': 'Small',
                'function': create_mixed_dataset,
                'args': {'n_samples': 100}
            }
        }
        
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
    
    def show_menu(self):
        """Display interactive menu"""
        print("\n" + "="*60)
        print("📊 AutoML Dataset Manager")
        print("="*60)
        print("\nAvailable Datasets:")
        print("-"*60)
        
        for key, info in self.datasets.items():
            status = self._check_dataset_exists(info['name'])
            status_icon = "✅" if status else "⬇️"
            print(f"{key}. {status_icon} {info['name']:<20} - {info['description']}")
            print(f"      Type: {info['type']:<15} Size: {info['size']}")
        
        print("\nOptions:")
        print("A. Download ALL datasets (except MNIST Full)")
        print("Q. Download quick-start datasets (Iris, Wine, MNIST Demo)")
        print("M. Download all molecular datasets")
        print("T. Download all tabular datasets")
        print("C. Check existing datasets")
        print("R. Remove all datasets")
        print("X. Exit")
        print("-"*60)
    
    def _check_dataset_exists(self, name: str) -> bool:
        """Check if dataset already exists"""
        name_to_file = {
            'Iris': 'iris.csv',
            'Wine': 'wine.csv',
            'Diabetes': 'diabetes.csv',
            'Housing': 'housing.csv',
            'MNIST (Demo)': 'mnist_small.csv',
            'MNIST (Full)': 'mnist_full.csv',
            'Synthetic Molecules': 'synthetic_molecules.csv',
            'Mixed Features': 'mixed_features.csv'
        }
        
        file_path = self.data_dir / name_to_file.get(name, '')
        return file_path.exists()
    
    def download_dataset(self, key: str) -> bool:
        """Download a specific dataset"""
        if key not in self.datasets:
            print(f"❌ Invalid selection: {key}")
            return False
        
        info = self.datasets[key]
        
        # Check if already exists
        if self._check_dataset_exists(info['name']):
            response = input(f"⚠️ {info['name']} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Skipped.")
                return False
        
        print(f"\n📥 Downloading {info['name']}...")
        try:
            info['function'](**info['args'])
            print(f"✅ {info['name']} downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to download {info['name']}: {e}")
            return False
    
    def download_multiple(self, keys: List[str]):
        """Download multiple datasets"""
        success = 0
        failed = 0
        
        for key in keys:
            if self.download_dataset(key):
                success += 1
            else:
                failed += 1
        
        print(f"\n📊 Summary: {success} successful, {failed} failed/skipped")
    
    def download_all(self, include_large: bool = False):
        """Download all datasets"""
        keys = list(self.datasets.keys())
        if not include_large:
            # Exclude MNIST Full
            keys = [k for k in keys if k != '6']
        
        print(f"\n📦 Downloading {len(keys)} datasets...")
        self.download_multiple(keys)
    
    def download_quick_start(self):
        """Download essential datasets for quick start"""
        print("\n🚀 Downloading quick-start datasets...")
        # Iris, Wine, MNIST Demo
        self.download_multiple(['1', '2', '5'])
    
    def download_molecular(self):
        """Download molecular datasets"""
        print("\n🧪 Downloading molecular datasets...")
        # Synthetic Molecules, Mixed Features
        self.download_multiple(['7', '8'])
    
    def download_tabular(self):
        """Download tabular datasets"""
        print("\n📊 Downloading tabular datasets...")
        # Iris, Wine, Diabetes, Housing
        self.download_multiple(['1', '2', '3', '4'])
    
    def check_existing(self):
        """Check which datasets exist"""
        print("\n📁 Existing Datasets:")
        print("-"*40)
        
        existing = []
        missing = []
        
        for info in self.datasets.values():
            if self._check_dataset_exists(info['name']):
                existing.append(info['name'])
            else:
                missing.append(info['name'])
        
        if existing:
            print("✅ Downloaded:")
            for name in existing:
                file_map = {
                    'Iris': 'data/iris.csv',
                    'Wine': 'data/wine.csv',
                    'Diabetes': 'data/diabetes.csv',
                    'Housing': 'data/housing.csv',
                    'MNIST (Demo)': 'data/mnist_small.csv',
                    'MNIST (Full)': 'data/mnist_full.csv',
                    'Synthetic Molecules': 'data/synthetic_molecules.csv',
                    'Mixed Features': 'data/mixed_features.csv'
                }
                file_path = Path(file_map.get(name, ''))
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"   - {name:<20} ({size_mb:.1f} MB)")
        
        if missing:
            print("\n⬇️ Not downloaded:")
            for name in missing:
                print(f"   - {name}")
    
    def remove_all(self):
        """Remove all datasets"""
        response = input("⚠️ Remove ALL datasets? This cannot be undone. (yes/no): ")
        if response.lower() == 'yes':
            import shutil
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
                self.data_dir.mkdir()
            print("✅ All datasets removed.")
        else:
            print("Cancelled.")
    
    def run(self):
        """Run interactive menu"""
        while True:
            self.show_menu()
            choice = input("\nSelect option: ").strip().upper()
            
            if choice == 'X':
                print("\n👋 Goodbye!")
                break
            elif choice == 'A':
                self.download_all(include_large=False)
            elif choice == 'Q':
                self.download_quick_start()
            elif choice == 'M':
                self.download_molecular()
            elif choice == 'T':
                self.download_tabular()
            elif choice == 'C':
                self.check_existing()
            elif choice == 'R':
                self.remove_all()
            elif choice in self.datasets:
                self.download_dataset(choice)
            else:
                print(f"❌ Invalid option: {choice}")
            
            input("\nPress Enter to continue...")


# ========================================
#     Main Entry Point
# ========================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Manager for AutoML')
    parser.add_argument('--all', action='store_true', help='Download all datasets (except MNIST full)')
    parser.add_argument('--quick', action='store_true', help='Download quick-start datasets')
    parser.add_argument('--mnist-full', action='store_true', help='Download full MNIST (70,000 samples)')
    parser.add_argument('--molecular', action='store_true', help='Download molecular datasets')
    parser.add_argument('--tabular', action='store_true', help='Download tabular datasets')
    parser.add_argument('--check', action='store_true', help='Check existing datasets')
    parser.add_argument('--create-all', action='store_true', help='Create all datasets (non-interactive)')
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    # Handle command-line arguments
    if args.all:
        manager.download_all()
    elif args.quick:
        manager.download_quick_start()
    elif args.mnist_full:
        manager.download_dataset('6')
    elif args.molecular:
        manager.download_molecular()
    elif args.tabular:
        manager.download_tabular()
    elif args.check:
        manager.check_existing()
    elif args.create_all:
        # Non-interactive mode: create all datasets
        print("🔧 Creating all datasets...")
        print("-"*40)
        manager.download_all(include_large=False)
        print("\n✅ All datasets created successfully!")
        print("\n💡 Usage examples:")
        print("   python automl.py train data=data/iris.csv target=petal_area")
        print("   python automl.py train data=data/wine.csv target=alcohol_score")
        print("   python automl.py train data=data/synthetic_molecules.csv target=wavelength")
        print("   python automl.py train data=data/mixed_features.csv target=yield")
    else:
        # Interactive mode
        manager.run()


if __name__ == "__main__":
    main()
"""
Sentinel-2 Bathymetry Extraction

Practical implementation for extracting bathymetry from Sentinel-2 imagery
based on the methodologies from the research papers you provided.

This script provides the core functionality for:
1. Loading Sentinel-2 data
2. Preprocessing (sun glint correction, atmospheric correction)
3. Bathymetry estimation using established algorithms
4. Validation and accuracy assessment

Based on:
- Stumpf et al. (2003) methodology from the MDPI paper
- ESA COAS01 preprocessing pipeline
- Methods comparison studies
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings


class Sentinel2Bathymetry:
    """
    Main class for extracting bathymetry from Sentinel-2 imagery.
    
    Implements the complete workflow from raw Sentinel-2 data to bathymetry maps.
    """
    
    def __init__(self):
        """Initialize Sentinel-2 bathymetry processor."""
        self.bands = {}
        self.processed_bands = {}
        self.bathymetry = None
        self.water_mask = None
        self.transform = None
        self.crs = None
        
    def load_sentinel2_image(self, image_path: str, 
                           band_mapping: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        """
        Load Sentinel-2 image with specified bands.
        
        Parameters:
        -----------
        image_path : str
            Path to Sentinel-2 image file
        band_mapping : dict, optional
            Mapping of band names to band indices
            Default: {'B2': 1, 'B3': 2, 'B4': 3, 'B8': 4, 'B11': 5}
            
        Returns:
        --------
        dict
            Dictionary of loaded bands
        """
        if band_mapping is None:
            band_mapping = {
                'B2': 1,   # Blue (490nm)
                'B3': 2,   # Green (560nm) 
                'B4': 3,   # Red (665nm)
                'B8': 4,   # NIR (842nm)
                'B11': 5   # SWIR (1610nm)
            }
        
        print(f"Loading Sentinel-2 image: {image_path}")
        
        with rasterio.open(image_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            
            for band_name, band_idx in band_mapping.items():
                band_data = src.read(band_idx).astype(np.float32)
                
                # Convert to reflectance if needed (Sentinel-2 L2A is already in reflectance)
                # If values are > 1, assume they need scaling
                if np.max(band_data) > 1:
                    band_data = band_data / 10000.0  # Sentinel-2 scaling factor
                
                self.bands[band_name] = band_data
                
        print(f"Loaded bands: {list(self.bands.keys())}")
        return self.bands
    
    def preprocess_sentinel2(self, apply_sun_glint: bool = True, 
                           apply_atmospheric: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply preprocessing to Sentinel-2 bands following ESA COAS01 methodology.
        
        Parameters:
        -----------
        apply_sun_glint : bool, default=True
            Apply sun glint correction
        apply_atmospheric : bool, default=True
            Apply atmospheric correction (DOS)
            
        Returns:
        --------
        dict
            Preprocessed bands
        """
        print("Preprocessing Sentinel-2 data...")
        
        processed = self.bands.copy()
        
        # Step 1: Sun glint correction (ESA COAS01 coefficients)
        if apply_sun_glint and all(band in self.bands for band in ['B2', 'B3', 'B4', 'B8']):
            print("  - Applying sun glint correction...")
            
            # ESA COAS01 coefficients
            glint_coefficients = {
                'B2': 0.91574,
                'B3': 1.00116,
                'B4': 1.0223
            }
            
            nir_offset = 0.001
            
            for band_name in ['B2', 'B3', 'B4']:
                if band_name in processed:
                    processed[f'{band_name}_deglint'] = (
                        processed[band_name] - 
                        glint_coefficients[band_name] * (processed['B8'] - nir_offset)
                    )
        
        # Step 2: Atmospheric correction - Dark Object Subtraction
        if apply_atmospheric:
            print("  - Applying atmospheric correction (DOS)...")
            
            # ESA COAS01 DOS values
            dos_values = {
                'B2_deglint': 0.005,
                'B3_deglint': 0.001
            }
            
            for band_name, dos_value in dos_values.items():
                if band_name in processed:
                    processed[f'{band_name.replace("_deglint", "")}_DOS'] = np.maximum(
                        processed[band_name] - dos_value, 0
                    )
        
        # Step 3: Create water mask
        print("  - Creating water mask...")
        self.water_mask = self.create_water_mask(processed)
        
        self.processed_bands = processed
        print("  - Preprocessing complete!")
        
        return processed
    
    def create_water_mask(self, bands: Dict[str, np.ndarray], 
                         ndwi_threshold: float = 0.2,
                         mndwi_threshold: float = 0.2) -> np.ndarray:
        """
        Create water mask using spectral indices.
        
        Parameters:
        -----------
        bands : dict
            Dictionary of bands
        ndwi_threshold : float, default=0.2
            NDWI threshold for water detection
        mndwi_threshold : float, default=0.2
            MNDWI threshold for water detection
            
        Returns:
        --------
        np.ndarray
            Boolean water mask (True = water)
        """
        epsilon = 1e-10
        
        # Use original bands for water masking
        green = bands.get('B3', bands.get('B3_deglint', np.zeros_like(list(bands.values())[0])))
        nir = bands.get('B8', np.zeros_like(green))
        swir = bands.get('B11', np.zeros_like(green))
        
        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + epsilon)
        
        # MNDWI = (Green - SWIR) / (Green + SWIR)
        if 'B11' in bands:
            mndwi = (green - swir) / (green + swir + epsilon)
            water_mask = (ndwi > ndwi_threshold) | (mndwi > mndwi_threshold)
        else:
            water_mask = ndwi > ndwi_threshold
        
        return water_mask
    
    def extract_bathymetry_stumpf(self, depth_samples: np.ndarray, 
                                sample_coordinates: np.ndarray,
                                n_constant: float = 1000.0,
                                use_quadratic: bool = False) -> Dict:
        """
        Extract bathymetry using Stumpf et al. (2003) algorithm.
        
        This follows the exact methodology from the MDPI paper you provided.
        
        Parameters:
        -----------
        depth_samples : np.ndarray
            Known depth values at calibration points
        sample_coordinates : np.ndarray
            Pixel coordinates of calibration points (row, col)
        n_constant : float, default=1000.0
            Scaling constant for logarithm
        use_quadratic : bool, default=False
            Use quadratic regression (as in WorldView-2 paper)
            
        Returns:
        --------
        dict
            Bathymetry results and statistics
        """
        print("Extracting bathymetry using Stumpf algorithm...")
        
        # Get preprocessed blue and green bands
        if 'B2_DOS' in self.processed_bands and 'B3_DOS' in self.processed_bands:
            blue_band = self.processed_bands['B2_DOS']
            green_band = self.processed_bands['B3_DOS']
            print("  - Using DOS-corrected bands")
        elif 'B2_deglint' in self.processed_bands and 'B3_deglint' in self.processed_bands:
            blue_band = self.processed_bands['B2_deglint']
            green_band = self.processed_bands['B3_deglint']
            print("  - Using deglinted bands")
        else:
            blue_band = self.bands['B2']
            green_band = self.bands['B3']
            print("  - Using original bands")
        
        # Calculate band ratio (Stumpf method)
        epsilon = 1e-10
        blue_safe = np.maximum(blue_band, epsilon)
        green_safe = np.maximum(green_band, epsilon)
        
        # pSDB = ln(n * B_blue) / ln(n * B_green)
        ratio_layer = np.log(n_constant * blue_safe) / np.log(n_constant * green_safe)
        
        # Apply water mask
        ratio_layer = np.where(self.water_mask, ratio_layer, np.nan)
        
        # Extract ratio values at calibration points
        if sample_coordinates.shape[1] == 2:
            rows, cols = sample_coordinates[:, 0].astype(int), sample_coordinates[:, 1].astype(int)
            ratio_samples = ratio_layer[rows, cols]
        else:
            raise ValueError("Sample coordinates must be (N, 2) array of (row, col)")
        
        # Remove invalid samples
        valid_mask = ~(np.isnan(ratio_samples) | np.isnan(depth_samples))
        ratio_clean = ratio_samples[valid_mask]
        depth_clean = depth_samples[valid_mask]
        
        if len(ratio_clean) < 3:
            raise ValueError("Insufficient valid calibration samples")
        
        print(f"  - Using {len(ratio_clean)} calibration samples")
        
        # Fit regression model
        if use_quadratic:
            # Quadratic regression (as mentioned in WorldView-2 paper)
            from sklearn.preprocessing import PolynomialFeatures
            poly_features = PolynomialFeatures(degree=2)
            X = poly_features.fit_transform(ratio_clean.reshape(-1, 1))
            model = LinearRegression()
            model.fit(X, depth_clean)
            
            # Predict for entire image
            ratio_flat = ratio_layer.flatten()
            valid_pixels = ~np.isnan(ratio_flat)
            
            bathymetry_flat = np.full_like(ratio_flat, np.nan)
            if np.any(valid_pixels):
                X_pred = poly_features.transform(ratio_flat[valid_pixels].reshape(-1, 1))
                bathymetry_flat[valid_pixels] = model.predict(X_pred)
            
            bathymetry_map = bathymetry_flat.reshape(ratio_layer.shape)
            
        else:
            # Linear regression (standard Stumpf)
            model = LinearRegression()
            X = ratio_clean.reshape(-1, 1)
            model.fit(X, depth_clean)
            
            # Predict for entire image
            bathymetry_map = model.predict(ratio_layer.reshape(-1, 1)).reshape(ratio_layer.shape)
        
        # Apply water mask to final result
        bathymetry_map = np.where(self.water_mask, bathymetry_map, np.nan)
        
        # Calculate statistics
        y_pred = model.predict(X if not use_quadratic else X)
        r2 = r2_score(depth_clean, y_pred)
        rmse = np.sqrt(np.mean((depth_clean - y_pred) ** 2))
        mae = mean_absolute_error(depth_clean, y_pred)
        
        # Store results
        self.bathymetry = bathymetry_map
        
        results = {
            'bathymetry_map': bathymetry_map,
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(depth_clean),
            'method': 'Stumpf (quadratic)' if use_quadratic else 'Stumpf (linear)',
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
        }
        
        print(f"  - Model fit: R² = {r2:.3f}, RMSE = {rmse:.2f}m, MAE = {mae:.2f}m")
        
        return results
    
    def validate_bathymetry(self, validation_depths: np.ndarray,
                          validation_coordinates: np.ndarray) -> Dict:
        """
        Validate bathymetry results using independent validation data.
        
        Parameters:
        -----------
        validation_depths : np.ndarray
            Known depth values for validation
        validation_coordinates : np.ndarray
            Pixel coordinates of validation points
            
        Returns:
        --------
        dict
            Validation statistics
        """
        if self.bathymetry is None:
            raise ValueError("No bathymetry map available. Run extraction first.")
        
        print("Validating bathymetry results...")
        
        # Extract predicted depths at validation locations
        if validation_coordinates.shape[1] == 2:
            rows, cols = validation_coordinates[:, 0].astype(int), validation_coordinates[:, 1].astype(int)
            predicted_depths = self.bathymetry[rows, cols]
        else:
            raise ValueError("Validation coordinates must be (N, 2) array")
        
        # Remove invalid predictions
        valid_mask = ~(np.isnan(predicted_depths) | np.isnan(validation_depths))
        predicted_clean = predicted_depths[valid_mask]
        observed_clean = validation_depths[valid_mask]
        
        if len(predicted_clean) < 3:
            warnings.warn("Insufficient valid validation samples")
            return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n_samples': 0}
        
        # Calculate validation statistics
        r2 = r2_score(observed_clean, predicted_clean)
        rmse = np.sqrt(np.mean((observed_clean - predicted_clean) ** 2))
        mae = mean_absolute_error(observed_clean, predicted_clean)
        bias = np.mean(predicted_clean - observed_clean)
        
        print(f"  - Validation: R² = {r2:.3f}, RMSE = {rmse:.2f}m, MAE = {mae:.2f}m")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'n_samples': len(predicted_clean),
            'predicted': predicted_clean,
            'observed': observed_clean
        }
    
    def save_bathymetry(self, output_path: str, compress: bool = True):
        """
        Save bathymetry map as GeoTIFF.
        
        Parameters:
        -----------
        output_path : str
            Output file path
        compress : bool, default=True
            Apply compression to output file
        """
        if self.bathymetry is None:
            raise ValueError("No bathymetry map to save")
        
        print(f"Saving bathymetry to: {output_path}")
        
        # Set up compression
        compress_options = {'compress': 'lzw'} if compress else {}
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=self.bathymetry.shape[0],
            width=self.bathymetry.shape[1],
            count=1,
            dtype=self.bathymetry.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan,
            **compress_options
        ) as dst:
            dst.write(self.bathymetry, 1)
    
    def create_visualization(self, output_dir: str = 'output', 
                           depth_range: Tuple[float, float] = (0, 20)):
        """
        Create visualization of bathymetry results.
        
        Parameters:
        -----------
        output_dir : str, default='output'
            Output directory for plots
        depth_range : tuple, default=(0, 20)
            Depth range for visualization (min, max)
        """
        if self.bathymetry is None:
            raise ValueError("No bathymetry map to visualize")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bathymetry map
        im1 = axes[0, 0].imshow(self.bathymetry, cmap='viridis_r', 
                               vmin=depth_range[0], vmax=depth_range[1])
        axes[0, 0].set_title('Satellite-Derived Bathymetry')
        plt.colorbar(im1, ax=axes[0, 0], label='Depth (m)')
        
        # Water mask
        axes[0, 1].imshow(self.water_mask, cmap='Blues')
        axes[0, 1].set_title('Water Mask')
        
        # RGB composite (if available)
        if all(band in self.bands for band in ['B2', 'B3', 'B4']):
            rgb = self._create_rgb_composite()
            axes[1, 0].imshow(rgb)
            axes[1, 0].set_title('RGB Composite')
        
        # Depth histogram
        valid_depths = self.bathymetry[~np.isnan(self.bathymetry)]
        if len(valid_depths) > 0:
            axes[1, 1].hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Depth (m)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Depth Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'bathymetry_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path / 'bathymetry_results.png'}")
    
    def _create_rgb_composite(self) -> np.ndarray:
        """Create RGB composite for visualization."""
        red = self.bands['B4']
        green = self.bands['B3'] 
        blue = self.bands['B2']
        
        # Stack and normalize
        rgb = np.stack([red, green, blue], axis=-1)
        
        # Percentile stretch
        for i in range(3):
            band = rgb[:, :, i]
            valid_data = band[np.isfinite(band)]
            if len(valid_data) > 0:
                p2, p98 = np.percentile(valid_data, [2, 98])
                rgb[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        
        return rgb


def load_calibration_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load calibration data from CSV file.
    
    Expected format: columns for 'row', 'col', 'depth'
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with calibration data
        
    Returns:
    --------
    tuple
        (coordinates, depths) arrays
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    coordinates = df[['row', 'col']].values
    depths = df['depth'].values
    
    return coordinates, depths


def main_example():
    """
    Example usage of Sentinel-2 bathymetry extraction.
    """
    print("Sentinel-2 Bathymetry Extraction Example")
    print("=" * 40)
    
    # Initialize processor
    processor = Sentinel2Bathymetry()
    
    print("This is a template for Sentinel-2 bathymetry extraction.")
    print("To run with real data, provide:")
    print("1. Sentinel-2 image path")
    print("2. Calibration data (CSV with row, col, depth columns)")
    print("3. Optional validation data")
    
    # Template usage:
    """
    # Load Sentinel-2 image
    bands = processor.load_sentinel2_image('path/to/sentinel2_image.tif')
    
    # Preprocess
    processed_bands = processor.preprocess_sentinel2()
    
    # Load calibration data
    cal_coords, cal_depths = load_calibration_data('calibration_data.csv')
    
    # Extract bathymetry
    results = processor.extract_bathymetry_stumpf(cal_depths, cal_coords)
    
    # Optional: validate with independent data
    val_coords, val_depths = load_calibration_data('validation_data.csv')
    validation = processor.validate_bathymetry(val_depths, val_coords)
    
    # Save results
    processor.save_bathymetry('bathymetry_map.tif')
    processor.create_visualization()
    
    print("\\nResults:")
    print(f"Calibration R² = {results['r2']:.3f}")
    print(f"Calibration RMSE = {results['rmse']:.2f}m")
    if 'validation' in locals():
        print(f"Validation R² = {validation['r2']:.3f}")
        print(f"Validation RMSE = {validation['rmse']:.2f}m")
    """


if __name__ == "__main__":
    main_example()

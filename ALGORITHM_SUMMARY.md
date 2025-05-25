# Comprehensive Satellite-Derived Bathymetry Algorithm Implementation

This repository provides a complete implementation of major satellite-derived bathymetry (SDB) algorithms, based on decades of research and recent comparative studies. The implementation covers empirical, semi-analytical, and machine learning approaches.

## Algorithm Categories

### 1. Empirical Methods

These algorithms establish empirical relationships between spectral reflectance and water depth using calibration data.

#### A. Polcyn et al. (1970) - Simple Band Ratio
- **Formula**: `Depth ∝ ln(B_blue) / ln(B_green)`
- **Description**: The foundational SDB algorithm using simple logarithmic band ratios
- **Advantages**: Simple, fast, requires minimal calibration
- **Limitations**: Linear relationship assumption, limited accuracy in complex waters
- **Best for**: Clear waters, initial depth estimates

#### B. Jupp (1988) - Linear Transform
- **Formula**: `Depth = a + b × ln(B_blue) + c × ln(B_green)`
- **Description**: Extension of simple ratio to include both bands explicitly
- **Advantages**: More flexible than simple ratios, better handling of band interactions
- **Limitations**: Still assumes linear relationships
- **Best for**: Moderate complexity waters, improved accuracy over simple ratios

#### C. Stumpf et al. (2003) - Band Ratio Method
- **Formula**: `pSDB = ln(n × B_blue) / ln(n × B_green)`, then `Depth = m1 × pSDB - m0`
- **Description**: Refined ratio method with scaling constant and linear/polynomial calibration
- **Advantages**: Widely used, good performance, handles various water types
- **Limitations**: Requires good calibration data, sensitive to atmospheric effects
- **Best for**: General purpose SDB, operational applications

#### D. Lyzenga (1978, 1981) - Multiple Linear Regression
- **Formula**: `Depth = a₀ + a₁×ln(B₁) + a₂×ln(B₂) + ... + aₙ×ln(Bₙ)`
- **Description**: Multiple regression using logarithmically transformed bands
- **Advantages**: Can use multiple bands, accounts for complex spectral relationships
- **Limitations**: Requires more calibration data, potential overfitting
- **Best for**: Multi-spectral data, complex water conditions

### 2. Semi-Analytical/Physics-Based Methods

These algorithms use radiative transfer theory to model the underwater light field.

#### A. Lee et al. (1999) - Radiative Transfer Model
- **Description**: Models water-leaving radiance using inherent optical properties
- **Parameters**: Phytoplankton absorption, CDOM, particle backscattering, depth, bottom type
- **Advantages**: Physics-based, accounts for water column properties, no calibration needed
- **Limitations**: Computationally intensive, requires accurate atmospheric correction
- **Best for**: Research applications, areas without calibration data

#### B. ALUT (Hedley et al., 2009) - Adaptive Look-Up Table
- **Description**: Uses Lee model with pre-computed look-up tables for optimization
- **Parameters**: 6 parameters (P, G, X, H, E, M) estimated through spectral matching
- **Advantages**: Faster than full radiative transfer, physics-based
- **Limitations**: Still computationally demanding, parameter space limitations
- **Best for**: Operational physics-based applications, coral reef environments

### 3. Machine Learning Methods

These algorithms use statistical learning to establish complex non-linear relationships.

#### A. Random Forest (RF)
- **Description**: Ensemble method using multiple decision trees
- **Features**: Multiple spectral indices, band ratios, transformed bands
- **Advantages**: Handles non-linear relationships, robust to outliers, feature importance
- **Limitations**: Requires training data, black box approach
- **Best for**: Complex water conditions, multiple data sources

#### B. Support Vector Regression (SVR)
- **Description**: Non-linear regression using kernel functions
- **Advantages**: Effective for high-dimensional data, good generalization
- **Limitations**: Parameter tuning required, computationally intensive
- **Best for**: Non-linear depth relationships, limited training data

#### C. Artificial Neural Networks (ANN)
- **Description**: Deep learning approaches for pattern recognition
- **Advantages**: Can model very complex relationships, multiple inputs
- **Limitations**: Requires large training datasets, prone to overfitting
- **Best for**: Large datasets, complex multi-sensor applications

## Implementation Features

### Preprocessing Modules

1. **Sun Glint Correction**
   - Sentinel-2 specific coefficients
   - NIR-based correction methods
   - Validation tools

2. **Atmospheric Correction**
   - Dark Object Subtraction (DOS)
   - Histogram-based methods
   - Adaptive approaches for deep water

3. **Water Masking**
   - NDWI and MNDWI indices
   - Multi-criteria water detection
   - Quality assessment

### Validation and Assessment

1. **Accuracy Metrics**
   - R², RMSE, MAE, bias
   - Cross-validation support
   - Uncertainty quantification

2. **Comparison Framework**
   - Standardized evaluation
   - Performance visualization
   - Algorithm recommendations

## Usage Guidelines

### Algorithm Selection

**For Operational Applications:**
- **Clear waters**: Stumpf algorithm (fast, reliable)
- **Moderate complexity**: Lyzenga RGB or Random Forest
- **Complex waters**: Machine Learning methods

**For Research Applications:**
- **Physics understanding**: Lee radiative transfer model
- **Method development**: ALUT or custom ML approaches
- **Comparative studies**: Use comparison framework

**For Different Sensors:**
- **Sentinel-2**: All algorithms supported, Stumpf widely used
- **Landsat**: Empirical methods preferred
- **WorldView**: High-resolution empirical methods
- **Hyperspectral**: Lee model or advanced ML methods

### Data Requirements

**Minimum Requirements:**
- Blue and green bands (all empirical methods)
- Water mask
- Calibration depth samples (empirical methods)

**Recommended:**
- Red band (improved accuracy)
- NIR band (sun glint correction, water masking)
- SWIR band (atmospheric correction, water masking)
- Validation depth samples

**Optimal:**
- Coastal/violet band (maximum penetration)
- Multiple calibration datasets
- Concurrent field measurements
- Atmospheric correction products

## Performance Expectations

### Typical Accuracy (in optimal conditions):
- **Empirical methods**: R² = 0.7-0.9, RMSE = 0.5-2.0m
- **Semi-analytical**: R² = 0.6-0.8, RMSE = 1.0-3.0m
- **Machine Learning**: R² = 0.8-0.95, RMSE = 0.3-1.5m

### Depth Limitations:
- **Clear waters**: 0-25m (Sentinel-2), 0-30m (high-resolution)
- **Moderate turbidity**: 0-15m
- **Turbid waters**: 0-5m

## References

1. Polcyn, F.C., et al. (1970). The measurement of water depth by remote sensing techniques.
2. Lyzenga, D.R. (1978). Passive remote sensing techniques for mapping water depth and bottom features.
3. Lyzenga, D.R. (1981). Remote sensing of bottom reflectance and water attenuation parameters.
4. Jupp, D.L.B. (1988). Background and extensions to depth of penetration mapping.
5. Lee, Z., et al. (1999). Hyperspectral remote sensing for shallow waters.
6. Stumpf, R.P., et al. (2003). Determination of water depth with high‐resolution satellite imagery.
7. Hedley, J.D., et al. (2009). Coral reef applications of Sentinel-2.

## Future Developments

- Deep learning architectures
- Multi-sensor fusion approaches
- Real-time processing capabilities
- Uncertainty quantification improvements
- Integration with ICESat-2 lidar data

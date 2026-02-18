# Spectra_convolution_plot
This repository contains Python tools for generating, convoluting, and plotting absorption spectra from theoretical (computational) and experimental data.   It automatically detects data units (eV or nm), applies broadening functions (Gaussian/Lorentzian), and generates comparative plots between Vibronic, Computed (Stick), and Experimental spectra.
## Features

- **Automatic Unit Detection:** Detects if input data is in eV or nm and standardizes to eV for calculation.
- **Broadening:** Convolutes "stick" spectra using Gaussian or Lorentzian functions.
- **Layering:** Plots experimental data (dashed lines) over filled computed areas for clear comparison.
- **Plot Sets:** Generates up to 4 sets of comparisons (Individual/Combined vs. Raw/Vibronic).
- **Output:** Saves high-resolution `.png` files for both Energy (eV) and Wavelength (nm).

## Scripts

### 1. `espectro_generator_automatic.py` (Recommended)
This script **automatically aligns** calculated spectra to experimental data.
- It finds the highest intensity peak in the experimental data.
- It calculates the shift required for the computed spectra to match that peak.
- It produces "Total" summed plots automatically.

### 2. `espectro_convolucionado_manual.py`
This script allows for **manual control**.
- You define the energy shift (eV) manually via CLI arguments or interactive prompts.
- You can manually scale the intensity of individual files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pedro-lara-chem/Spectra_convolution_plot.git
   cd Spectra-convolution_plot
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
## Usage 
Important: Run these scripts inside the directory containing your data files (.txt, .dat, etc.).
## File Naming convention
The scripts categorize files based on their filenames:

* *exp*: Treated as Experimental data (plotted as dashed lines).

* *vib*: Treated as Vibronic data (plotted as is).

*  Other: Treated as Raw Computed (Stick) spectra (Energy vs Osc. Strength). These will be convoluted.
## Running the automatic aligner
```bash
python espectro_generator_automatic.py --func gaussian --fwhm 0.2
```
## Running the manual plotter
```bash
# Example: Gaussian broadening with 0.3 eV FWHM
python espectro_convolucionado_manual.py -f gaussian -w 0.3

# Example: Pre-define shifts (0.5 eV for computed, -0.1 for vibronic)
python espectro_convolucionado_manual.py -sc 0.5 -sv -0.1
```
## Command Line Arguments

Both scripts can be run from the terminal with specific arguments to control the broadening function, width, and (in the manual version) energy shifts.

### 1. Automatic Script (`espectro_generator_automatic.py`)
This script automatically aligns computed spectra to the experimental reference.

| Argument | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--func` | `-f` | `str` | `'gaussian'` | Broadening function type. Options: `'gaussian'` or `'lorentzian'`. |
| `--fwhm` | `-w` | `float` | `0.2` | Full-Width at Half-Maximum (FWHM) for the broadening function in eV. |

**Example:**
```bash
python espectro_generator_automatic.py -f lorentzian -w 0.3
```
### Manual Script Arguments (`espectro_convolucionado_manual.py`)

This script allows for manual control over energy shifting and scaling via command-line arguments or interactive prompts.

| Argument | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--func` | `-f` | `str` | `'gaussian'` | Broadening function type. Options: `'gaussian'` or `'lorentzian'`. |
| `--fwhm` | `-w` | `float` | `0.2` | Full-Width at Half-Maximum (FWHM) for the broadening function in eV. |
| `--shift_computed` | `-sc` | `float` | `None` | Manual energy shift (eV) for **Computed ('raw')** spectra. If omitted, the script will ask interactively. |
| `--shift_vibronic` | `-sv` | `float` | `None` | Manual energy shift (eV) for **Vibronic ('vib')** spectra. If omitted, the script will ask interactively. |

**Example Usage:**
```bash
# Run with Lorentzian broadening (0.3 eV) and explicit shifts
python espectro_convolucionado_manual.py -f lorentzian -w 0.3 -sc 0.5 -sv -0.1
```
## Acknowledgments
This software was developed by **Pedro Lara Salcedo at Excited States Reactivity Group** at **Universidad Autónoma de Madrid**.  This work was supported by **Diseño y caracterización de nuevos materiales moleculares y optimización de fármacos: Sinergia experimento y teoría$$** under grant number **$$PGC2018-094644-B-C21$$**.

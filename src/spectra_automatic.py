"""
This script generates and plots absorption spectra from theoretical and
experimental data files located in its execution directory.

This version *AUTOMATICALLY* aligns spectra.
If one or more 'exp' files are provided, the script finds the
highest intensity peak among all 'exp' files and uses its energy
as the 'reference peak'.
It then finds the peak for each 'raw' (convoluted) and 'vib'
spectrum and calculates a shift (E_ref - E_peak) to align
them with the experimental data.
If no 'exp' file is found, all shifts are set to 0.0.

It scans the directory for files and categorizes them:
- Files with 'vib' in the name are plotted as is (assumed convoluted).
- Files with 'exp' in the name are plotted as is (experimental data).
- All other compatible files are treated as raw 'stick' spectra
  (Energy vs. Oscillator Strength) and are convoluted using either a
  Gaussian or Lorentzian function.

The script auto-detects if the energy column is in eV or nm and
converts to eV internally for all processing.

It generates *up to* four sets of plots:
1.  Set 1: Individual Vibronic vs. Individual Raw
2.  Set 2: Combined Vibronic vs. Individual Raw
3.  Set 3: Individual Vibronic vs. Combined Raw
4.  Set 4: Combined Vibronic vs. Combined Raw

*** NEW: Experimental Plot in ALL Figures ***
The experimental spectrum is now plotted on *every* figure,
including the individual deconvolution plots for each 'raw' file.

*** NEW: Styling and Layering ***
- Experimental ('exp') files are always plotted as a dashed black line
  with the highest zorder (plot-layer priority) for maximum clarity.
- Computed ('raw') and 'vib' files are now plotted with a filled area
  (same color as the line, 30% opacity) at a lower zorder, so they
  appear "behind" the experimental line.

*** NEW: Automatic 'Total' Plots & Manual Scaling ***
- If more than one 'raw' or 'vib' file is found, a "Total"
  (summed) spectrum is *automatically* generated (the script
  no longer asks).
- The manual scaling prompt now *also* asks for scaling factors
  for these "Total Computed" and "Total Vibronic" plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse  # Import argparse

H_C_eV_nm = 1239.8 # E(eV) = 1239.8 / lambda(nm)

# --- Helper for subscript labels ---
def format_subscript(n):
    """Converts an integer into a string 'Sₙ' with subscript digits."""
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return f"S{str(n).translate(subscript_digits)}"

def get_cli_args():
    """
    Parses command-line arguments for broadening function type and FWHM.
    """
    parser = argparse.ArgumentParser(description="Generate and plot absorption spectra with auto-alignment.")
    
    parser.add_argument(
        '-f', '--func',
        type=str,
        default='gaussian',
        choices=['gaussian', 'lorentzian'],
        help="Broadening function type: 'gaussian' or 'lorentzian'."
    )
    
    parser.add_argument(
        '-w', '--fwhm',
        type=float,
        default=0.2,
        help="Full-Width at Half-Maximum (FWHM) for broadening."
    )
    
    args = parser.parse_args()
    
    if args.fwhm <= 0:
        print("Error: FWHM must be a positive number.")
        exit()
        
    return args.func, args.fwhm

# --- MODIFIED: Function to ask for manual scaling (now includes 'Total' plots) ---
def get_manual_scaling(raw_files, vib_files, generate_total_computed, generate_total_vibronic):
    """
    Asks the user if they want to manually scale raw, vibronic, and/or total
    spectra and, if so, prompts for a scaling factor for each.
    Returns: A dictionary mapping filename (or 'Total Computed') to scaling factor.
    """
    scaling_factors = {}
    
    # Set default scale to 1.0 for all files and totals
    all_files = raw_files + vib_files
    for filepath in all_files:
        filename = os.path.basename(filepath)
        scaling_factors[filename] = 1.0
        
    scaling_factors['Total Computed'] = 1.0
    scaling_factors['Total Vibronic'] = 1.0
        
    # --- Ask for RAW files ---
    if raw_files:
        try:
            choice_raw = input("\nDo you want to manually scale individual computed ('raw') spectra? [y/N]: ").strip().lower()
        except EOFError: # Handle non-interactive environments
            choice_raw = 'n'
            
        if choice_raw == 'y':
            print("  > Enter scaling factor (e.g., 0.8, 1.2). Press Enter for default (1.0).")
            for filepath in raw_files:
                filename = os.path.basename(filepath)
                try:
                    scale_input = input(f"    - Scale for '{filename}': ").strip()
                    if scale_input: # Not empty
                        scale_factor = float(scale_input)
                        scaling_factors[filename] = scale_factor
                    # If empty, it keeps the default 1.0
                except ValueError:
                    print(f"    > Invalid input. Using default 1.0 for {filename}.")
                    # Keeps the default 1.0
                except EOFError:
                    break # Stop asking
    
    # --- Ask for VIB files ---
    if vib_files:
        try:
            choice_vib = input("\nDo you want to manually scale individual vibronic ('vib') spectra? [y/N]: ").strip().lower()
        except EOFError: # Handle non-interactive environments
            choice_vib = 'n'
            
        if choice_vib == 'y':
            print("  > Enter scaling factor (e.g., 0.8, 1.2). Press Enter for default (1.0).")
            for filepath in vib_files:
                filename = os.path.basename(filepath)
                try:
                    scale_input = input(f"    - Scale for '{filename}': ").strip()
                    if scale_input: # Not empty
                        scale_factor = float(scale_input)
                        scaling_factors[filename] = scale_factor
                except ValueError:
                    print(f"    > Invalid input. Using default 1.0 for {filename}.")
                except EOFError:
                    break
    
    # --- NEW: Ask for 'Total Computed' scaling ---
    if generate_total_computed:
        try:
            choice_total_raw = input("\nDo you want to manually scale the 'Total Computed' spectrum? [y/N]: ").strip().lower()
        except EOFError:
            choice_total_raw = 'n'
        if choice_total_raw == 'y':
            try:
                scale_input = input(f"    - Scale for 'Total Computed': ").strip()
                if scale_input:
                    scaling_factors['Total Computed'] = float(scale_input)
            except (ValueError, EOFError):
                print("    > Invalid input. Using default 1.0.")

    # --- NEW: Ask for 'Total Vibronic' scaling ---
    if generate_total_vibronic:
        try:
            choice_total_vib = input("\nDo you want to manually scale the 'Total Vibronic' spectrum? [y/N]: ").strip().lower()
        except EOFError:
            choice_total_vib = 'n'
        if choice_total_vib == 'y':
            try:
                scale_input = input(f"    - Scale for 'Total Vibronic': ").strip()
                if scale_input:
                    scaling_factors['Total Vibronic'] = float(scale_input)
            except (ValueError, EOFError):
                print("    > Invalid input. Using default 1.0.")
    
    return scaling_factors

def load_and_convert_to_eV(filepath):
    """
    Loads data from a file.
    Detects if the first column is eV or nm based on median value.
    Converts to eV if necessary.
    Returns: (energies_eV, intensities, original_indices)
    """
    try:
        data = np.loadtxt(filepath, comments=['#', '@'])
    except Exception as e:
        print(f"Error loading {os.path.basename(filepath)}: {e}. Skipping.")
        return None, None, None

    if data.size == 0:
        print(f"Warning: Skipping empty file: {os.path.basename(filepath)}")
        return None, None, None
    
    if data.ndim == 1:
        data = np.atleast_2d(data)
        
    if data.ndim == 1 or data.shape[1] < 2:
        print(f"Warning: Skipping file with incorrect data shape: {os.path.basename(filepath)}")
        return None, None, None
    
    data[data[:, 0] == 0, 0] = 1e-9 
    
    energies = data[:, 0]
    intensities = data[:, 1]
    original_indices = np.arange(1, len(energies) + 1)
    
    if np.median(energies) > 50.0:
        print(f"  > Detected 'nm' in {os.path.basename(filepath)}. Converting to eV.")
        energies_eV = H_C_eV_nm / energies
        sort_indices = np.argsort(energies_eV)
        energies_eV = energies_eV[sort_indices]
        intensities = intensities[sort_indices]
        original_indices = original_indices[sort_indices]
    else:
        print(f"  > Detected 'eV' in {os.path.basename(filepath)}. Converting to nm.")
        energies_eV = energies
        
    return energies_eV, intensities, original_indices

# --- NEW: Function to find the reference peak from all 'exp' files ---
def get_reference_peak_eV(exp_files):
    """
    Loads all exp files, finds the highest intensity peak among all of
    them, and returns the energy (eV) of that peak.
    """
    global_peak_E = None
    global_max_I = -np.inf
    found_exp_data = False

    print("\nFinding experimental reference peak...")
    for filepath in exp_files:
        try:
            energies, intensities, _ = load_and_convert_to_eV(filepath) 
            if energies is None: continue
            
            local_max_I = np.max(intensities)
            if local_max_I > global_max_I:
                global_max_I = local_max_I
                global_peak_E = energies[np.argmax(intensities)]
                found_exp_data = True
        except Exception as e:
            print(f"Error reading {os.path.basename(filepath)} for reference peak: {e}.")

    if found_exp_data:
        print(f"  > Experimental reference peak found at: {global_peak_E:.3f} eV")
        return global_peak_E
    else:
        print("  > No valid 'exp' data found. Auto-shifting disabled.")
        return None

# --- NEW: Function to find the peak of a single 'raw' or 'vib' file ---
def get_file_peak_eV(filepath, x_axis_calc_grid, is_raw, func_type, fwhm):
    """
    Finds the peak energy (eV) for a single file.
    If 'is_raw', it convolutes the spectrum first.
    If not 'is_raw' (i.e., 'vib'), it finds the max of the loaded data.
    """
    energies, intensities, indices = load_and_convert_to_eV(filepath)
    if energies is None:
        return None

    if is_raw:
        # --- Perform convolution to find the peak ---
        total_spectrum = np.zeros_like(x_axis_calc_grid)
        if func_type == 'gaussian':
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            for E, f, idx in zip(energies, intensities, indices):
                peak = f * np.exp(-((x_axis_calc_grid - E)**2) / (2.0 * sigma**2))
                total_spectrum += peak
                
        elif func_type == 'lorentzian':
            gamma = fwhm / 2.0
            gamma_sq = gamma**2
            for E, f, idx in zip(energies, intensities, indices):
                peak = f * gamma_sq / ((x_axis_calc_grid - E)**2 + gamma_sq)
                total_spectrum += peak
        
        if np.max(total_spectrum) > 0:
            peak_E = x_axis_calc_grid[np.argmax(total_spectrum)]
            return peak_E
        else:
            return None # Failed convolution
    
    else:
        # --- For 'vib' files, just find the max of the raw data ---
        if np.max(intensities) > 0:
            peak_E = energies[np.argmax(intensities)]
            return peak_E
        else:
            return None # Empty file

def categorize_files(script_dir):
    """
    Scans the script's directory for files and categorizes them.
    Returns: (raw_files, vib_files, exp_files)
    """
    raw_files = []
    vib_files = []
    exp_files = []
    ignore_extensions = ['.py', '.png', '.jpg', '.jpeg', '.pdf', '.md', '.pyc']
    
    print("\nScanning for data files...")
    for filename in os.listdir(script_dir):
        filepath = os.path.join(script_dir, filename)
        if os.path.isfile(filepath):
            name_no_ext, ext = os.path.splitext(filename)
            if ext.lower() in ignore_extensions:
                continue 
            
            name_lower = os.path.splitext(filename)[0].lower()
            
            if 'total_computed' in name_lower or 'total_vibronic' in name_lower:
                print(f"  Skipping potential output file: {filename}")
                continue 

            if 'vib' in name_lower:
                vib_files.append(filepath)
                print(f"  Found 'vib' file: {filename}")

            elif 'exp' in name_lower:
                exp_files.append(filepath)
                print(f"  Found 'exp' file: {filename}")
            else:
                try:
                    test_data = np.loadtxt(filepath, comments=['#', '@'])
                    is_valid_raw = False
                    if test_data.size > 0:
                        if test_data.ndim == 1 and test_data.shape == (2,):
                            is_valid_raw = True
                        elif test_data.ndim == 2 and test_data.shape[1] >= 2:
                            is_valid_raw = True

                    if is_valid_raw:
                        raw_files.append(filepath)
                        print(f"  Found 'raw' file: {filename}")
                    else:
                        if not any(ext in filename for ext in ['.py', '.png', '.txt']):
                            print(f"  Skipping non-data file: {filename}")
                except Exception:
                    if not any(ext in filename for ext in ['.py', '.png', '.txt']):
                         print(f"  Skipping non-data file: {filename}")
                    
    return raw_files, vib_files, exp_files

def create_calculation_grid(all_files_lists, default_fwhm):
    """
    Finds the global min/max energy (in eV) across ALL files.
    This creates a wide 'x_axis' to be used for all convolution
    calculations, ensuring no data is cut off.
    """
    global_min_E = np.inf
    global_max_E = -np.inf
    found_data = False

    print("\n  > Creating wide calculation grid by scanning ALL files...")
    for file_list in all_files_lists:
        for filepath in file_list:
            try:
                energies, _, _ = load_and_convert_to_eV(filepath) 
                if energies is None: continue
                
                if energies.min() < global_min_E: global_min_E = energies.min()
                if energies.max() > global_max_E: global_max_E = energies.max()
                found_data = True
            except Exception as e:
                pass 

    if not found_data:
        print("\nError: No valid data files found. Exiting.")
        return None

    padding_eV = default_fwhm * 10
    
    final_min_E = global_min_E - padding_eV
    if final_min_E <= 0.01:
        final_min_E = 0.01
        
    final_max_E = global_max_E + padding_eV
    
    print(f"  > Calculation grid set: {final_min_E:.2f} eV to {final_max_E:.2f} eV")
    x_axis = np.linspace(final_min_E, final_max_E, 2000)
    return x_axis

def get_plot_limits(exp_files):
    """
    Finds the min/max energy (in eV) across files to set
    the final plot limits (the 'zoom').
    Returns: (E_min, E_max)
    """

    local_min_E = np.inf
    local_max_E = -np.inf

    print("  > Setting final plot limits.")
    for filepath in exp_files:
        try:
            energies, _, _ = load_and_convert_to_eV(filepath) 
            if energies is None: continue
            
            if energies.min() < local_min_E: local_min_E = energies.min()
            if energies.max() > local_max_E: local_max_E = energies.max()
        except Exception as e:
            print(f"Error reading {os.path.basename(filepath)} for limits: {e}.")

    print(local_max_E,local_min_E)
    range_eV = local_max_E - local_min_E
    if range_eV == 0:
        padding_eV = 0.1 # default padding for single point
    else:
        padding_eV = range_eV * 0.05
    
    E_min_plot = local_min_E - padding_eV
    E_max_plot = local_max_E + padding_eV
    print(E_max_plot,E_min_plot)
    print(f"  > Final plot 'zoom' set: {E_min_plot:.2f} eV to {E_max_plot:.2f} eV")
    return (E_min_plot, E_max_plot)

def finalize_and_save_plot(fig_tuple, ax_tuple, base_filename, title_suffix, plot_eV_limits, script_dir):
    """Finalizes and saves a pair of eV and nm plots."""
    
    fig_eV, fig_nm = fig_tuple
    ax_eV, ax_nm = ax_tuple

    if fig_eV is None or ax_eV is None:
        print(f"\nSkipping '{base_filename}' plot generation (missing data).")
        return # Plot was not generated (e.g., missing combined data)

    # --- Finalize eV Plot ---
    ax_eV.set_xlabel("Energy (eV)", fontsize=14)
    ax_eV.set_ylabel("Normalized Intensity", fontsize=14)
    ax_eV.set_title(f"Experimental vs. {title_suffix} (Energy)", fontsize=16)
    ax_eV.legend(loc='best', fontsize=10)
    ax_eV.grid(True, linestyle=':', alpha=0.6)
    ax_eV.set_ylim(bottom=0)
    ax_eV.set_xlim(plot_eV_limits[0], plot_eV_limits[1])
    fig_eV.tight_layout()
    
    # --- Finalize nm Plot ---
    ax_nm.set_xlabel("Wavelength (nm)", fontsize=14)
    ax_nm.set_ylabel("Normalized Intensity", fontsize=14)
    ax_nm.set_title(f"Experimental vs. {title_suffix} (Wavelength)", fontsize=16)
    ax_nm.legend(loc='best', fontsize=10)
    ax_nm.grid(True, linestyle=':', alpha=0.6)
    ax_nm.set_ylim(bottom=0)
    
    L_max = H_C_eV_nm / plot_eV_limits[0]
    L_min = H_C_eV_nm / plot_eV_limits[1]
    ax_nm.set_xlim(L_min, L_max)
    fig_nm.tight_layout()

    # --- Save Figures ---
    fig_ev_path = os.path.join(script_dir, f"spectrum_{base_filename}_eV.png")
    fig_nm_path = os.path.join(script_dir, f"spectrum_{base_filename}_nm.png")
    
    try:
        fig_eV.savefig(fig_ev_path, dpi=300, bbox_inches='tight')
        fig_nm.savefig(fig_nm_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plots to:\n  {os.path.basename(fig_ev_path)}\n  {os.path.basename(fig_nm_path)}")
    except Exception as e:
        print(f"\nError saving plots '{base_filename}': {e}")
    finally:
        plt.close(fig_eV)
        plt.close(fig_nm)


def main():      
    script_dir = os.getcwd()

    # 1. Get user preferences from command line
    func_type, fwhm = get_cli_args()
    
    # --- REMOVED: All manual shift input logic ---
    
    print("\nScript settings:")
    print(f"  Broadening function: {func_type}")
    print(f"  FWHM: {fwhm}")
    print(f"  Shifting: AUTOMATIC (based on 'exp' file peak)")
    print(f"  Deconvolution threshold: Oscillator Strength > 0.05")
    print(f"  Output: Plots will be saved automatically.")

    # 2. Find and categorize all files
    raw_files, vib_files, exp_files = categorize_files(script_dir)
    all_files_lists = [raw_files, vib_files, exp_files]

    generate_total_computed = len(raw_files) > 1
    if generate_total_computed:
        print(f"  > Found {len(raw_files)} 'raw' files. Will generate 'Total Computed' sum.")

    generate_total_vibronic = len(vib_files) > 1
    if generate_total_vibronic:
        print(f"  > Found {len(vib_files)} 'vib' files. Will generate 'Total Vibronic' sum.")

    # 3. Determine calculation grid and plot limits
    x_axis = create_calculation_grid(all_files_lists, fwhm)
    if x_axis is None:
        return
        
    if not exp_files:
            plot_eV_limits = get_plot_limits(raw_files)
    else:
            plot_eV_limits = get_plot_limits(exp_files) 

    # 4. --- NEW: Automatic Shift Calculation ---
    E_peak_exp = get_reference_peak_eV(exp_files)
    
    computed_shifts = {}
    vibronic_shifts = {}
    
    if E_peak_exp is not None:
        print("\nCalculating automatic shifts...")
        for filepath in raw_files:
            E_peak_raw = get_file_peak_eV(filepath, x_axis, True, func_type, fwhm)
            if E_peak_raw is not None:
                shift = E_peak_exp - E_peak_raw
                computed_shifts[filepath] = shift
                print(f"  > Shift for '{os.path.basename(filepath)}' (raw): {shift:+.3f} eV")
            else:
                computed_shifts[filepath] = 0.0
        
        for filepath in vib_files:
            E_peak_vib = get_file_peak_eV(filepath, x_axis, False, None, None)
            if E_peak_vib is not None:
                shift = E_peak_exp - E_peak_vib
                vibronic_shifts[filepath] = shift
                print(f"  > Shift for '{os.path.basename(filepath)}' (vib): {shift:+.3f} eV")
            else:
                vibronic_shifts[filepath] = 0.0
    else:
        print("\nDisabling auto-shift: No experimental reference peak found. All shifts set to 0.0 eV.")
        for filepath in raw_files:
            computed_shifts[filepath] = 0.0
        for filepath in vib_files:
            vibronic_shifts[filepath] = 0.0
    
    # Shifts for 'Total' plots will be calculated later, after summing
    total_computed_shift = 0.0
    total_vibronic_shift = 0.0

    # 5. Get manual scaling factors
    scaling_factors = get_manual_scaling(raw_files, vib_files, generate_total_computed, generate_total_vibronic)
    
    print("\nGenerating plots...")
    
    # 7. Setup the plot sets
    print("\nSetting up plot figures...")
    
    fig_P1_eV, ax_P1_eV = plt.subplots(figsize=(14, 8))
    fig_P1_nm, ax_P1_nm = plt.subplots(figsize=(14, 8))

    fig_P2_eV, ax_P2_eV = (None, None)
    fig_P2_nm, ax_P2_nm = (None, None)
    if generate_total_vibronic:
        fig_P2_eV, ax_P2_eV = plt.subplots(figsize=(14, 8))
        fig_P2_nm, ax_P2_nm = plt.subplots(figsize=(14, 8))

    fig_P3_eV, ax_P3_eV = (None, None)
    fig_P3_nm, ax_P3_nm = (None, None)
    if generate_total_computed:
        fig_P3_eV, ax_P3_eV = plt.subplots(figsize=(14, 8))
        fig_P3_nm, ax_P3_nm = plt.subplots(figsize=(14, 8))

    fig_P4_eV, ax_P4_eV = (None, None)
    fig_P4_nm, ax_P4_nm = (None, None)
    if generate_total_computed and generate_total_vibronic:
        fig_P4_eV, ax_P4_eV = plt.subplots(figsize=(14, 8))
        fig_P4_nm, ax_P4_nm = plt.subplots(figsize=(14, 8))
    
    figs_P1 = (fig_P1_eV, fig_P1_nm); axes_P1 = (ax_P1_eV, ax_P1_nm)
    figs_P2 = (fig_P2_eV, fig_P2_nm); axes_P2 = (ax_P2_eV, ax_P2_nm)
    figs_P3 = (fig_P3_eV, fig_P3_nm); axes_P3 = (ax_P3_eV, ax_P3_nm)
    figs_P4 = (fig_P4_eV, fig_P4_nm); axes_P4 = (ax_P4_eV, ax_P4_nm)
    
    all_combined_axes_eV = [ax for ax in [ax_P1_eV, ax_P2_eV, ax_P3_eV, ax_P4_eV] if ax is not None]
    all_combined_axes_nm = [ax for ax in [ax_P1_nm, ax_P2_nm, ax_P3_nm, ax_P4_nm] if ax is not None]

    # 8. Initialize sum arrays
    total_computed_spectrum_sum = np.zeros_like(x_axis) if generate_total_computed else None
    total_vibronic_spectrum_sum = np.zeros_like(x_axis) if generate_total_vibronic else None

    # 9. Process and plot 'exp' files FIRST
    print("\nProcessing 'exp' files...")
    exp_plot_data = [] 
    for filepath in exp_files:
        filename = os.path.basename(filepath)
        label_name = os.path.splitext(filename)[0]
        
        try:
            energies, intensities, _ = load_and_convert_to_eV(filepath) 
            if energies is None: continue
                
            max_val = np.max(intensities)
            if max_val == 0:
                print(f"Warning: Data for {filename} is all zeros. Skipping.")
                continue
            
            normalized_intensities = intensities / max_val
            
            style = '--'
            width = 2.0
            color = 'black'
            zorder = 100 
            
            energies_nm = H_C_eV_nm / energies
            sort_idx_nm = np.argsort(energies_nm)
            x_plot_nm = energies_nm[sort_idx_nm]
            y_plot_nm = (normalized_intensities)[sort_idx_nm]
            
            exp_plot_data.append((energies, normalized_intensities, label_name, x_plot_nm, y_plot_nm))

            for ax in all_combined_axes_eV:
                ax.plot(energies, normalized_intensities, 
                        label=label_name, linestyle=style, linewidth=width, zorder=zorder, color=color)

            for ax in all_combined_axes_nm:
                ax.plot(x_plot_nm, y_plot_nm,
                         label=label_name, linestyle=style, linewidth=width, zorder=zorder, color=color)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")


    # 10. Process and plot RAW files (Convolution)
    print("\nProcessing 'raw' files...")
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        label_name = os.path.splitext(filename)[0]
        
        try:
            energies, osc_strengths, indices = load_and_convert_to_eV(filepath) 
            
            if energies is None: continue
                
            total_spectrum = np.zeros_like(x_axis)
            deconvoluted_peaks = [] 

            if func_type == 'gaussian':
                sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                for E, f, idx in zip(energies, osc_strengths, indices):
                    peak = f * np.exp(-((x_axis - E)**2) / (2.0 * sigma**2))
                    total_spectrum += peak
                    if f > 0.05: 
                        label = format_subscript(idx)
                        deconvoluted_peaks.append((peak, label, E, f))
                        
            elif func_type == 'lorentzian':
                gamma = fwhm / 2.0
                gamma_sq = gamma**2
                for E, f, idx in zip(energies, osc_strengths, indices):
                    peak = f * gamma_sq / ((x_axis - E)**2 + gamma_sq)
                    total_spectrum += peak
                    if f > 0.05: 
                        label = format_subscript(idx)
                        deconvoluted_peaks.append((peak, label, E, f))

            if generate_total_computed:
                total_computed_spectrum_sum += total_spectrum

            max_val = np.max(total_spectrum)
            if max_val == 0:
                print(f"Warning: Spectrum for {filename} is all zeros. Skipping.")
                continue
            
            normalized_total_spectrum = total_spectrum / max_val
            
            # --- MODIFIED: Get auto-shift ---
            shift_eV = computed_shifts.get(filepath, 0.0)
            label_eV = label_name
            label_nm = label_name
            shift_title_info = "" 

            if shift_eV != 0.0:
                label_eV = f"{label_name} (Shift: {shift_eV:+.2f} eV)"
                label_nm = f"{label_name} (Shift: {shift_eV:+.2f} eV)" 
                shift_title_info = f"(Shifted {shift_eV:+.2f} eV)"
                
            x_axis_shifted = x_axis + shift_eV
            
            scale_factor = scaling_factors.get(filename, 1.0)
            scaled_spectrum = normalized_total_spectrum * scale_factor
            
            if scale_factor != 1.0:
                label_eV_scaled = f"{label_eV} (x{scale_factor:.2f})"
                label_nm_scaled = f"{label_nm} (x{scale_factor:.2f})"
                print(f"  > Applying scale factor {scale_factor:.2f} to {filename} for combined plot.")
            else:
                label_eV_scaled = label_eV
                label_nm_scaled = label_nm
            
            x_axis_nm = H_C_eV_nm / x_axis_shifted
            sort_idx_nm = np.argsort(x_axis_nm)
            x_plot_nm = x_axis_nm[sort_idx_nm]
            y_plot_total_nm_scaled = (scaled_spectrum)[sort_idx_nm]
            
            line, = ax_P1_eV.plot(x_axis_shifted, scaled_spectrum, 
                                    label=label_eV_scaled, linewidth=2.5, zorder=10)
            ax_P1_eV.fill_between(x_axis_shifted, scaled_spectrum, 0,
                                    color=line.get_color(), alpha=0.3, zorder=9)
            
            if ax_P2_eV:
                ax_P2_eV.plot(x_axis_shifted, scaled_spectrum, 
                                label=label_eV_scaled, linewidth=2.5, zorder=10, 
                                color=line.get_color())
                ax_P2_eV.fill_between(x_axis_shifted, scaled_spectrum, 0,
                                    color=line.get_color(), alpha=0.3, zorder=9)

            ax_P1_nm.plot(x_plot_nm, y_plot_total_nm_scaled, 
                            label=label_nm_scaled, linewidth=2.5, zorder=10, 
                            color=line.get_color())
            ax_P1_nm.fill_between(x_plot_nm, y_plot_total_nm_scaled, 0,
                            color=line.get_color(), alpha=0.3, zorder=9)

            if ax_P2_nm:
                ax_P2_nm.plot(x_plot_nm, y_plot_total_nm_scaled, 
                                label=label_nm_scaled, linewidth=2.5, zorder=10, 
                                color=line.get_color())
                ax_P2_nm.fill_between(x_plot_nm, y_plot_total_nm_scaled, 0,
                                color=line.get_color(), alpha=0.3, zorder=9)


            y_plot_total_nm_unscaled = (normalized_total_spectrum)[sort_idx_nm]
            
            print(f"  > Generating individual deconvolution plots for {filename}...")
            fig_ind_eV, ax_ind_eV = plt.subplots(figsize=(14, 8))
            fig_ind_nm, ax_ind_nm = plt.subplots(figsize=(14, 8))
            
            for (energies_exp, norm_int_exp, label_exp, x_nm_exp, y_nm_exp) in exp_plot_data:
                ax_ind_eV.plot(energies_exp, norm_int_exp, label=label_exp, 
                               linestyle='--', linewidth=2.0, zorder=100, color='black')
                ax_ind_nm.plot(x_nm_exp, y_nm_exp, label=label_exp,
                               linestyle='--', linewidth=2.0, zorder=100, color='black')

            # --- MODIFIED: Plot SCALED spectrum ---
            ax_ind_eV.plot(x_axis_shifted, scaled_spectrum, 
                           label=label_eV_scaled, linewidth=2.5, zorder=10,
                           color=line.get_color())
            # --- MODIFIED: Fill SCALED area ---
            ax_ind_eV.fill_between(x_axis_shifted, scaled_spectrum, 0,
                           color=line.get_color(), alpha=0.3, zorder=9)
                           
            # --- MODIFIED: Plot SCALED spectrum (nm) ---
            ax_ind_nm.plot(x_plot_nm, y_plot_total_nm_scaled, 
                           label=label_nm_scaled, linewidth=2.5, zorder=10, 
                           color=line.get_color())
            # --- MODIFIED: Fill SCALED area (nm) ---
            ax_ind_nm.fill_between(x_plot_nm, y_plot_total_nm_scaled, 0,
                           color=line.get_color(), alpha=0.3, zorder=9)

            # --- MODIFIED: Plot SCALED deconvoluted peaks ---
            for i, (peak, label, E_max_eV, f_max) in enumerate(deconvoluted_peaks):
                deconv_label = None 
                peak_color = f'C{i+1}' # Get a new color (C1, C2, etc.)
                
                # --- NEW: Apply scaling to peak and max height ---
                # We scale the *normalized* peak for the deconvolution plot
                scaled_peak = (peak / max_val) * scale_factor
                scaled_f_max = (f_max / max_val) * scale_factor
                
                ax_ind_eV.plot(x_axis_shifted, scaled_peak,
                               linestyle='--', linewidth=1.0, 
                               label=deconv_label, zorder=5, color=peak_color)
                ax_ind_eV.fill_between(x_axis_shifted, scaled_peak, 0,
                                color=peak_color, alpha=0.3, zorder=4)
                
                E_max_shifted = E_max_eV + shift_eV
                ax_ind_eV.annotate(label,
                             xy=(E_max_shifted, scaled_f_max), 
                             xytext=(E_max_shifted, scaled_f_max * 1.05),
                             textcoords='data', color=peak_color,
                             fontsize=9, fontweight='bold',
                             ha='center', va='bottom', zorder=6)
                
                y_plot_peak_nm = (scaled_peak)[sort_idx_nm] 
                ax_ind_nm.plot(x_plot_nm, y_plot_peak_nm,
                         linestyle='--', linewidth=1.0, label=deconv_label,
                         zorder=5, color=peak_color)
                ax_ind_nm.fill_between(x_plot_nm, y_plot_peak_nm, 0,
                                 color=peak_color, alpha=0.3, zorder=4)

                E_max_nm_shifted = H_C_eV_nm / E_max_shifted
                ax_ind_nm.annotate(label,
                             xy=(E_max_nm_shifted, scaled_f_max),
                             xytext=(E_max_nm_shifted, scaled_f_max * 1.05),
                             textcoords='data', color=peak_color,
                             fontsize=9, fontweight='bold',
                             ha='center', va='bottom', zorder=6)
            
            title_eV = f"Deconvolution for {label_name} (Energy) {shift_title_info}"
            title_nm = f"Deconvolution for {label_name} (Wavelength) {shift_title_info}"

            ax_ind_eV.set_xlabel("Energy (eV)", fontsize=14)
            ax_ind_eV.set_ylabel("Normalized Intensity / Abs. Osc. Strength", fontsize=14) 
            ax_ind_eV.set_title(title_eV, fontsize=16)
            ax_ind_eV.grid(True, linestyle=':', alpha=0.6)
            ax_ind_eV.set_ylim(bottom=0)
            ax_ind_eV.legend(loc='best', fontsize=10)
            
            ax_ind_eV.set_xlim(plot_eV_limits[0], plot_eV_limits[1])
            fig_ind_eV.tight_layout()
            
            ax_ind_nm.set_xlabel("Wavelength (nm)", fontsize=14)
            ax_ind_nm.set_ylabel("Normalized Intensity / Abs. Osc. Strength", fontsize=14)
            ax_ind_nm.set_title(title_nm, fontsize=16)
            ax_ind_nm.grid(True, linestyle=':', alpha=0.6)
            ax_ind_nm.set_ylim(bottom=0)
            ax_ind_nm.legend(loc='best', fontsize=10)
            
            E_min_ind, E_max_ind = ax_ind_eV.get_xlim()
            if E_min_ind <= 0.01: E_min_ind = 0.01 
            L_max_ind = H_C_eV_nm / E_min_ind
            L_min_ind = H_C_eV_nm / E_max_ind
            ax_ind_nm.set_xlim(L_min_ind, L_max_ind)
            fig_ind_nm.tight_layout()

            fig1_ind_path = os.path.join(script_dir, f"spectrum_plot_ind_{label_name}_eV.png")
            fig2_ind_path = os.path.join(script_dir, f"spectrum_plot_ind_{label_name}_nm.png")
            try:
                fig_ind_eV.savefig(fig1_ind_path, dpi=300, bbox_inches='tight')
                fig_ind_nm.savefig(fig2_ind_path, dpi=300, bbox_inches='tight')
                print(f"    Saved individual plots to:\n      {os.path.basename(fig1_ind_path)}\n      {os.path.basename(fig2_ind_path)}")
                plt.close(fig_ind_eV)
                plt.close(fig_ind_nm) 
            except Exception as e:
                print(f"    Error saving individual plots: {e}")
                plt.close(fig_ind_eV)
                plt.close(fig_ind_nm)

        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")

    # --- NEW: Calculate 'Total Computed' shift ---
    if generate_total_computed and E_peak_exp is not None and total_computed_spectrum_sum is not None:
        if np.max(total_computed_spectrum_sum) > 0:
            E_peak_total_raw = x_axis[np.argmax(total_computed_spectrum_sum)]
            total_computed_shift = E_peak_exp - E_peak_total_raw
            print(f"  > Shift for 'Total Computed': {total_computed_shift:+.3f} eV")
        
    # 11. Plot the "Total Computed" spectrum (if requested)
    if generate_total_computed:
        print("  > Plotting summed computed spectrum...")
        max_val_total = np.max(total_computed_spectrum_sum)
        if max_val_total > 0:
            norm_total_computed_spectrum = total_computed_spectrum_sum / max_val_total
            
            # --- MODIFIED: Use auto-shift ---
            shift_eV_total = total_computed_shift
            label_eV_total = "Total Computed"
            label_nm_total = "Total Computed"
            
            if shift_eV_total != 0.0:
                label_eV_total = f"Total Computed (Shift: {shift_eV_total:+.2f} eV)"
                label_nm_total = f"Total Computed (Shift: {shift_eV_total:+.2f} eV)"

            scale_factor_total = scaling_factors.get('Total Computed', 1.0)
            scaled_total_spectrum = norm_total_computed_spectrum * scale_factor_total
            if scale_factor_total != 1.0:
                label_eV_total = f"{label_eV_total} (x{scale_factor_total:.2f})"
                label_nm_total = f"{label_nm_total} (x{scale_factor_total:.2f})"
                print(f"  > Applying scale factor {scale_factor_total:.2f} to 'Total Computed'.")

            x_axis_shifted_total = x_axis + shift_eV_total
            
            x_axis_nm_total = H_C_eV_nm / x_axis_shifted_total
            sort_idx_nm_total = np.argsort(x_axis_nm_total)
            x_plot_nm_total = x_axis_nm_total[sort_idx_nm_total]
            y_plot_nm_total = scaled_total_spectrum[sort_idx_nm_total]
            
            plot_color = 'red'
            
            ax_P3_eV.plot(x_axis_shifted_total, scaled_total_spectrum,
                          label=label_eV_total, linewidth=2.5, zorder=10, color=plot_color)
            ax_P3_eV.fill_between(x_axis_shifted_total, scaled_total_spectrum, 0,
                          color=plot_color, alpha=0.3, zorder=9)
            
            ax_P3_nm.plot(x_plot_nm_total, y_plot_nm_total,
                          label=label_nm_total, linewidth=2.5, zorder=10, color=plot_color)
            ax_P3_nm.fill_between(x_plot_nm_total, y_plot_nm_total, 0,
                          color=plot_color, alpha=0.3, zorder=9)
            
            if ax_P4_eV:
                ax_P4_eV.plot(x_axis_shifted_total, scaled_total_spectrum,
                                label=label_eV_total, linewidth=2.5, zorder=10, color=plot_color)
                ax_P4_eV.fill_between(x_axis_shifted_total, scaled_total_spectrum, 0,
                                color=plot_color, alpha=0.3, zorder=9)
                                
                ax_P4_nm.plot(x_plot_nm_total, y_plot_nm_total,
                                  label=label_nm_total, linewidth=2.5, zorder=10, color=plot_color)
                ax_P4_nm.fill_between(x_plot_nm_total, y_plot_nm_total, 0,
                                  color=plot_color, alpha=0.3, zorder=9)
        else:
            print("  > Warning: Sum of computed spectra is zero. Skipping 'Total Computed' plot.")
            figs_P3 = (None, None); axes_P3 = (None, None)
            figs_P4 = (None, None); axes_P4 = (None, None)

    # 12. Process and plot 'vib' files (Direct plotting, SHIFTED)
    print("\nProcessing 'vib' files...")
    for filepath in vib_files:
        filename = os.path.basename(filepath)
        label_name = os.path.splitext(filename)[0]
        
        try:
            energies, intensities, _ = load_and_convert_to_eV(filepath) 
            if energies is None: continue
                
            intensities_interp = np.interp(x_axis, energies, intensities, left=0, right=0)

            if generate_total_vibronic and total_vibronic_spectrum_sum is not None:
                total_vibronic_spectrum_sum += intensities_interp
            
            max_val = np.max(intensities_interp)
            if max_val == 0:
                print(f"Warning: Zero intensity for {filename}, skipping normalization.")
                continue
            normalized_intensities = intensities_interp / max_val 

            # --- MODIFIED: Get auto-shift ---
            shift_eV = vibronic_shifts.get(filepath, 0.0)
            label_eV = label_name
            label_nm = label_name
            
            if shift_eV != 0.0:
                label_eV = f"{label_name} (Shift: {shift_eV:+.2f} eV)"
                label_nm = f"{label_name} (Shift: {shift_eV:+.2f} eV)"

            energies_shifted = x_axis + shift_eV
            
            scale_factor = scaling_factors.get(filename, 1.0)
            scaled_intensities = normalized_intensities * scale_factor
            
            if scale_factor != 1.0:
                label_eV = f"{label_eV} (x{scale_factor:.2f})"
                label_nm = f"{label_nm} (x{scale_factor:.2f})"
                print(f"  > Applying scale factor {scale_factor:.2f} to {filename} for combined plot.")
            
            energies_nm_shifted = H_C_eV_nm / energies_shifted
            sort_idx_nm = np.argsort(energies_nm_shifted)
            x_plot_nm = energies_nm_shifted[sort_idx_nm]
            y_plot_nm = scaled_intensities[sort_idx_nm]
            
            style = 'solid'
            width = 1.5

            line, = axes_P1[0].plot(energies_shifted, scaled_intensities, 
                    label=label_eV, linestyle=style, linewidth=width, zorder=10)
            axes_P1[0].fill_between(energies_shifted, scaled_intensities, 0,
                    color=line.get_color(), alpha=0.3, zorder=9)
            
            axes_P1[1].plot(x_plot_nm, y_plot_nm,
                     label=label_nm, linestyle=style, linewidth=width,
                     color=line.get_color(), zorder=10)
            axes_P1[1].fill_between(x_plot_nm, y_plot_nm, 0,
                     color=line.get_color(), alpha=0.3, zorder=9)
                     
            if axes_P3[0]:
                axes_P3[0].plot(energies_shifted, scaled_intensities,
                                     label=label_eV, linestyle=style, linewidth=width,
                                     color=line.get_color(), zorder=10)
                axes_P3[0].fill_between(energies_shifted, scaled_intensities, 0,
                                     color=line.get_color(), alpha=0.3, zorder=9)
                                     
                axes_P3[1].plot(x_plot_nm, y_plot_nm,
                                     label=label_nm, linestyle=style, linewidth=width,
                                     color=line.get_color(), zorder=10)
                axes_P3[1].fill_between(x_plot_nm, y_plot_nm, 0,
                                     color=line.get_color(), alpha=0.3, zorder=9)

        except Exception as e:
            print(f"Error processing 'vib' file {filename}: {e}. Skipping.")

    # --- NEW: Calculate 'Total Vibronic' shift ---
    if generate_total_vibronic and E_peak_exp is not None and total_vibronic_spectrum_sum is not None:
        if np.max(total_vibronic_spectrum_sum) > 0:
            E_peak_total_vib = x_axis[np.argmax(total_vibronic_spectrum_sum)]
            total_vibronic_shift = E_peak_exp - E_peak_total_vib
            print(f"  > Shift for 'Total Vibronic': {total_vibronic_shift:+.3f} eV")

    # 12a. Plot the "Total Vibronic" spectrum (if requested)
    if generate_total_vibronic and total_vibronic_spectrum_sum is not None:
        print("  > Plotting summed vibronic spectrum...")
        max_val_total_vib = np.max(total_vibronic_spectrum_sum)
        if max_val_total_vib > 0:
            norm_total_vib_spectrum = total_vibronic_spectrum_sum / max_val_total_vib
            
            # --- MODIFIED: Use auto-shift ---
            shift_eV_vib_total = total_vibronic_shift
            label_eV_vib_total = "Total Vibronic"
            label_nm_vib_total = "Total Vibronic"
            
            if shift_eV_vib_total != 0.0:
                label_eV_vib_total = f"Total Vibronic (Shift: {shift_eV_vib_total:+.2f} eV)"
                label_nm_vib_total = f"Total Vibronic (Shift: {shift_eV_vib_total:+.2f} eV)"

            scale_factor_total_vib = scaling_factors.get('Total Vibronic', 1.0)
            scaled_total_vib_spectrum = norm_total_vib_spectrum * scale_factor_total_vib
            if scale_factor_total_vib != 1.0:
                label_eV_vib_total = f"{label_eV_vib_total} (x{scale_factor_total_vib:.2f})"
                label_nm_vib_total = f"{label_nm_vib_total} (x{scale_factor_total_vib:.2f})"
                print(f"  > Applying scale factor {scale_factor_total_vib:.2f} to 'Total Vibronic'.")

            x_axis_shifted_vib_total = x_axis + shift_eV_vib_total
            
            x_axis_nm_vib_total = H_C_eV_nm / x_axis_shifted_vib_total
            sort_idx_nm_vib_total = np.argsort(x_axis_nm_vib_total)
            x_plot_nm_vib_total = x_axis_nm_vib_total[sort_idx_nm_vib_total]
            y_plot_nm_vib_total = scaled_total_vib_spectrum[sort_idx_nm_vib_total]
            
            plot_color = 'cyan'
            
            if axes_P2[0]:
                axes_P2[0].plot(x_axis_shifted_vib_total, scaled_total_vib_spectrum,
                              label=label_eV_vib_total, linewidth=3.0, zorder=10, color=plot_color)
                axes_P2[0].fill_between(x_axis_shifted_vib_total, scaled_total_vib_spectrum, 0,
                              color=plot_color, alpha=0.3, zorder=9)
                                                    
                axes_P2[1].plot(x_plot_nm_vib_total, y_plot_nm_vib_total,
                              label=label_nm_vib_total, linewidth=3.0, zorder=10, color=plot_color)
                axes_P2[1].fill_between(x_plot_nm_vib_total, y_plot_nm_vib_total, 0,
                              color=plot_color, alpha=0.3, zorder=9)

            if axes_P4[0]:
                axes_P4[0].plot(x_axis_shifted_vib_total, scaled_total_vib_spectrum,
                                 label=label_eV_vib_total, linewidth=3.0, zorder=10, color=plot_color)
                axes_P4[0].fill_between(x_axis_shifted_vib_total, scaled_total_vib_spectrum, 0,
                                 color=plot_color, alpha=0.3, zorder=9)
                                         
                axes_P4[1].plot(x_plot_nm_vib_total, y_plot_nm_vib_total,
                                 label=label_nm_vib_total, linewidth=3.0, zorder=10, color=plot_color)
                axes_P4[1].fill_between(x_plot_nm_vib_total, y_plot_nm_vib_total, 0,
                                 color=plot_color, alpha=0.3, zorder=9)

        else:
            print("  > Warning: Sum of vibronic spectra is zero. Skipping 'Total Vibronic' plot.")
            figs_P2 = (None, None); axes_P2 = (None, None)
            figs_P4 = (None, None); axes_P4 = (None, None) 

    # 14. Finalize and save ALL plot sets
    print("\nFinalizing and saving all plots...")
    
    finalize_and_save_plot(figs_P1, axes_P1, "plot_set_1_IndVib_IndRaw", 
                           "Individual Vibronic vs. Individual Computed", 
                           plot_eV_limits, script_dir)
    
    finalize_and_save_plot(figs_P2, axes_P2, "plot_set_2_CombVib_IndRaw", 
                           "Combined Vibronic vs. Individual Computed", 
                           plot_eV_limits, script_dir)

    finalize_and_save_plot(figs_P3, axes_P3, "plot_set_3_IndVib_CombRaw", 
                           "Individual Vibronic vs. Combined Computed", 
                           plot_eV_limits, script_dir)

    finalize_and_save_plot(figs_P4, axes_P4, "plot_set_4_CombVib_CombRaw", 
                           "Combined Vibronic vs. Combined Computed", 
                           plot_eV_limits, script_dir)
    
    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()


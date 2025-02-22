def cooling_fit():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import pandas as pd
    import os
    from PIL import Image

    table_path = 'table_juist.txt'
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_lin_regress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2']    
    data = pd.read_csv(table_path, names=list_headers, header=0)
    sysn_list = data['n'].unique()
    sysN_list = data['N'].unique()
    sysCR_list = data['cr'].unique()
    systype_list = data['type'].unique()
    image_list = []

    for typeloop in range(len(systype_list)):
        for nloop in range(len(sysn_list)):
            for Nloop in range(len(sysN_list)):
                fig, ax = plt.subplots(figsize=(10, 5))
                data_cr_filtered = data[(data['type'] == systype_list[typeloop]) & (data['n'] == sysn_list[nloop]) & (data['N'] == sysN_list[Nloop])]
                cr_sorted = data_cr_filtered.sort_values(by='cr', ascending=True, ignore_index=True)

                # Filter out rows where 'cr' or 'opt_linregress' contains NaN or inf values
                cr_sorted = cr_sorted.replace([np.inf, -np.inf], np.nan).dropna(subset=['cr', 'hyper_fit'])

                # Ensure there are enough points for fitting
                if len(cr_sorted) < 2:
                    print(f"Not enough valid data points for system: {systype_list[typeloop]} n = {sysn_list[nloop]} N = {sysN_list[Nloop]}")
                    continue

                def funcCool(crdata, T0, B, A):
                    return T0 - (B / np.log10(A * crdata))

                # Dynamically determine initial guesses based on data ranges
                T0_initial = cr_sorted['hyper_fit'].mean()  # Use the mean of the opt_linregress as an initial guess for T0
                B_initial = (cr_sorted['hyper_fit'].max() - cr_sorted['hyper_fit'].min()) / 10  # Some scaled difference
                A_initial = 0.1  # Reasonable starting point for A
                initial_guesses_cool = [T0_initial, B_initial, A_initial]

                # Set tighter parameter bounds to avoid unrealistic behavior
                lower_bounds = [200, 1, 0.001]
                upper_bounds = [400, 300, 1]

                try:
                    # Fit cooling rate with bounds
                    popt_cool, pcov_cool = curve_fit(
                        funcCool, cr_sorted['cr'], cr_sorted['hyper_fit'], 
                        p0=initial_guesses_cool, bounds=(lower_bounds, upper_bounds), maxfev=1000000
                    )
                except Exception as e:
                    print(f"Fit failed for system: {systype_list[typeloop]}, n = {sysn_list[nloop]}, N = {sysN_list[Nloop]}. Error: {e}")
                    continue

                T_glass_cool_err = np.sqrt(np.diag(pcov_cool))[0]

                trialcr = np.linspace(cr_sorted['cr'].min(), cr_sorted['cr'].max(), 1000)
                d_fit_cool = funcCool(trialcr, *popt_cool)

                # Plot the data and fit
                ax.semilogx(cr_sorted['cr'], cr_sorted['hyper_fit'], 'o', label='Data', color='b')
                ax.semilogx(trialcr, d_fit_cool, 'r--', label='Fitted curve')  # Fixed redundant linestyle
                ax.set_ylabel(r"${T_g}$" + '(K)')
                ax.set_xlabel(r"${\dot{\gamma}}$" + " (K/ps)")
                ax.set_ylim([100, 400])
                ax.set_title("Cooling rate dependence for " + str(systype_list[typeloop]) + ' n = ' + str(round(sysn_list[nloop])) + ' N = ' + str(round(sysN_list[Nloop])))

                # Add text box for Tg infinity
                textstr = (r'T$_{g}^{\infty}$ = ' + str(round(popt_cool[0])) + ' K')
                props = dict(boxstyle='square', facecolor='green', alpha=0.6)
                ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)

                # Define the fig_name for saving
                fig_name = f'crvsT_{systype_list[typeloop]}_n_{round(sysn_list[nloop])}_N_{round(sysN_list[Nloop])}.png'
                
                os.chdir(r'D:\pythongraphs\crvsT')
                img_path = os.path.join(os.getcwd(), fig_name)
                image_list.append(img_path)
                plt.savefig(fig_name)
                plt.close()  # Close the figure to avoid memory issues

    # Create PDF file from saved images
    pdf_images = []
    for image in image_list:
        img = Image.open(image)
        pdf_images.append(img.convert('RGB'))  # Ensure proper format for PDF
        
    if pdf_images:
        pdf_path = 'D:/pythongraphs/crvsT.pdf'
        pdf_images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=pdf_images[1:])

    return

def cooling_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from PIL import Image

    table_path = 'table.txt'
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_lin_regress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2']
    data = pd.read_csv(table_path, names=list_headers, header=0)
    sysn_list = data['n'].unique()
    sysN_list = data['N'].unique()
    systype_list = data['type'].unique()
    image_list = []

    # List of cooling rates to exclude
    exclude_cr = [0.5, 0.05]

    for typeloop in range(len(systype_list)):
        for nloop in range(len(sysn_list)):
            for Nloop in range(len(sysN_list)):
                # Filter data for the current system and exclude specific cooling rates
                data_cr_filtered = data[(data['type'] == systype_list[typeloop]) &
                                        (data['n'] == sysn_list[nloop]) &
                                        (data['N'] == sysN_list[Nloop]) &
                                        (~data['cr'].isin(exclude_cr))]  # Exclude cooling rates

                cr_sorted = data_cr_filtered.sort_values(by='cr', ascending=True, ignore_index=True)

                # Proceed if data is not empty
                if not cr_sorted.empty:
                    fig, ax = plt.subplots(figsize=(7, 6))  # Adjusted size for consistency

                    # Remove NaN and infinite values
                    cr_sorted = cr_sorted.replace([np.inf, -np.inf], np.nan).dropna(subset=['cr', 'opt_linregress', 'opt_linregress_err'])

                    # Plot the data points with error bars on a semilog x-axis
                    ax.errorbar(
                        cr_sorted['cr'], cr_sorted['opt_linregress'], 
                        yerr=cr_sorted['opt_linregress_err'], fmt='o', 
                        color='#FF1493', markersize=8, capsize=5, elinewidth=1.5, capthick=1.5,
                        label="Data with error"
                    )

                    # Plot as semilog on x-axis
                    ax.semilogx(cr_sorted['cr'], cr_sorted['opt_linregress'], 'o', color='#FF1493')

                    # Style adjustments
                    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')
                    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray')
                    ax.set_ylabel(r"${T_g}$ (K)", fontsize=12)
                    ax.set_xlabel(r"Cooling Rate ($\dot{\gamma}$) (K/ps)", fontsize=12)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    #ax.set_ylim([100, 400])
                    ax.set_title(f"Cooling Rate Dependence for {systype_list[typeloop]} (n={int(sysn_list[nloop])}, N={int(sysN_list[Nloop])})", fontsize=14)
                    #ax.legend(loc='upper left', fontsize=10)

                    # Save the figure
                    os.makedirs('crvsT_plots', exist_ok=True)
                    fig_name = f'crvsT_{systype_list[typeloop]}_n_{int(sysn_list[nloop])}_N_{int(sysN_list[Nloop])}.png'
                    img_path = os.path.join('crvsT_plots', fig_name)
                    image_list.append(img_path)
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=300)
                    plt.close()

    # Create a PDF file from the saved images
    if image_list:
        pdf_images = []
        for image in image_list:
            img = Image.open(image)
            pdf_images.append(img.convert('RGB'))
        pdf_path = 'crvsT_plots/crvsT_plots.pdf'
        pdf_images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=pdf_images[1:])

    print("Plots have been saved and compiled into a PDF.")

# Call the function
cooling_plot()


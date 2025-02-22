def density():
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    N_p = {8: 3, 18: 8, 28: 13, 38: 18}

    table_path = 'table_final_pdms.txt'
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_lin_regress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2', 'shapiro']    
    data_pdms = pd.read_csv(table_path, names=list_headers, header=0)
    data_filtered_pdms = data_pdms[data_pdms['N']==100].copy()
    data_filtered_pdms.loc[:, 'N_p'] = data_filtered_pdms['n'].map(N_p)
    data_filtered_pdms = data_filtered_pdms.dropna(subset=['N_p'])
    data_filtered_pdms.loc[:, 'N_p'] = data_filtered_pdms['N_p'].astype(int)

    ax.scatter(data_filtered_pdms['N_p'], data_filtered_pdms['density'], marker='o', color='#00008B', label='PDMS')
    
    table_path = 'table.txt'
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_lin_regress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2', 'shapiro']    
    data_pdes = pd.read_csv(table_path, names=list_headers, header=0)
    data_filtered_pdes = data_pdes[data_pdes['N']==100].copy()
    data_filtered_pdes.loc[:, 'N_p'] = data_filtered_pdes['n'].map(N_p)
    data_filtered_pdes = data_filtered_pdes.dropna(subset=['N_p'])
    data_filtered_pdes.loc[:, 'N_p'] = data_filtered_pdes['N_p'].astype(int)

    ax.scatter(data_filtered_pdes['N_p'], data_filtered_pdes['density'], marker='o', color='#FF1493', label='PDES')
    ax.legend(loc='lower right', fontsize = 12)
    
    y_pdms = 1030
    y_pdes = 890
    text_x = (min(data_filtered_pdms['N_p']) + max(data_filtered_pdms['N_p'])) / 2 
    line_x_min = min(data_filtered_pdms['N_p']) - 1 
    line_x_max = max(data_filtered_pdms['N_p']) + 1
    text_padding = 1.5 

    ax.axhline(y=y_pdms, xmin=0, xmax=0.33, color='#00008B', linestyle="--", alpha=0.4)
    ax.axhline(y=y_pdms, xmin=0.66, xmax=1, color='#00008B', linestyle="--", alpha=0.4)
    ax.text(text_x, y_pdms, r'$\rho_{exp}^{PDMS} = 1030$ kg/m³', color='#00008B', fontsize=14, 
        ha='center', va='center')

    ax.axhline(y=y_pdes, xmin=0, xmax=0.33, color='#FF1493', linestyle="--", alpha=0.4)
    ax.axhline(y=y_pdes, xmin=0.66, xmax=1, color='#FF1493', linestyle="--", alpha=0.4)
    ax.text(text_x, y_pdes, r'$\rho_{exp}^{PDES} = 890$ kg/m³', color='#FF1493', fontsize=14, 
        ha='center', va='center')

    ax.set_xlabel(r'$N{_p}$', fontsize = 14)
    ax.set_ylabel("Density (kg/m³)")

    unique_Np = sorted(data_filtered_pdms['N_p'].unique())
    ax.set_xticks(unique_Np)
    ax.set_xticklabels([str(int(n)) for n in unique_Np], fontsize=16)

    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray')

    ax.tick_params(axis='both', which='major', labelsize=16)
    #plt.title("Density as a Function of Chain Length (n) (N=50)")

    plt.tight_layout()
    plt.savefig("density_Np.png", dpi=300)
    plt.legend()
    plt.grid(True)
    plt.show()

density()
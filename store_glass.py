def store_glass(sysType, sysn, sysN, sysCR, option, T_glass):
    import pandas as pd
    import os
    import numpy as np

    # Change to the right directory
    os.chdir(r'D:\pythongraphs')

    # Define headers
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_linregress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2', 'shapiro']    
    
    table_path = 'table.txt'
    
    if os.path.exists(table_path):
        table = pd.read_csv(table_path, names=list_headers, header=0,skip_blank_lines=True,on_bad_lines='skip')
    else:
        table = pd.DataFrame(columns=list_headers)

    table.reset_index(drop=True, inplace=True)
    
    sysn = float(sysn)
    sysN = float(sysN)
    
    if sysCR is not None:
        if sysCR[0] == '0':
            if isinstance(sysCR, str):
                sysCR = float(sysCR[0]+'.' + sysCR[1:])  # Ensure it's a proper float with a decimal point
        else:
            sysCR = float(sysCR)
        
        table['cr'] = pd.to_numeric(table['cr'], errors='coerce')
    #print(sysCR)

    table['n'] = pd.to_numeric(table['n'], errors='coerce')
    table['N'] = pd.to_numeric(table['N'], errors='coerce')
    

    if option in table.columns:
        table[option] = pd.to_numeric(table[option], errors='coerce')

    if sysCR is not None:
        filtered_table = table[
            (table['type'] == sysType) & 
            (np.isclose(table['n'], sysn)) & 
            (np.isclose(table['N'], sysN)) & 
            (np.isclose(table['cr'], sysCR))
        ]
    else:
        filtered_table = table[
            (table['type'] == sysType) & 
            (np.isclose(table['n'], sysn)) & 
            (np.isclose(table['N'], sysN))
        ]
        filtered_table = table[(table['type'] == sysType) & 
                               (table['n'] == sysn) & 
                           (table['N'] == sysN)]
    
    if filtered_table.empty:
        new_row = pd.DataFrame([{
            'type': sysType,
            'n': sysn,
            'N': sysN,
            'cr': np.nan if sysCR is None else sysCR,
            'density': np.nan,
            'lin_regress': np.nan,
            'lin_regress_err': np.nan,
            'lin_regress_r2': np.nan,
            'opt_linregress': np.nan,
            'opt_linregress_err': np.nan,
            'opt_linregress_r2': np.nan,
            'hyper_fit': np.nan,
            'hyper_fit_err': np.nan,
            'hyper_fit_r2': np.nan,
            'sqrt_fit': np.nan,
            'sqrt_fit_err': np.nan,
            'sqrt_fit_r2': np.nan,
            'piece_fit': np.nan,
            'piece_fit_err': np.nan,
            'piece_fit_r2': np.nan,
            'shapiro': np.nan,
            option: float(T_glass)
        }])

        
        table = pd.concat([table, new_row], ignore_index=True)
        #table = table._append(new_row, ignore_index=True)
        #print("No existing row found. Adding new row.")
        #new_row = {col: np.nan for col in list_headers}  # Initialize new row with NaN values
        #new_row.update({'type': sysType, 'n': sysn, 'N': sysN, 'cr': sysCR, option: float(T_glass)})  # Add the value for the specified option
        #print(new_row)
        #table = table._append(new_row, ignore_index=True)
        #new_index = len(table)
        #table.loc[new_index] = [sysType, sysn, sysN, sysCR, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #table.at[new_index, option] = float(T_glass)
    else:
        #ipdate the existing row
        filter_index = filtered_table.index[0]
        #print(filter_index, option)
        table.at[filter_index, option] = float(T_glass)
    
    if sysCR is None and T_glass is not None:
        table.loc[(table['type'] == sysType) & 
                  (np.isclose(table['n'], sysn)) & 
                  (np.isclose(table['N'], sysN)), option] = T_glass
    #sort the table
    table.sort_values(by=['type', 'n', 'N', 'cr'], ascending=True, inplace=True, ignore_index=True)
    #save the updated table to the same file
    table.to_csv(table_path, index=False, header=True, lineterminator='\n')
    #table_check = pd.read_csv(table_path, names=list_headers, header=0)
    return

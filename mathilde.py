# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:58:25 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Write participation matrix to LaTeX table or Excel

@author: chhil
"""

import xlwt
import numpy as np


def latex_P_matrix(P,state_labels,Matlab_array,filename,maximum_col,bold_limit):
    # P = Participation matrix
    # state_labels = labels of entries in the statevector
    # Matlab_array = True if the state_labels comes from a cell array in Matlab
    # filename = Name of the file to write the latex document in
        
    limit = bold_limit # Limit for making value bold
    n_elm = maximum_col # Number of elements pr. table
    n1 = (P.shape)[0] # Shape of P
    n2 = (P.shape)[1] # Shape of P
    N = int(np.ceil(n2/n_elm)) # Number of iteration
    n_hlines = 7 # Horizontal line pr. 6 values
    counter = 1 
    fid = open(filename,'w') # Open file with write access
    
    # Create a table pr. 4 eigenvalues
    for h in range(0,N):
        fid.write('\\begin{tabular}{')
        if h<N-1: # Number of elemest for all but last table            
            for k in range(0,n_elm):
                fid.write('l')
                if k == n_elm-1:
                    fid.write('|l}\r')
                
        
        else: # Last table should only be as wide as remaining eigenvalues            
            for k in range (0,n2-(N-1)*n_elm):
                    fid.write('l')
                    if k==n2-(N-1)*n_elm-1:
                        fid.write('|l}\r')
                   
        for k in range(0,n1):                    
            if h<N-1: # All but last has n_elm columns
                for l in range(h*n_elm,(h+1)*n_elm):
                    # Bold if abs larger than limit
                    if (np.abs(P[k,l])>=limit):
                        fid.write('$\\mathbf{(%.3f \\angle %.2f)}$\t& ' % (np.abs(P[k,l]),np.angle(P[k,l], deg=True)))                   
                    else:
                        fid.write('$(%.3f \\angle %.2f)$\t& ' % (np.abs(P[k,l]),np.angle(P[k,l], deg=True)))
                    
            else: # Last is only remaining values wide
                 for l in range(h*n_elm,n2):
                    # Bold if abs larger than limit
                    if (np.abs(P[k,l])>=limit):
                        fid.write('$\\mathbf{(%.3f \\angle %.2f)}$\t& ' % (np.abs(P[k,l]),np.angle(P[k,l], deg=True)))
                    else:
                        fid.write('$(%.3f \\angle %.2f)$\t& ' % (np.abs(P[k,l]),np.angle(P[k,l], deg=True)))
            # Labels for state vectors
            if (Matlab_array):
                fid.write('$%s$ \\\\ \r' % (state_labels[k,0][0]))
            else:
                fid.write('$%s$ \\\\ \r' % (state_labels[k]))
            
            # Horizontal line pr. n_hlines state variables
            if (k==n_hlines*counter-1 and k != n1-1):
                fid.write(' \\hline \r')                
                counter += 1
        # 2 horizontal lines at end        
        fid.write(' \\hline \r\\hline \r')
        
        # Put labels for eigenvalues below table
        if h<N-1:
            for l in range(1+h*n_elm,(h+1)*n_elm+1):
                fid.write('$\\qquad\\lambda_{%i}$\t&' % (l))   
            
        else:
             for l in range(1+h*n_elm,n2+1):
                 fid.write('$\\qquad\\lambda_{%i}$\t&' % (l))   
             
        # End table
        fid.write('\r\\end{tabular}\r\r')
        
        counter = 1
    
    # Close file
    fid.close()
    
    return


def excel_P_matrix(P,state_labels,Matlab_array,filename,bold_limit):
    # P = Participation matrix
    # state_labels = labels of entries in the statevector
    # Matlab_array = True if the state_labels comes from a cell array in Matlab
    # filename = Name of the file to write the latex document in
    
    limit = bold_limit # Limit for making value bold
    # Style of fields
    style = xlwt.easyxf(num_format_str='#,###0.000');
    style0 = xlwt.easyxf('font: bold on', num_format_str='#,###0.000');
    style_labels = xlwt.easyxf('border: left thin');    
    style_lambda = xlwt.easyxf('border: top double');
    
    # Create workbook
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Participation matrix')
    
    n1 = (P.shape)[0] # Shape of P
    n2 = (P.shape)[1] # Shape of P
    
    # Write magnitude of P matrix
    for i in range(0,n1):
        for j in range(0,n2):
            if np.abs(P[i,j]) > limit:
                ws.write(i, j, np.abs(P[i,j]),style0)
            else:
                ws.write(i, j, np.abs(P[i,j]),style)
            
        # Write labels of state vector to excel
        if (Matlab_array):
            ws.write(i, n2, state_labels[i,0][0],style_labels)
        else:
            ws.write(i, n2, state_labels[i],style_labels)    
    
    # Write lamba values to table
    for j in range(0,n2):
        name = u'\N{GREEK SMALL LETTER LAMDA}' +'_' + str(j+1)
        ws.write(n1, j,name,style_lambda)
    
    # Save workbook
    wb.save(filename)
    return


print("Hello")
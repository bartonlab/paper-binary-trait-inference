# LIBRARIES
import os
import sys
import numpy as np
import pandas as pd
import re
import urllib.request
from math import isnan
import glob
import subprocess
from sympy import Matrix,nsimplify

# GitHub
MPL_DIR = 'src'
HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'

NUC = ['-', 'A', 'C', 'G', 'T']
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'

# FUNCTIONS
def index2frame(i):
    """ Return the open reading frames corresponding to a given HXB2 index. """

    frames = []

    if ( 790<=i<=2292) or (5041<=i<=5619) or (8379<=i<=8469) or (8797<=i<=9417):
        frames.append(1)
    if (5831<=i<=6045) or (6062<=i<=6310) or (8379<=i<=8653):
        frames.append(2)
    if (2253<=i<=5096) or (5559<=i<=5850) or (5970<=i<=6045) or (6225<=i<=8795):
        frames.append(3)

    return frames

def codon2aa(c):
    """ Return the amino acid character corresponding to the input codon. """

    # If all nucleotides are missing, return gap
    if c[0]=='-' and c[1]=='-' and c[2]=='-': return '-'

    # Else if some nucleotides are missing, return '?'
    elif c[0]=='-' or c[1]=='-' or c[2]=='-': return '?'

    # If the first or second nucleotide is ambiguous, AA cannot be determined, return 'X'
    elif c[0] in ['W', 'S', 'M', 'K', 'R', 'Y'] or c[1] in ['W', 'S', 'M', 'K', 'R', 'Y']: return 'X'

    # Else go to tree
    elif c[0]=='T':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'F'
            elif  c[2] in ['A', 'G', 'R']: return 'L'
            else:                          return 'X'
        elif c[1]=='C':                    return 'S'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'Y'
            elif  c[2] in ['A', 'G', 'R']: return '*'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'C'
            elif  c[2]=='A':               return '*'
            elif  c[2]=='G':               return 'W'
            else:                          return 'X'
        else:                              return 'X'

    elif c[0]=='C':
        if   c[1]=='T':                    return 'L'
        elif c[1]=='C':                    return 'P'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'H'
            elif  c[2] in ['A', 'G', 'R']: return 'Q'
            else:                          return 'X'
        elif c[1]=='G':                    return 'R'
        else:                              return 'X'

    elif c[0]=='A':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'I'
            elif  c[2] in ['A', 'M', 'W']: return 'I'
            elif  c[2]=='G':               return 'M'
            else:                          return 'X'
        elif c[1]=='C':                    return 'T'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'N'
            elif  c[2] in ['A', 'G', 'R']: return 'K'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'S'
            elif  c[2] in ['A', 'G', 'R']: return 'R'
            else:                          return 'X'
        else:                              return 'X'

    elif c[0]=='G':
        if   c[1]=='T':                    return 'V'
        elif c[1]=='C':                    return 'A'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'D'
            elif  c[2] in ['A', 'G', 'R']: return 'E'
            else:                          return 'X'
        elif c[1]=='G':                    return 'G'
        else:                              return 'X'

    else:                                  return 'X'

def read_file(name):
    result = []  # initialize the result list
    p = open(HIV_DIR+'/input/'+name,'r')
    for line in p:
        # split the line into items and convert them to integers
        line_data = [int(item) for item in line.split()]
        result.append(line_data)
    p.close()
    return result

def read_file_s(name):
    result = [] # initialize the result list
    p = open(HIV_DIR+'/input/'+name,'r')
    for line in p:
        line_data = []  # store the data for each line
        for item in line.split():  # split the line into items
            # if the item contains a '/', split it into two integers and add them to the list
            if '/' in item:  
                line_data.append(list(map(int, item.split('/'))))
            # if the item does not contain a '/', add it to the list
            else:  
                line_data.append([int(item)])
        result.append(line_data)
    p.close()
    return result

# def get_unique_sequence(tag,i,j):
#     seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
#     #initial states
#     states = []
#     if len(j) == 1:
#         initial_states = [seq[0][i+2],seq[0][i+j[0]+2]]
#         states.append(initial_states)
#         for l in range(len(seq)):
#             temp_states = [seq[l][i+2],seq[l][i+j[0]+2]]
#             temp_states = np.trunc(temp_states)
#             temp_states = list(temp_states)
#             n = 0
#             for jj in range(len(states)):
#                 current_states = list(states[jj])
#                 if current_states != temp_states:
#                     n += 1
#             if n == len(states):
#                 states.append(temp_states)
#     elif len(j) == 2:
#         initial_states = [seq[0][i+2],seq[0][i+j[0]+2],seq[0][i+j[1]+2]]
#         states.append(initial_states)
#         for l in range(len(seq)):
#             temp_states = [seq[l][i+2],seq[l][i+j[0]+2],seq[l][i+j[1]+2]]
#             temp_states = np.trunc(temp_states)
#             temp_states = list(temp_states)
#             n = 0
#             for jj in range(len(states)):
#                 current_states = list(states[jj])
#                 if current_states != temp_states:
#                     n += 1
#             if n == len(states):
#                 states.append(temp_states)
#     else:
#         print('error')
#     return states

def get_frame(tag, poly, nuc, i_alig, i_HXB2, shift, TF_sequence,polymorphic_sites,poly_states):
    """ Return number of reading frames in which the input nucleotide is nonsynonymous in context, compared to T/F. """

    ns = []

    frames = index2frame(i_HXB2)
    
    match_states = poly_states[poly_states.T[polymorphic_sites.index(i_alig)]==NUC.index(nuc)]
    
    for fr in frames:

        pos = int((i_HXB2+shift-fr)%3) # position of the nucleotide in the reading frame
        TF_codon = [temp_nuc for temp_nuc in TF_sequence[i_alig-pos:i_alig-pos+3]]

        if len(TF_codon)<3:
            print('\tmutant at site %d in codon for CH%s that does not terminate in alignment' % (i_alig,tag[-5:]))

        else:
            mut_codon       = [a for a in TF_codon]
            mut_codon[pos]  = nuc
            replace_indices = [k for k in range(3) if (k+i_alig-pos) in polymorphic_sites and k!=pos]

            # If any other sites in the codon are polymorphic, consider mutation in context
            if len(replace_indices)>0:
                is_ns = False
                for s in match_states:
                    TF_codon = [temp_nuc for temp_nuc in TF_sequence[i_alig-pos:i_alig-pos+3]]
                    for k in replace_indices:
                        mut_codon[k] = NUC[s[polymorphic_sites.index(k+i_alig-pos)]]
                        TF_codon[k]  = NUC[s[polymorphic_sites.index(k+i_alig-pos)]]
                    if codon2aa(mut_codon) != codon2aa(TF_codon):
                        is_ns = True
                if is_ns:
                    ns.append(fr)

            elif codon2aa(mut_codon) != codon2aa(TF_codon):
                ns.append(fr)

    return ns

def get_trait(tag):
    
    df_info = pd.read_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    """Get escape sites"""
    # get all epitopes for one tag
    df_rows = df_info[df_info['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()

    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    escape_TF     = [] # corresponding wild type nucleotide
    epinames      = [] # name for binary trait

    for epi in unique_epitopes:
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()
        unique_sites = [int(site) for site in unique_sites]

        if len(df_e) > 1:# if there are more than escape mutation instead of escape site for this epitope
            epi_name = epi[0]+epi[-1]+str(len(epi))
            epinames.append(epi_name)
            escape_group.append(list(unique_sites))
            escape_TF_epi = []  
            for site in unique_sites:
                tf_values = []
                df_site = df_info[df_info['polymorphic_index'] == site]
                for i in range(len(df_site)):
                    if df_site.iloc[i].escape != True:

                        tf_values.append(int(NUC.index(df_site.iloc[i].nucleotide)))
                escape_TF_epi.append(tf_values)
            escape_TF.append(escape_TF_epi)

    return escape_group, escape_TF, epinames

def find_nons_mutations(tag):
    '''
    find trait sites and corresponding TF sequence and save the information into new csv file
    '''

    """Load the set of epitopes targeted by patients"""
    df_poly  = pd.read_csv('%s/notrait/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_epi   = pd.read_csv('%s/epitopes.csv'%HIV_DIR, comment='#', memory_map=True)
    df_index = pd.read_csv('%s/notrait/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    
    # TF sequence
    TF_sequence = []
    for i in range(len(df_index)):
        TF_sequence.append(df_index.iloc[i].TF)
    
    # alignment for polymorphic sites
    df_index_p  = df_index[df_index['polymorphic'].notna()]
    polymorphic_sites  = []
    for i in range(len(df_index_p)):
        polymorphic_sites.append(int(df_index_p.iloc[i].alignment))
    
    # sequence for polymorphic sites
    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    poly_times = np.zeros(len(seq))
    poly_states = np.zeros((len(seq),len(seq[0])-2),dtype=int)
    for i in range(len(seq)):
        poly_times[i] = int(seq[i][0])
        for j in range(len(seq[0])-2):
            poly_states[i][j] = int(seq[i][j+2])

    escape_values = ['False'] * len(df_poly)  # define the inserted column "escape"

    df_poly['epitope'].notna()

    for i in range(len(df_poly)):
        if pd.notna(df_poly.at[i, 'epitope']) and df_poly.iloc[i].nonsynonymous > 0 and df_poly.iloc[i].nucleotide != df_poly.iloc[i].TF:
            poly = int(df_poly.iloc[i].polymorphic_index)
            nons = df_poly.iloc[i].nonsynonymous
            i_alig = df_poly.iloc[i].alignment_index
            HXB2_s = df_poly.iloc[i].HXB2_index

            # get HXB2 index and shift
            try:
                i_HXB2 = int(HXB2_s)
                shift = 0
            except:
                i_HXB2 = int(HXB2_s[:-1])
                shift = ALPHABET.index(HXB2_s[-1]) + 1
            
            frames = index2frame(i_HXB2)

            """ judge if this site is trait site """
            if len(frames) == nons : #the mutation in this site is nonsynonymous in all reading frame
                escape_values[i] = 'True'
            else:
                """get the reading frame of the mutant site"""
                nuc = df_poly.iloc[i].nucleotide
                nonsfram = get_frame(tag, poly, nuc, i_alig, i_HXB2, shift, TF_sequence, polymorphic_sites, poly_states)

                """get the reading frame of the epitope"""
                df       = df_epi[(df_epi.epitope == df_poly.iloc[i].epitope)]
                epiframe = df.iloc[0].readingframe

                ''' decide whether this polymorphic site is nonsynonymous in its epitope reading frame '''
                if epiframe in nonsfram:
                    escape_values[i] = 'True'

    """modify the csv file and save it"""
    columns_to_remove = ['exposed', 'edge_gap','flanking','glycan']
    df_poly = df_poly.drop(columns=columns_to_remove)

    df_poly.insert(4, 'escape', escape_values)
    df_poly.to_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), index=False,na_rep='nan')

    escape_group, escape_TF,epinames = get_trait(tag)

    if len(escape_group)!= 0:
        print(f'CH{tag[-5:]} has {len(escape_group)} binary traits,', end = ' ')
        for n in range(len(escape_group)):
            print(f'epitope {epinames[n]} : {escape_group[n]},', end = ' ')
        print()
    else:
        print('%s has no bianry trait'%tag)

# loading data from dat file
def getSequence(history,escape_TF,escape_group):
    sVec      = []
    nVec      = []
    eVec      = []

    temp_sVec   = []
    temp_nVec   = []
    temp_eVec   = []

    times       = []
    time        = 0
    times.append(time)

    ne          = len(escape_group)

    for t in range(len(history)):
        if history[t][0] != time:
            time = history[t][0]
            times.append(int(time))
            sVec.append(temp_sVec)
            nVec.append(temp_nVec)
            eVec.append(temp_eVec)
            temp_sVec   = []
            temp_nVec   = []
            temp_eVec   = []

        temp_nVec.append(history[t][1])
        temp_sVec.append(history[t][2:])

        if ne > 0: # the patient contains escape group
            temp_escape = np.zeros(ne, dtype=int)
            for n in range(ne):
                for nn in range(len(escape_group[n])):
                    index = escape_group[n][nn] + 2
                    if history[t][index] not in escape_TF[n][nn]:
                        temp_escape[n] = 1
                        break
            temp_eVec.append(temp_escape)

        if t == len(history)-1:
            sVec.append(temp_sVec)
            nVec.append(temp_nVec)
            eVec.append(temp_eVec)

    return sVec,nVec,eVec

# get muVec
def getMutantS(seq_length, sVec):
    q = len(NUC)
    # use muVec matrix to record the index of time-varying sites
    muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
    x_length  = 0

    for i in range(seq_length):            
        # find all possible alleles in site i
        alleles     = [int(sVec[t][k][i]) for t in range(len(sVec)) for k in range(len(sVec[t]))]
        allele_uniq = np.unique(alleles)
        for allele in allele_uniq:
            muVec[i][int(allele)] = x_length
            x_length += 1

    return x_length,muVec

# calculate single and pair allele frequency (multiple case)
def get_allele_frequency(sVec,nVec,eVec,muVec,x_length):

    seq_length = len(muVec)
    ne         = len(eVec[0][0])

    x  = np.zeros((len(nVec),x_length))           # single allele frequency
    xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
    for t in range(len(nVec)):
        pop_size_t = np.sum([nVec[t]])
        for k in range(len(nVec[t])):
            # individual locus part
            for i in range(seq_length):
                qq = int(sVec[t][k][i])
                aa = int(muVec[i][qq])
                if aa != -1: # if aa = -1, it means the allele does not exist
                    x[t,aa] += nVec[t][k]
                    for j in range(int(i+1), seq_length):
                        qq = int(sVec[t][k][j])
                        bb = int(muVec[j][qq])
                        if bb != -1:
                            xx[t,aa,bb] += nVec[t][k]
                            xx[t,bb,aa] += nVec[t][k]
            # escape part
            for n in range(ne):
                aa = int(x_length-ne+n)
                x[t,aa] += eVec[t][k][n] * nVec[t][k]
                for m in range(int(n+1), ne):
                    bb = int(x_length-ne+m)
                    xx[t,aa,bb] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
                    xx[t,bb,aa] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
                for j in range(seq_length):
                    qq = int(sVec[t][k][j])
                    bb = int(muVec[j][qq])
                    if bb != -1:
                        xx[t,bb,aa] += eVec[t][k][n] * nVec[t][k]
                        xx[t,aa,bb] += eVec[t][k][n] * nVec[t][k]
        x[t,:]    = x[t,:]/pop_size_t
        xx[t,:,:] = xx[t,:,:]/pop_size_t
    return x,xx

# diffusion matrix C
def diffusion_matrix_at_t(x_0, x_1, xx_0, xx_1, dt, C_int):
    x_length = len(x_0)
    for i in range(x_length):
        dcov = dt * (((3 - (2 * x_1[i])) * (x_0[i] + x_1[i])) - 2 * x_0[i] * x_0[i]) / 6
        if abs(dcov) > 1:
            C_int[i, i] += round(dcov)
        if abs(C_int[i, i]) < 1:
                C_int[i, i] = 0

        for j in range(int(i+1) ,x_length):
            dCov1 = -dt * (2 * x_0[i] * x_0[j] + 2 * x_1[i] * x_1[j] + x_0[i] * x_1[j] + x_1[i] * x_0[j]) / 6
            dCov2 =  dt * (xx_0[i,j] + xx_1[i,j]) / 2
            dcov  = dCov1 + dCov2

            if abs(dcov) > 1:
                C_int[i, j] += round(dcov)
                C_int[j, i] += round(dcov)

            if abs(C_int[i, j]) < 1:
                C_int[i, j] = 0
                C_int[j, i] = 0

    return C_int

def determine_dependence_test(tag):

    # obtain raw sequence data
    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    escape_group, escape_TF, epinames = get_trait(tag)

    # information for escape group
    seq_length   = len(seq[0])-2
    ne           = len(escape_group)
    
    if ne == 0:
        f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
        g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')
        ff = open('%s/input/traitdis/traitdis-%s.dat'%(HIV_DIR,tag), 'w')
        f.close()
        g.close()
        ff.close()
        return
    
    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(seq,escape_TF,escape_group)
    x_length,muVec = getMutantS(seq_length, sVec)
    x_length      += ne

    # get all frequencies, 
    # x: single allele frequency, xx: pair allele frequency
    x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec,x_length)

    df_poly   = pd.read_csv('%s/interim/%s-poly.csv'%(HIV_DIR,tag), memory_map=True)

    Independent = [True] * ne

    for n in range(ne):
        
        n_index = []
        # mutation inside this epitope
        for nn in escape_group[n]:
            df_i = df_poly[df_poly['polymorphic_index'] == nn]
            for ii in range(len(df_i)):
                if df_i.iloc[ii].escape == True:
                    n_index.append(int(muVec[nn][NUC.index(df_i.iloc[ii].nucleotide)]))
        n_mutations = len(n_index)
        
        # mutation has the same frequency with the binary trait
        x_all = x.T
        variants_name = []

        # Individual locus part
        for i in range(len(x_all)-ne):
            if np.array_equal(x_all[i], x_all[x_length-ne+n]) or np.array_equal(x_all[i]+x_all[x_length-ne+n], np.ones(len(x))):
                # correlated or anti-correlated 
                # (sometimes one variant has more than 1 mutations, so we need to check anti_correlated)
                if i not in n_index: # make sure the variant is outside this epitope
                    n_index.append(i)
                    
                    # find the variant name
                    result = np.where(muVec == i)
                    variant = str(result[0][0]) + NUC[result[1][0]]
                    df_i = df_poly[df_poly['polymorphic_index'] == result[0][0]]
                    variant_name = str(variant) + '(' 
                    if pd.notna(df_i.iloc[0]['epitope']):
                        epi = df_i.iloc[0]['epitope']
                        variant_name += str(epi[0]) + str(epi[-1]) + str(len(epi)) + ', '
                    if df_i.iloc[0]['TF'] == NUC[result[1][0]]:
                        variant_name += 'WT)'
                    else:
                        variant_name += ')'
                    variants_name.append(variant_name)

        # Trait part
        for i in range(ne):
            if i != n and np.array_equal(x_all[x_length-ne+n], x_all[x_length-ne+i]):
                n_index.append(x_length-ne+i)
                variants_name.append(epinames[i])

        # epitope it self
        n_index.append(x_length-ne+n)

        n_length = len(n_index)
        x_n = x[:,n_index]   # single allele frequency
        xx_n  = np.zeros((len(nVec),n_length,n_length))  # pair allele frequency
        for t in range(len(nVec)):
            xx_t = xx[t]
            xx_n[t] = xx_t[np.ix_(n_index, n_index)]

        C_int = np.zeros((n_length,n_length))
        for t in range(1, len(x_n)):
            diffusion_matrix_at_t(x_n[t-1], x_n[t], xx_n[t-1], xx_n[t], 9e6, C_int)

        # ## print the output to check manually
        if len(variants_name) > 0:
            print(f'CH{tag[-5:]} {epinames[n]}\n{variants_name}\n{C_int}',end='\n')


def determine_dependence(tag):

    # obtain raw sequence data
    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    escape_group, escape_TF, epinames = get_trait(tag)

    # information for escape group
    seq_length   = len(seq[0])-2
    ne           = len(escape_group)
    
    if ne == 0:
        return
    
    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(seq,escape_TF,escape_group)
    x_length,muVec = getMutantS(seq_length, sVec)
    x_length      += ne

    # get all frequencies, 
    # x: single allele frequency, xx: pair allele frequency
    x,xx  = get_allele_frequency(sVec,nVec,eVec,muVec,x_length)

    C_int = np.zeros((x_length,x_length))
    for t in range(1,len(x)):
        diffusion_matrix_at_t(x[t-1], x[t], xx[t-1], xx[t], 9e6, C_int)
    
    # # Calculate the reduced row echelon form of the covariance matrix
    # sympy_matrix = Matrix(C_int)
    # rref_matrix, pivots = sympy_matrix.rref()
    # co_rr = np.array(rref_matrix).astype(float)

    # save the covariance matrix into a temporary file with integer
    np.savetxt('temp_cov.np.dat',C_int,fmt='%d')

    # run the c++ code to get the reduced row echelon form
    status = subprocess.run('./rref.out', shell=True)

    # load the reduced row echelon form
    co_rr = np.loadtxt('temp_rref.np.dat')
    ll = len(co_rr)
    
    # delete the temporary files
    status = subprocess.run('rm temp_*.dat', shell=True)

    pivots = []
    for row in range(ll):
        for col in range(ll):
            if co_rr[row, col] != 0:
                pivots.append(col)
                break

    df_poly   = pd.read_csv('%s/interim/%s-poly.csv'%(HIV_DIR,tag), memory_map=True)

    Independent = [True] * ne
    
    for n in range(ne):
        column_index = ll-ne+n
        if column_index not in pivots:
            
            Independent[n] = False
            print(f'CH{tag[-5:]} : trait {epinames[n]}, linked variants:', end=' ')
            
            '''find the linked variants'''
            related_columns = []
            for row in range(ll):
                # if the target column has a non-zero value in the current row, and the pivot column of this row is related
                if co_rr[row, column_index] != 0:
                    related_columns.append(pivots[row])

            for i in related_columns:
                if i < ll-ne:
                    result = np.where(muVec == i)
                    variant = str(result[0][0]) + NUC[result[1][0]]
                    df_i = df_poly[df_poly['polymorphic_index'] == result[0][0]]
                    
                    if pd.notna(df_i.iloc[0]['epitope']):
                        epi = df_i.iloc[0]['epitope']
                        print(f'{variant}({epi[0]}{epi[-1]}{len(epi)}', end='')
                    else:
                        print(f'{variant}(', end='')
                        
                    if df_i.iloc[0]['TF'] == NUC[result[1][0]]:
                        print(f', WT)', end=', ')
                    else:
                        print(f')', end=', ')
                            
                else:
                    nn = i - ll + ne
                    print(f'{epinames[nn]}', end=', ')
            print()

    "store the information for trait sites into files (trait site and TF trait sequences)"
    f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
    g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')
    ff = open('%s/input/traitdis/traitdis-%s.dat'%(HIV_DIR,tag), 'w')

    for n in range(ne):
        if Independent[n]:
            f.write('%s\n'%'\t'.join([str(i) for i in escape_group[n]]))
            for m in range(len(escape_group[n])):
                if m != len(escape_group[n]) - 1:
                    g.write('%s\t'%'/'.join([str(i) for i in escape_TF[n][m]]))
                else:
                    g.write('%s'%'/'.join([str(i) for i in escape_TF[n][m]]))
            g.write('\n')
    f.close()
    g.close()

    "store the information for trait sites into files (the number of normal sites between 2 trait sites)"
    df_sequence = pd.read_csv('%s/notrait/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True,usecols=['alignment','polymorphic'])
    for i in range(len(escape_group)):
        if Independent[i]:
            i_dis = []
            for j in range(len(escape_group[i])-1):
                index0 = df_sequence[df_sequence['polymorphic']==escape_group[i][j]].iloc[0].alignment
                index1 = df_sequence[df_sequence['polymorphic']==escape_group[i][j+1]].iloc[0].alignment
                i_dis.append(int(index1-index0))
            ff.write('%s\n'%'\t'.join([str(i) for i in i_dis]))
    ff.close()

def determine_dependence_new(tag):

    # obtain raw sequence data
    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    escape_group, escape_TF, epinames = get_trait(tag)

    # information for escape group
    seq_length   = len(seq[0])-2
    ne           = len(escape_group)
    
    if ne == 0:
        f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
        g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')
        ff = open('%s/input/traitdis/traitdis-%s.dat'%(HIV_DIR,tag), 'w')
        f.close()
        g.close()
        ff.close()
        return
    
    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(seq,escape_TF,escape_group)
    x_length,muVec = getMutantS(seq_length, sVec)
    x_length      += ne

    # get all frequencies, 
    # x: single allele frequency, xx: pair allele frequency
    x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec,x_length)

    df_poly   = pd.read_csv('%s/interim/%s-poly.csv'%(HIV_DIR,tag), memory_map=True)

    Independent = [True] * ne

    for n in range(ne):
        
        n_index = []
        # mutation inside this epitope
        for nn in escape_group[n]:
            df_i = df_poly[df_poly['polymorphic_index'] == nn]
            for ii in range(len(df_i)):
                if df_i.iloc[ii].escape == True:
                    n_index.append(int(muVec[nn][NUC.index(df_i.iloc[ii].nucleotide)]))
        n_mutations = len(n_index)
        
        # mutation has the same frequency with the binary trait
        x_all = x.T
        variants_name = []

        # Individual locus part
        for i in range(len(x_all)-ne):
            if np.array_equal(x_all[i], x_all[x_length-ne+n]) or np.array_equal(x_all[i]+x_all[x_length-ne+n], np.ones(len(x))):
                
                if i not in n_index: # make sure the variant is outside this epitope
                    n_index.append(i)
                    
                    # find the variant name
                    result = np.where(muVec == i)
                    variant = str(result[0][0]) + NUC[result[1][0]]
                    df_i = df_poly[df_poly['polymorphic_index'] == result[0][0]]
                    variant_name = str(variant)+'('
                    
                    if pd.notna(df_i.iloc[0]['epitope']):
                        epi = df_i.iloc[0]['epitope']
                        variant_name += str(epi[0]) + str(epi[-1]) + str(len(epi)) + ', '

                    if df_i.iloc[0]['TF'] == NUC[result[1][0]]:
                        variant_name += 'WT)'
                    else:
                        variant_name += ')'

                    variants_name.append(variant_name)

        # Trait part
        for i in range(ne):
            if i != n and np.array_equal(x_all[x_length-ne+n], x_all[x_length-ne+i]):
                n_index.append(x_length-ne+i)
                variants_name.append(epinames[i])

        # epitope it self
        n_index.append(x_length-ne+n)

        n_length = len(n_index)
        x_n = x[:,n_index]   # single allele frequency
        xx_n  = np.zeros((len(nVec),n_length,n_length))  # pair allele frequency
        for t in range(len(nVec)):
            xx_t = xx[t]
            xx_n[t] = xx_t[np.ix_(n_index, n_index)]

        C_int = np.zeros((n_length,n_length))
        for t in range(1, len(x_n)):
            diffusion_matrix_at_t(x_n[t-1], x_n[t], xx_n[t-1], xx_n[t], 9e6, C_int)

        # save the covariance matrix into a temporary file with 6 significant figures
        # np.savetxt('temp_cov.np.dat',C_int,fmt='%d')

        # # run the c++ code to get the reduced row echelon form
        # status = subprocess.run('./rref.out', shell=True)

        # # load the reduced row echelon form
        # co_rr = np.loadtxt('temp_rref.np.dat')
        # ll = len(co_rr)
        
        # # delete the temporary files
        # status = subprocess.run('rm temp_*.dat', shell=True)

        # pivots = []
        # for row in range(ll):
        #     for col in range(ll):
        #         if co_rr[row, col] != 0:
        #             pivots.append(col)
        #             break

        # Calculate the reduced row echelon form of the covariance matrix
        sympy_matrix = Matrix(C_int)
        sympy_matrix = sympy_matrix.applyfunc(nsimplify)
        rref_matrix, pivots = sympy_matrix.rref()
        co_rr = np.array(rref_matrix).astype(float)

        # ## print the output to check manually
        # print(f'{epinames[n]}\n{variants_name}\n{C_int}',end='\n')

        column_index = n_length-1
        if column_index not in pivots:
            Independent[n] = False
            print(f'CH{tag[-5:]} : trait {epinames[n]}, ({n_mutations} NS), linked variants:', end=' ')

            '''find the linked variants'''
            related_columns = []
            for row in range(n_length):
                # if the target column has a non-zero value in the current row, and the pivot column of this row is related
                if co_rr[row, column_index] != 0:
                    related_columns.append(pivots[row])

            for i in related_columns:
                if i < n_mutations:
                    site   = n_index[i]
                    result = np.where(muVec == site)
                    variant = str(result[0][0]) + NUC[result[1][0]]
                    print(f'{variant}', end=', ')
                elif i == n_length-1:
                    print(f'{epinames[n]}', end=', ')
                else:
                    print(f'{variants_name[i-n_mutations]}', end=', ')
            print()

    "store the information for trait sites into files (trait site and TF trait sequences)"
    f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
    g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')
    ff = open('%s/input/traitdis/traitdis-%s.dat'%(HIV_DIR,tag), 'w')

    for n in range(ne):
        if Independent[n]:
            f.write('%s\n'%'\t'.join([str(i) for i in escape_group[n]]))
            for m in range(len(escape_group[n])):
                if m != len(escape_group[n]) - 1:
                    g.write('%s\t'%'/'.join([str(i) for i in escape_TF[n][m]]))
                else:
                    g.write('%s'%'/'.join([str(i) for i in escape_TF[n][m]]))
            g.write('\n')
    f.close()
    g.close()

    "store the information for trait sites into files (the number of normal sites between 2 trait sites)"
    df_sequence = pd.read_csv('%s/notrait/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True,usecols=['alignment','polymorphic'])
    for i in range(len(escape_group)):
        if Independent[i]:
            i_dis = []
            for j in range(len(escape_group[i])-1):
                index0 = df_sequence[df_sequence['polymorphic']==escape_group[i][j]].iloc[0].alignment
                index1 = df_sequence[df_sequence['polymorphic']==escape_group[i][j+1]].iloc[0].alignment
                i_dis.append(int(index1-index0))
            ff.write('%s\n'%'\t'.join([str(i) for i in i_dis]))
    ff.close()

def get_independent():
    
    COV_DIR = HIV_DIR + '/output/covariance'
    
    flist = glob.glob('%s/c-*.dat'%COV_DIR)

    for f in flist:
        name = f.split('/')[-1]
        tag = name.split('.')[0]

        temp_cov = 'temp_cov.np.dat'
        temp_rr  = 'temp_rref.np.dat'
        out_rr   = '%s/rr-%s.dat' % (COV_DIR, tag)

        status = subprocess.run('cp %s %s' % (f, temp_cov), shell=True)
        status = subprocess.run('./rref.out', shell=True)
        status = subprocess.run('mv %s %s' % (temp_rr, out_rr), shell=True)

    status = subprocess.run('rm %s' % (temp_cov), shell=True)
    print('Done!')


# def find_trait_site(tag):
#     """get all epitopes for one tag"""
#     df_poly = pd.read_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
#     df_rows = df_poly[df_poly['epitope'].notna()]
#     unique_epitopes = df_rows['epitope'].unique()

#     trait_all   = []
#     trait_all_i = []

#     "store the information for trait sites into files (trait site and TF trait sequences)"
#     f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
#     g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')

#     for epi in unique_epitopes:

#         trait_sites = []
#         df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
#         trait_sites = df_e['polymorphic_index'].unique()

#         if len(df_e) > 1: # if there are more than escape mutation instead of escape site for this epitope
#             f.write('%s\n'%'\t'.join([str(i) for i in trait_sites]))
#             trait_all.append(trait_sites)
#             TF_seq  = []
#             for j in range(len(trait_sites)):
#                 n_poly  = df_e[df_e['polymorphic_index'] == trait_sites[j]]
#                 TF      = n_poly.iloc[0].TF
#                 TF_seq.append(NUC.index(TF))
#             g.write('%s\n'%'\t'.join([str(i) for i in TF_seq]))
#         elif len(trait_sites) > 0 :
#             trait_all_i.append(trait_sites)
#     f.close()
#     g.close()

#     if len(trait_all)!= 0 and len(trait_all_i) == 0:
#         print('-- %s has %d escape groups, they are %s'%(tag,len(trait_all),','.join([str(i) for i in trait_all])))
#     elif len(trait_all)== 0 and len(trait_all_i) != 0:
#         print('== %s has no escape group, the special sites are: %s'%(tag,','.join([str(i) for i in trait_all_i])))
#     elif len(trait_all) != 0 and len(trait_all_i) != 0:
#         print('++ %s has %d (+%d) escape groups, they are %s, the special sites are %s'
#               %(tag,len(trait_all),len(trait_all_i),','.join([str(i) for i in trait_all]),','.join([str(i) for i in trait_all_i])))
#     else:
#         print('   %s has no triat site'%tag)
    
def get_all_variants(seq,df_poly):
    all_variants = []
    TF_seq       = []
    L   = len(seq[0])-2
    for i in range(L):
        i_poly     = df_poly[df_poly['polymorphic_index'] == i]
        TF_nuc     = NUC.index(i_poly.iloc[0].TF)
        TF_seq.append(TF_nuc)
        for j in range(len(i_poly)):
            if i_poly.iloc[j].nucleotide != i_poly.iloc[j].TF:
                mut_allele = i_poly.iloc[j].nucleotide
                mut_site   = i
                variant    = str(mut_site) + '_' + str(mut_allele)
                all_variants.append(variant)
    if len(TF_seq) != len(seq[0])-2:
        print('there is something wrong with TF sequence')

    return all_variants,TF_seq

def analyze_result(tag,verbose=True):
    '''
    collect data and then write into csv file
    '''

    def get_xp(seq,traitsite,polyseq):
        times = []

        for i in range(len(seq)):
            times.append(seq[i][0])
        uniq_t = np.unique(times)
        xp    = np.zeros([len(traitsite),len(uniq_t)])

        for t in range(len(uniq_t)):
            tid = times==uniq_t[t]
            counts = np.sum(tid)
            seq_t = seq[tid][:,2:]
            for i in range(len(traitsite)):
                num = 0
                for n in range(len(seq_t)):
                    poly_value = sum([abs(seq_t[n][int(traitsite[i][j])]-polyseq[i][j]) for j in range(len(traitsite[i]))])
                    if poly_value > 0:
                        num += 1
                xp[i,t] = num/counts

        return xp

    def get_xp_s(seq,traitsite,traitallele):
        times = []

        for i in range(len(seq)):
            times.append(seq[i][0])
        uniq_t = np.unique(times)
        xp    = np.zeros([len(traitsite),len(uniq_t)])

        for t in range(len(uniq_t)):
            tid = times==uniq_t[t]
            counts = np.sum(tid)
            seq_t = seq[tid][:,2:]
            for i in range(len(traitsite)):
                num = 0
                for n in range(len(seq_t)):
                    
                    nonsyn = True
                    for j in range(len(traitsite[i])):
                        if seq_t[n][int(traitsite[i][j])] not in traitallele[i][j]:
                            nonsyn = False
                    if nonsyn == False:
                        num += 1
                
                xp[i,t] = num/counts

        return xp

    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    L       = len(seq[0])-2    #the number of polymorphic sites

    if verbose:
        sc      = np.loadtxt('%s/output/sc-%s.dat'%(HIV_DIR,tag))
    else:
        sc      = np.loadtxt('%s/noR/output/sc-%s.dat'%(HIV_DIR,tag))

    sc_old  = np.loadtxt('%s/notrait/output/sc-%s.dat'%(HIV_DIR,tag))

    try:
        traitsite = read_file('traitsite/traitsite-%s.dat'%(tag))
    except:
        traitsite = []
        print(f'{tag} does not have traitsite file')

    trait_sites = []
    for i in range(len(traitsite)):
        for j in range(len(traitsite[i])):
            trait_sites.append(traitsite[i][j])

    df_poly = pd.read_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    index_cols  = ['polymorphic_index', 'alignment_index', 'HXB2_index','nonsynonymous','escape','nucleotide',]
    index_cols += ['TF','consensus','epitope','exposed','edge_gap','flanking','s_MPL','s_SL']
    cols = [i for i in list(df_poly) if i not in index_cols]
    times = [int(cols[i].split('_')[-1]) for i in range(len(cols))]

    if verbose:
        f = open(HIV_DIR+'/analysis/'+tag+'-analyze.csv','w')
    else:
        f = open(HIV_DIR+'/noR/analysis/'+tag+'-analyze.csv','w')
    f.write('polymorphic_index,alignment,HXB2_index,nucleotide,TF,consensus,epitope,escape,sc_old,sc_MPL,tc_MPL')
    f.write(',%s' % (','.join(cols)))
    f.write('\n')

    for ii in range(len(df_poly)):
        polymorphic_index = df_poly.iloc[ii].polymorphic_index
        alignment         = df_poly.iloc[ii].alignment_index
        HXB2_index        = df_poly.iloc[ii].HXB2_index
        nucleotide        = df_poly.iloc[ii].nucleotide
        TF                = df_poly.iloc[ii].TF
        consensus         = df_poly.iloc[ii].consensus
        epitope           = df_poly.iloc[ii].epitope
        escape            = df_poly.iloc[ii].escape

        # get selection coefficient
        nuc_index  = NUC.index(nucleotide)+polymorphic_index*5
        TF_index   = NUC.index(TF)+polymorphic_index*5
        sc_MPL     = sc[nuc_index]-sc[TF_index]
        sc_mpl_old = sc_old[nuc_index]-sc_old[TF_index]
        tc_MPL     = 'nan'
        df_i       = df_poly.iloc[ii]
        if sc_MPL != 0:
            for i in range(len(traitsite)):
                if polymorphic_index in traitsite[i]:
                    tc_MPL = sc[i+L*5]
        f.write('%d,%d,%s,%s,' % (polymorphic_index, alignment, HXB2_index, nucleotide))
        f.write('%s,%s,%s,%s,%s,%s,%s' % (TF, consensus, epitope, escape, sc_mpl_old, sc_MPL, tc_MPL))
        f.write(',%s' % (','.join([str(df_i[c]) for c in cols])))
        f.write('\n')
    f.close()

    if len(traitsite) != 0:
        if verbose:
            df = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        else:
            df = pd.read_csv('%s/noR/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

        index_cols = ['polymorphic_index', 'alignment']
        cols = [i for i in list(df) if i not in index_cols]

        polyseq  = read_file_s('traitseq/traitseq-'+tag+'.dat')
        xp = get_xp_s(seq,traitsite,polyseq)

        if verbose:
            g = open('%s/group/escape_group-%s.csv'%(HIV_DIR,tag),'w')
        else:
            g = open('%s/noR/group/escape_group-%s.csv'%(HIV_DIR,tag),'w')
        g.write('polymorphic_index')
        g.write(',%s' % (','.join(cols)))
        for t in range(len(times)):
            g.write(',xp_at_%s'%times[t])
        g.write('\n')

        for i in range(len(traitsite)):
            for j in range(len(traitsite[i])):
                df_poly = df[(df.polymorphic_index == traitsite[i][j]) & (df.nucleotide != df.TF) & (df.escape == True)]
                for n in range(len(df_poly)):
                    g.write('%d' %traitsite[i][j])
                    g.write(',%s' % (','.join([str(df_poly.iloc[n][c]) for c in cols])))
                    for t in range(len(times)):
                        g.write(',%f'%xp[i,t])
                    g.write('\n')

def modify_seq(tag):
    '''
    change sequence and escape information to calculate Δsij, reverting mutant
    variant into wild type for individual locus part and thowing out one escape
    group for escape part.
    '''

    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    df_poly = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
    L   = len(seq[0])-2

    all_variants,TF_seq    = get_all_variants(seq,df_poly)
    trait_sites = read_file('traitsite/traitsite-'+tag+'.dat')

    g = open('%s/HIV_sij.sh'%(MPL_DIR), "a")
    g.write('g++ main.cpp inf.cpp io.cpp -O3 -mcpu=apple-a14 -std=c++11 -lgsl -lgslcblas -o mpl\n')

    for i in range(len(all_variants)):
        variant   = all_variants[i]
        trait_site = int(variant.split('_')[0])
        mut_alle  = variant.split('_')[-1]
        mut_index = NUC.index(mut_alle)

        g.write('./mpl -d ../data/HIV -i input/sequence/%s/%s.dat '%(tag,variant))
        g.write('-o output/%s/sc_%s.dat -g 10 -m input/Zanini-extended.dat -r input/r_rates/r-%s.dat '%(tag,variant,tag))
        g.write('-e input/traitsite/traitsite-%s.dat -es input/traitseq/traitseq-%s.dat '%(tag,tag))
        g.write('-ed input/traitdis/traitdis-%s.dat\n'%(tag))

        f = open('%s/input/sequence/%s/%s.dat'%(HIV_DIR,tag,all_variants[i]), "w")
        for j in range(len(seq)):
            seq_modi  = seq[j]
            poly_states = []
            for ii in range(L):
                if ii == trait_site and seq_modi[ii+2] == mut_index:
                    poly_states.append(str(TF_seq[trait_site]))
                else:
                    poly_states.append(str(int(seq_modi[ii+2])))
            f.write('%d\t1\t%s\n'%(seq_modi[0],' '.join(poly_states)))
        f.close()

    for n in range(len(trait_sites)):
        variant   = 'epi'+str(int(n))
        
        g.write('./mpl -d ../data/HIV -i input/sequence/%s/%s.dat '%(tag,variant))
        g.write('-o output/%s/sc_%s.dat -g 10 -m input/Zanini-extended.dat -r input/r_rates/r-%s.dat '%(tag,variant,tag))
        g.write('-e input/traitsite/traitsite-%s.dat -es input/traitseq/traitseq-%s.dat '%(tag,tag))
        g.write('-ed input/traitdis/traitdis-%s.dat\n'%(tag))
            
        f = open('%s/input/sequence/%s/%s.dat'%(HIV_DIR,tag,variant), "w")
        for j in range(len(seq)):
            seq_modi  = seq[j]
            poly_states = []
            for ii in range(L):
                if ii in trait_sites[n]:
                    poly_states.append(str(TF_seq[ii]))
                else:
                    poly_states.append(str(int(seq_modi[ii+2])))
            f.write('%d\t1\t%s\n'%(seq_modi[0],' '.join(poly_states)))
        f.close()

    g.close()

def cal_sij(tag):

    df_poly = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    all_variants,TF_seq  = get_all_variants(seq,df_poly)
    traitsite = read_file('traitsite/traitsite-'+tag+'.dat')

    fsij    = open('%s/sij/%s-sij.csv'%(HIV_DIR,tag),'w')
    fsij.write('mask_polymorphic_index,mask_nucleotide,target_polymorphic_index,target_nucleotide,effect,distance\n')

    trait_alle = 'polypart'

    for i in range(len(all_variants)):
        #get site and allele for mask variant
        variant = all_variants[i]
        sc      = np.loadtxt('%s/output/%s/sc_%s.dat'%(HIV_DIR,tag,variant))
        trait_site_i = int(variant.split('_')[0])
        mut_alle_i  = variant.split('_')[-1]

        i_poly     = df_poly[df_poly['polymorphic_index'] == trait_site_i]
        i_align    = i_poly.iloc[0].alignment #get aligment for this site

        # individual site part
        for j in range(len(all_variants)):
            trait_site_j = int(all_variants[j].split('_')[0])
            mut_alle_j  = all_variants[j].split('_')[-1]

            j_poly    = df_poly[df_poly['polymorphic_index'] == trait_site_j]
            j_align   = j_poly.iloc[0].alignment
            dis       = abs(int(j_align)-int(i_align))#distance between mask variant and target variant, > 0

            if trait_site_i != trait_site_j:
                jj_poly    = j_poly[j_poly['nucleotide'] == mut_alle_j]
                if len(jj_poly)>0:
                    j_TF       = jj_poly.iloc[0].TF
                    nuc_index  = NUC.index(mut_alle_j) + trait_site_j*5
                    TF_index   = NUC.index(j_TF) + trait_site_j*5
                    sc_j_0     = jj_poly.iloc[0].sc_MPL
                    sc_j_1     = sc[nuc_index]-sc[TF_index]
                    sij        = sc_j_0 - sc_j_1
                else:
                    print('wrong with %s,site is %d,allele is %s'%(all_variants[j],trait_site_j,mut_alle_j))
            else:
                sij = 0 #let sij = 0 for the same sites

            fsij.write('%s,%s,%s,%s,%s,%s\n'%(trait_site_i,mut_alle_i,trait_site_j,mut_alle_j,sij,dis))

        #escape part
        dis       = 'nan'
        for j in range(len(traitsite)):
            variant_j = 'epi'+str(int(j))
            p_poly    = df_poly[df_poly['polymorphic_index'] == traitsite[j][0]]
            pp_poly   = p_poly[p_poly['nucleotide'] != p_poly['TF']]
            tc_j_0    = pp_poly.iloc[0].tc_MPL
            tc_j_1    = sc[j-len(traitsite)]
            sij       = tc_j_0 - tc_j_1
            fsij.write('%s,%s,%s,%s,%s,%s\n'%(trait_site_i,mut_alle_i,variant_j,trait_alle,sij,dis))

    for i in range(len(traitsite)):
        #get site and allele for mask variant
        variant_i = 'epi'+str(int(i))
        sc        = np.loadtxt('%s/output/%s/sc_epi%d.dat'%(HIV_DIR,tag,i))
        dis       = 'nan'
        # individual site part
        for j in range(len(all_variants)):
            trait_site_j = int(all_variants[j].split('_')[0])
            mut_alle_j  = all_variants[j].split('_')[-1]

            j_poly    = df_poly[df_poly['polymorphic_index'] == trait_site_j]
            j_align   = j_poly.iloc[0].alignment

            if trait_site_j not in traitsite[i]:
                jj_poly    = j_poly[j_poly['nucleotide'] == mut_alle_j]
                if len(jj_poly)>0:
                    j_TF       = jj_poly.iloc[0].TF
                    nuc_index  = NUC.index(mut_alle_j) + trait_site_j*5
                    TF_index   = NUC.index(j_TF) + trait_site_j*5
                    sc_j_0     = jj_poly.iloc[0].sc_MPL
                    sc_j_1     = sc[nuc_index]-sc[TF_index]
                    sij        = sc_j_0 - sc_j_1
                else:
                    print('wrong with %s,site is %d,allele is %s'%(all_variants[j],trait_site_j,mut_alle_j))
            else:
                sij = 0 #let sij = 0 for the sites within the epitope

            fsij.write('%s,%s,%s,%s,%s,%s\n'%(variant_i,trait_alle,trait_site_j,mut_alle_j,sij,dis))

        #escape part
        for j in range(len(traitsite)):
            variant_j = 'epi'+str(int(j))
            p_poly    = df_poly[df_poly['polymorphic_index'] == traitsite[j][0]]
            pp_poly   = p_poly[p_poly['nucleotide'] != p_poly['TF']]
            if j != i :
                tc_j_0  = pp_poly.iloc[0].tc_MPL
                tc_j_1  = sc[j-len(traitsite)]
                sij     = tc_j_0 - tc_j_1
            else:
                sij = 0
            fsij.write('%s,%s,%s,%s,%s,%s\n'%(variant_i,trait_alle,variant_j,trait_alle,sij,dis))

    fsij.close()

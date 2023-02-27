# LIBRARIES
import os
import sys
import numpy as np
import pandas as pd
import re
import urllib.request
from math import isnan

# GitHub
MPL_DIR = 'src'
HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'

NUC = ['-', 'A', 'C', 'G', 'T']

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
    poly = []
    p = open(HIV_DIR+'/input/'+name,'r')
    maxlen = 0
    maxnum = 0
    for line in p:
        temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
        data = [float(item) for item in temp]
        poly.append(data)
        #print(data)
        if len(data) > maxlen:
            maxlen = len(data)
        maxnum += 1
    arr = np.zeros((maxnum, maxlen))*np.nan
    for cnt in np.arange(maxnum):
        arr[cnt, 0:len(poly[cnt])] = np.array(poly[cnt])
    p.close()
    for i in range(len(poly)):
        for j in range(len(poly[i])):
            poly[i][j] = int(poly[i][j])
    return poly

def get_unique_sequence(tag,i,j):
    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    #initial states
    states = []
    if len(j) == 1:
        initial_states = [seq[0][i+2],seq[0][i+j[0]+2]]
        states.append(initial_states);
        for l in range(len(seq)):
            temp_states = [seq[l][i+2],seq[l][i+j[0]+2]];
            temp_states = np.trunc(temp_states)
            temp_states = list(temp_states)
            n = 0;
            for jj in range(len(states)):
                current_states = list(states[jj])
                if current_states != temp_states:
                    n += 1;
            if n == len(states):
                states.append(temp_states)
    elif len(j) == 2:
        initial_states = [seq[0][i+2],seq[0][i+j[0]+2],seq[0][i+j[1]+2]]
        states.append(initial_states);
        for l in range(len(seq)):
            temp_states = [seq[l][i+2],seq[l][i+j[0]+2],seq[l][i+j[1]+2]];
            temp_states = np.trunc(temp_states)
            temp_states = list(temp_states)
            n = 0;
            for jj in range(len(states)):
                current_states = list(states[jj])
                if current_states != temp_states:
                    n += 1;
            if n == len(states):
                states.append(temp_states)
    else:
        print('error')
    return states

def get_frame(tag, polymorphic_sites, nuc):
    """ Return number of reading frames in which the input nucleotide is nonsynonymous in context, compared to T/F. """

    df_sequence = pd.read_csv('%s/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True,usecols=['alignment','polymorphic','HXB2','TF']);
    ns = [];

    n_frame = df_sequence[df_sequence['polymorphic'] == polymorphic_sites]
    i = n_frame.iloc[0].alignment # index for the polymorphic site in index file
    i_HXB2 = int(n_frame.iloc[0].HXB2)
    frames = index2frame(i_HXB2);

    for fr in frames:

        pos = int((i_HXB2-fr)%3) # position of the nucleotide in the reading frame
        TF_codon = df_sequence.iloc[i-pos].TF + df_sequence.iloc[i-pos+1].TF + df_sequence.iloc[i-pos+2].TF
        TF_codon = [a for a in TF_codon]

        if len(TF_codon)<3 and verbose:
            print('\tmutant at site %d in codon that does not terminate in alignment, assuming syn' % i)

        else:
            mut_codon = [a for a in TF_codon]
            mut_codon[pos] = nuc
            replace_indices = [k for k in range(3) if isnan(df_sequence.iloc[k+i-pos].polymorphic) != True and k!=pos]

            # If any other sites in the codon are polymorphic, consider mutation in context
            if len(replace_indices) > 0:
                dd = len(replace_indices) + 1
                #print('there are %d polymorphic sites in this codon'%dd)
                k_index = [];

                if len(replace_indices) == 2:
                    k_1 = replace_indices[0]
                    k_2 = replace_indices[1]
                    k_index.append(k_1-pos);
                    k_index.append(k_2-pos);
                elif len(replace_indices) == 1:
                    k = replace_indices[0]
                    if   pos == 0:
                        k_index.append(1);
                    elif pos == 2:
                        k_index.append(-1);
                    elif pos == 1:
                        k_index.append(k-1);
                else:
                    print('error, there are more than 3 nucleotides in one codon')

                is_ns = False
                match_states = get_unique_sequence(tag,polymorphic_sites,k_index);
                for s in match_states:
                    TF_codon = TF_codon = df_sequence.iloc[i-pos].TF + df_sequence.iloc[i-pos+1].TF + df_sequence.iloc[i-pos+2].TF
                    TF_codon = [a for a in TF_codon]
                    # possible codon
                    if len(replace_indices) == 1:
                        mut_codon[k] = NUC[int(s[1])]
                        TF_codon[k]  = NUC[int(s[1])]
                        ss = [NUC.index(mut_codon[pos]),NUC.index(mut_codon[k])]
                    elif len(replace_indices) == 2:
                        mut_codon[k_1] = NUC[int(s[1])]
                        TF_codon[k_1]  = NUC[int(s[1])]
                        mut_codon[k_2] = NUC[int(s[2])]
                        TF_codon[k_2]  = NUC[int(s[2])]
                        ss = [NUC.index(mut_codon[pos]),NUC.index(mut_codon[k_1]),NUC.index(mut_codon[k_2])]
                    #check if this codon can be observed
                    if ss in match_states:
                        if codon2aa(mut_codon)!=codon2aa(TF_codon):
                            is_ns = True
                if is_ns:
                    ns.append(fr)
            elif codon2aa(mut_codon) != codon2aa(TF_codon):
                ns.append(fr)
    return ns

def find_site(tag,min_n):
    '''
    find escape sites and corresponding TF sequence
    '''

    # Load the set of epitopes targeted by patients
    df_poly    = pd.read_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True,usecols=['polymorphic_index','HXB2_index','nonsynonymous','nucleotide','TF','epitope'])
    df_epi   = pd.read_csv('%s/epitopes.csv'%HIV_DIR, comment='#', memory_map=True)

    # get all epitopes for one tag
    epi_list   = [];
    for i in range(len(df_poly)):
        if type(df_poly.iloc[i].epitope) == str:
            if df_poly.iloc[i].epitope not in epi_list:
                epi_list.append(df_poly.iloc[i].epitope)


    f = open('%s/input/polysite/%s-polysite.dat'%(HIV_DIR,tag), 'w')
    g = open('%s/input/polyseq/polysequence-%s.dat'%(HIV_DIR,tag), 'w')
    poly_all   = [];
    for n in range(len(epi_list)):
        poly_sites = [];
        for i in range(len(df_poly)):
            if df_poly.iloc[i].epitope == epi_list[n] and df_poly.iloc[i].nonsynonymous > 0:
                poly = int(df_poly.iloc[i].polymorphic_index);
                nons = df_poly.iloc[i].nonsynonymous;

                """ get the corresponding reading frame"""
                HXB2_s = df_poly.iloc[i].HXB2_index
                if isinstance(HXB2_s, str) == True:
                    HXB2_1 = re.findall('\d+', HXB2_s)
                    HXB2_i = int(int(HXB2_1[0]))
                elif isinstance(HXB2_s, (int,np.int32,np.int64)) == True:
                    HXB2_i = int(HXB2_s)

                else:
                    print('escape site %d in %s needs double check (data type is)' %(poly_new,tag))
                    print(type(HXB2_s))
                    break

                frames = index2frame(HXB2_i)

                """ judge if this site is escape site """
                if len(frames) == nons :
                    poly_sites.append(poly);
                else:
                    """get the reading frame of the mutant site"""
                    nuc = df_poly.iloc[i].nucleotide
                    TF  = df_poly.iloc[i].TF
                    nonsfram = get_frame(tag, poly, nuc);

                    """get the reading frame of the epitope"""
                    df       = df_epi[(df_epi.epitope == epi_list[n])]
                    epiframe = df.iloc[0].readingframe

                    ''' decide whether this polymorphic site is nonsynonymous in its epitope reading frame '''
                    if epiframe in nonsfram:
                        poly_sites.append(poly);
        poly_sites = np.unique(poly_sites)
        if len(poly_sites) > min_n:
            f.write('%s\n'%'\t'.join([str(i) for i in poly_sites]))
            poly_all.append(poly_sites)
            TF_seq  = []
            for j in range(len(poly_sites)):
                n_poly  = df_poly[df_poly['polymorphic_index'] == poly_sites[j]]
                TF      = n_poly.iloc[0].TF
                TF_seq.append(NUC.index(TF))
            g.write('%s\n'%'\t'.join([str(i) for i in TF_seq]))
    f.close()
    g.close()

    n_pol = len(poly_all)
    if n_pol != 0:
        print('%s has %d escape groups, they are %s'%(tag,n_pol,' '.join([str(i) for i in poly_all])))
    else:
        print('%s has no escape site'%tag)

def get_all_variants(seq,df_poly):
    all_variants = [];
    TF_seq       = [];
    L   = len(seq[0])-2
    for i in range(L):
        i_poly     = df_poly[df_poly['polymorphic_index'] == i];
        TF_nuc     = NUC.index(i_poly.iloc[0].TF)
        TF_seq.append(TF_nuc)
        for j in range(len(i_poly)):
            if i_poly.iloc[j].nucleotide != i_poly.iloc[j].TF:
                mut_allele = i_poly.iloc[j].nucleotide;
                mut_site   = i;
                variant    = str(mut_site) + '_' + str(mut_allele);
                all_variants.append(variant)
    if len(TF_seq) != len(seq[0])-2:
        print('there is something wrong with TF sequence')

    return all_variants,TF_seq


def analyze_result(tag):
    '''
    collect data and then write into csv file
    '''

    def get_xp(seq,polysite,polyseq):
        times = [];

        for i in range(len(seq)):
            times.append(seq[i][0])
        uniq_t = np.unique(times)
        xp    = np.zeros([len(polysite),len(uniq_t)])

        for t in range(len(uniq_t)):
            tid = times==uniq_t[t]
            counts = np.sum(tid)
            seq_t = seq[tid][:,2:]
            for i in range(len(polysite)):
                num = 0
                for n in range(len(seq_t)):
                    poly_value = sum([abs(seq_t[n][int(polysite[i][j])]-polyseq[i][j]) for j in range(len(polysite[i]))])
                    if poly_value > 0:
                        num += 1
                xp[i,t] = num/counts

        return xp

    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag));
    L       = len(seq[0])-2;    #the number of polymorphic sites

    sc      = np.loadtxt('%s/output/sc-%s.dat'%(HIV_DIR,tag))
    sc_old  = np.loadtxt('%s/output_nopoly/sc-%s.dat'%(HIV_DIR,tag))

    polysite = read_file('polysite/%s-polysite.dat'%(tag))

    poly_sites = []
    for i in range(len(polysite)):
        for j in range(len(polysite[i])):
            poly_sites.append(polysite[i][j])

    df_poly = pd.read_csv('%s/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    index_cols  = ['polymorphic_index', 'alignment_index', 'HXB2_index','nonsynonymous','nucleotide', 'TF',]
    index_cols += [ 'consensus','epitope','exposed','edge_gap','flanking','glycan','s_MPL','s_SL']
    cols = [i for i in list(df_poly) if i not in index_cols]
    times = [int(cols[i].split('_')[-1]) for i in range(len(cols))]

    f = open(HIV_DIR+'/analysis/'+tag+'-analyze.csv','w')
    f.write('polymorphic_index,alignment,HXB2_index,nucleotide,TF,epitope,sc_old,sc_MPL,pc_MPL')
    f.write(',%s' % (','.join(cols)))
    f.write('\n')

    for ii in range(len(df_poly)):
        polymorphic_index = df_poly.iloc[ii].polymorphic_index;
        alignment         = df_poly.iloc[ii].alignment_index;
        HXB2_index        = df_poly.iloc[ii].HXB2_index;
        nucleotide        = df_poly.iloc[ii].nucleotide;
        TF                = df_poly.iloc[ii].TF;
        epitope           = df_poly.iloc[ii].epitope;
        # get selection coefficient
        nuc_index  = NUC.index(nucleotide)+polymorphic_index*5;
        TF_index   = NUC.index(TF)+polymorphic_index*5;
        sc_MPL     = sc[nuc_index]-sc[TF_index];
        sc_mpl_old = sc_old[nuc_index]-sc_old[TF_index];
        pc_MPL     = 'nan';
        df_i       = df_poly.iloc[ii];
        if sc_MPL != 0:
            for i in range(len(polysite)):
                if polymorphic_index in polysite[i]:
                    pc_MPL = sc[i+L*5]
        f.write('%d,%d,%s,%s,' % (polymorphic_index, alignment, HXB2_index, nucleotide))
        f.write('%s,%s,%f,%f,%s' % (TF, epitope, sc_mpl_old, sc_MPL,pc_MPL))
        f.write(',%s' % (','.join([str(df_i[c]) for c in cols])))
        f.write('\n')
    f.close()

    if len(polysite) != 0:
        df = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        index_cols = ['polymorphic_index', 'alignment']
        cols = [i for i in list(df) if i not in index_cols]

        polyseq  = read_file('polyseq/polysequence-'+tag+'.dat')
        xp = get_xp(seq,polysite,polyseq)


        g = open('%s/poly/%s.csv'%(HIV_DIR,tag),'w')
        g.write('polymorphic_index')
        g.write(',%s' % (','.join(cols)))
        for t in range(len(times)):
            g.write(',xp_at_%s'%times[t])
        g.write('\n')

        for i in range(len(polysite)):
            for j in range(len(polysite[i])):
                df_poly = df[(df.polymorphic_index == polysite[i][j]) & (df.nucleotide != df.TF)]
                for n in range(len(df_poly)):
                    g.write('%d' %polysite[i][j])
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
    poly_sites = read_file('polysite/'+tag+'-polysite.dat')
    TF_poly    = read_file('polyseq/polysequence-'+tag+'.dat')

    g = open('%s/sij.sh'%(MPL_DIR), "a")

    for i in range(len(all_variants)):
        variant   = all_variants[i]
        poly_site = int(variant.split('_')[0])
        mut_alle  = variant.split('_')[-1];
        mut_index = NUC.index(mut_alle);
        g.write('./mpl -d ../../data/HIV -i input/sequence/%s/%s.dat -o output/%s/sc_%s.dat '%(tag,variant,tag,variant))
        g.write('-g 1e4 -N 1e3 -m input/Zanini-extended.dat -p input/polysite/%s-polysite.dat '%tag)
        g.write('-ps input/polyseq/polysequence-%s.dat\n'%tag)

        f = open('%s/input/sequence/%s/%s.dat'%(HIV_DIR,tag,all_variants[i]), "w")
        for j in range(len(seq)):
            seq_modi  = seq[j];
            poly_states = [];
            for ii in range(L):
                if ii == poly_site and seq_modi[ii+2] == mut_index:
                    poly_states.append(str(TF_seq[poly_site]));
                else:
                    poly_states.append(str(int(seq_modi[ii+2])));
            f.write('%d\t1\t%s\n'%(seq_modi[0],' '.join(poly_states)))
        f.close()

    for n in range(len(poly_sites)):
        variant   = 'epi'+str(int(n))
        g.write('./mpl -d ../../data/HIV -i input/sequence/%s/%s.dat -o output/%s/sc_%s.dat '%(tag,variant,tag,variant))
        g.write('-g 1e4 -N 1e3 -m input/Zanini-extended.dat -p input/polysite/%s-polysite.dat '%tag)
        g.write('-ps input/polyseq/polysequence-%s.dat\n'%tag)

        f = open('%s/input/sequence/%s/%s.dat'%(HIV_DIR,tag,variant), "w")
        for j in range(len(seq)):
            seq_modi  = seq[j];
            poly_states = [];
            for ii in range(L):
                if ii in poly_sites[n]:
                    poly_states.append(str(TF_seq[ii]));
                else:
                    poly_states.append(str(int(seq_modi[ii+2])));
            f.write('%d\t1\t%s\n'%(seq_modi[0],' '.join(poly_states)))
        f.close()

    g.close()


def cal_sij(tag):

    df_poly = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    all_variants,TF_seq  = get_all_variants(seq,df_poly)
    polysite = read_file('polysite/'+tag+'-polysite.dat')

    fsij    = open('%s/sij/%s-sij.csv'%(HIV_DIR,tag),'w')
    fsij.write('mask_polymorphic_index,mask_nucleotide,target_polymorphic_index,target_nucleotide,effect,distance\n')

    poly_alle = 'polypart'

    for i in range(len(all_variants)):
        #get site and allele for mask variant
        variant = all_variants[i];
        sc      = np.loadtxt('%s/output/%s/sc_%s.dat'%(HIV_DIR,tag,variant));
        poly_site_i = int(variant.split('_')[0])
        mut_alle_i  = variant.split('_')[-1];

        i_poly     = df_poly[df_poly['polymorphic_index'] == poly_site_i];
        i_align    = i_poly.iloc[0].alignment #get aligment for this site

        # individual site part
        for j in range(len(all_variants)):
            poly_site_j = int(all_variants[j].split('_')[0])
            mut_alle_j  = all_variants[j].split('_')[-1];

            j_poly    = df_poly[df_poly['polymorphic_index'] == poly_site_j];
            j_align   = j_poly.iloc[0].alignment;
            dis       = abs(int(j_align)-int(i_align));#distance between mask variant and target variant, > 0

            if poly_site_i != poly_site_j:
                jj_poly    = j_poly[j_poly['nucleotide'] == mut_alle_j];
                if len(jj_poly)>0:
                    j_TF       = jj_poly.iloc[0].TF;
                    nuc_index  = NUC.index(mut_alle_j) + poly_site_j*5
                    TF_index   = NUC.index(j_TF) + poly_site_j*5
                    sc_j_0     = jj_poly.iloc[0].sc_MPL;
                    sc_j_1     = sc[nuc_index]-sc[TF_index];
                    sij        = sc_j_0 - sc_j_1;
                else:
                    print('wrong with %s,site is %d,allele is %s'%(all_variants[j],poly_site_j,mut_alle_j));
            else:
                sij = 0 #let sij = 0 for the same sites

            fsij.write('%s,%s,%s,%s,%s,%s\n'%(poly_site_i,mut_alle_i,poly_site_j,mut_alle_j,sij,dis));

        #escape part
        dis       = 'nan'
        for j in range(len(polysite)):
            variant_j = 'epi'+str(int(j))
            p_poly    = df_poly[df_poly['polymorphic_index'] == polysite[j][0]];
            pp_poly   = p_poly[p_poly['nucleotide'] != p_poly['TF']];
            pc_j_0    = pp_poly.iloc[0].pc_MPL;
            pc_j_1    = sc[j-len(polysite)];
            sij       = pc_j_0 - pc_j_1;
            fsij.write('%s,%s,%s,%s,%s,%s\n'%(poly_site_i,mut_alle_i,variant_j,poly_alle,sij,dis));

    for i in range(len(polysite)):
        #get site and allele for mask variant
        variant_i = 'epi'+str(int(i))
        sc        = np.loadtxt('%s/output/%s/sc_epi%d.dat'%(HIV_DIR,tag,i));
        dis       = 'nan'
        # individual site part
        for j in range(len(all_variants)):
            poly_site_j = int(all_variants[j].split('_')[0])
            mut_alle_j  = all_variants[j].split('_')[-1];

            j_poly    = df_poly[df_poly['polymorphic_index'] == poly_site_j];
            j_align   = j_poly.iloc[0].alignment;

            if poly_site_j not in polysite[i]:
                jj_poly    = j_poly[j_poly['nucleotide'] == mut_alle_j];
                if len(jj_poly)>0:
                    j_TF       = jj_poly.iloc[0].TF;
                    nuc_index  = NUC.index(mut_alle_j) + poly_site_j*5
                    TF_index   = NUC.index(j_TF) + poly_site_j*5
                    sc_j_0     = jj_poly.iloc[0].sc_MPL;
                    sc_j_1     = sc[nuc_index]-sc[TF_index];
                    sij        = sc_j_0 - sc_j_1;
                else:
                    print('wrong with %s,site is %d,allele is %s'%(all_variants[j],poly_site_j,mut_alle_j));
            else:
                sij = 0 #let sij = 0 for the sites within the epitope

            fsij.write('%s,%s,%s,%s,%s,%s\n'%(variant_i,poly_alle,poly_site_j,mut_alle_j,sij,dis));

        #escape part
        for j in range(len(polysite)):
            variant_j = 'epi'+str(int(j))
            p_poly    = df_poly[df_poly['polymorphic_index'] == polysite[j][0]];
            pp_poly   = p_poly[p_poly['nucleotide'] != p_poly['TF']];
            if j != i :
                pc_j_0  = pp_poly.iloc[0].pc_MPL;
                pc_j_1  = sc[j-len(polysite)];
                sij     = pc_j_0 - pc_j_1;
            else:
                sij = 0
            fsij.write('%s,%s,%s,%s,%s,%s\n'%(variant_i,poly_alle,variant_j,poly_alle,sij,dis));
    fsij.close();

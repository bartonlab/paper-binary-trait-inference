#############  PACKAGES  #############
import numpy as np

import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from scipy import integrate
import scipy.interpolate as sp_interpolate

import seaborn as sns

from colorsys import hls_to_rgb
from dataclasses import dataclass

import mplot as mp
import re

############# PARAMETERS #############

# GLOBAL VARIABLES -- simulation

NUC = ['-', 'A', 'C', 'G', 'T']

# GitHub

HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'

# Standard color scheme

BKCOLOR  = '#252525'
C_BEN    = '#EB4025'
C_BEN_LT = '#F08F78'
C_NEU    = '#969696'
C_NEU_LT = '#E8E8E8'
C_DEL    = '#3E8DCF'
C_DEL_LT = '#78B4E7'
C_group  = ['#32b166','#e5a11c', '#a48cf4','#ff69b4','#ff8c00','#36ada4','#f0e54b']

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.5)
DOUBLE_COLUMN   = cm2inch(17.4)

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4

# paper style
FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZELINE     = 0.6

FIGPROPS = {
    'transparent' : True,
    #'bbox_inches' : 'tight'
}

DEF_ERRORPROPS = {
    'mew'        : AXWIDTH,
    'markersize' : SMALLSIZEDOT/2,
    'fmt'        : 'o',
    'elinewidth' : SIZELINE/2,
    'capthick'   : 0,
    'capsize'    : 0
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False
}

############# PLOTTING  FUNCTIONS #############

def read_file(name):
    trait = []
    p = open(SIM_DIR+'/jobs/'+name,'r')
    maxlen = 0
    maxnum = 0
    for line in p:
        temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
        data = [float(item) for item in temp]
        trait.append(data)
        #print(data)
        if len(data) > maxlen:
            maxlen = len(data)
        maxnum += 1
    arr = np.zeros((maxnum, maxlen))*np.nan
    for cnt in np.arange(maxnum):
        arr[cnt, 0:len(trait[cnt])] = np.array(trait[cnt])
    p.close()
    for i in range(len(trait)):
        for j in range(len(trait[i])):
            trait[i][j] = int(trait[i][j])
    return trait


def plot_example_mpl(**pdata):
    """
    Example evolutionary trajectory for a 50-site system and inferred selection coefficients
    and trait coefficients, together with aggregate properties for different levels of sampling..
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    dg       = pdata['dg']
    pop_size = pdata['N']
    xfile    = pdata['xfile']

    n_ben    = pdata['n_ben']
    n_neu    = pdata['n_neu']
    n_del    = pdata['n_del']
    n_tra    = pdata['n_tra']
    s_ben    = pdata['s_ben']
    s_neu    = pdata['s_neu']
    s_del    = pdata['s_del']
    s_tra    = pdata['s_tra']

    r_seed = pdata['r_seed']
    np.random.seed(r_seed)

    show_fig = pdata['show_fig']
    # load and process data files

    data  = np.loadtxt('%s/jobs/sequences/example-%s.dat' % (SIM_DIR, xfile))

    x_index = xfile.split('_')[0]
    trait_site = read_file('traitsite/traitsite-%s.dat'%(x_index))

    #allele frequency x
    x     = []
    for i in range(0, n_gen, dg):
        idx    = data.T[0]==i
        t_data = data[idx].T[2:].T
        t_num  = data[idx].T[1].T
        t_freq = np.einsum('i,ij->j', t_num, t_data) / float(np.sum(t_num))
        x.append(t_freq)
    x = np.array(x).T # get allele frequency (binary case)

    #trait frequency y
    y    = []
    for t in range(0, n_gen, dg):
        idx    = data.T[0]==t
        t_num  = data[idx].T[1].T
        t_fre     = []
        for i in range(len(trait_site)):
            t_data_i  = t_num*0
            for j in range(len(trait_site[i])):
                site = trait_site[i][j]
                t_data_i += data[idx].T[site+2]
            for k in range(len(t_data_i)):
                if t_data_i[k] != 0:
                    t_data_i[k] = 1
            t_freq_i = np.einsum('i,i', t_num, t_data_i) / float(np.sum(t_num))
            t_fre.append(t_freq_i)
        y.append(t_fre)
    y = np.array(y).T # get trait frequency

    s_true  = [s_ben for i in range(n_ben)] + [0 for i in range(n_neu)]
    s_true += [s_del for i in range(n_del)] + [s_tra for i in range(n_tra)]
    
    # binary case
    s_inf      = np.loadtxt('%s/jobs/output/sc-%s.dat' %(SIM_DIR,x_index))

    # multiple case
    # s_origin   = np.loadtxt('%s/jobs/output_multiple/sc-%s.dat' %(SIM_DIR,x_index))
    # s_inf   = np.zeros(seq_length+len(trait_site))
    # for i in range(seq_length):
    #     s_inf[i] = s_origin[2*i+1] - s_origin[2*i]
    # s_inf[-2:] = s_origin[-2:]

    cov     = np.loadtxt('%s/jobs/covariance/covariance-%s.dat' %(SIM_DIR,x_index))
    ds      = np.linalg.inv(cov) / pop_size

    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tra1 = dict(left=0.10, right=0.45, bottom=0.55, top=0.95)
    box_tra2 = dict(left=0.55, right=0.90, bottom=0.55, top=0.95)
    box_coe1 = dict(left=0.10, right=0.45, bottom=0.10, top=0.42)
    box_coe2 = dict(left=0.55, right=0.90, bottom=0.10, top=0.42)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_coe1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe1)
    gs_coe2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe2)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_coe1 = plt.subplot(gs_coe1[0, 0])
    ax_coe2 = plt.subplot(gs_coe2[0, 0])

    dx = -0.04
    dy = -0.02

    ## a -- all allele trajectories together

    pprops = { 'xticks':      [0, 200, 400, 600, 800,1000],
               'yticks':      [0,1],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Generation',
               'ylabel':      'Allele\nfrequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open' }

    xdat = [range(0, n_gen, dg) for k in range(n_ben)]
    ydat = [k for k in x[:n_ben]]
    mp.line(ax=ax_tra1, x=xdat, y=ydat, colors=[C_BEN_LT for k in range(len(x))], **pprops)

    xdat = [range(0, n_gen, dg) for k in range(n_del)]
    ydat = [k for k in x[n_ben+n_neu:]]
    mp.line(ax=ax_tra1, x=xdat, y=ydat, colors=[C_DEL_LT for k in range(len(x))], **pprops)

    xdat = [range(0, n_gen, dg) for k in range(n_neu)]
    ydat = [k for k in x[n_ben:n_ben+n_neu]]
    pprops['plotprops']['alpha'] = 0.4
    mp.plot(type='line',ax=ax_tra1, x=xdat, y=ydat, colors = [C_NEU for k in range(len(x))], **pprops)

    ax_tra1.text(box_tra1['left']+dx, box_tra1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- all trait trajectories together

    pprops = { 'xticks':      [0, 200, 400, 600, 800,1000],
               'yticks':      [0, 1],
               'yminorticks': [0.25,0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Generation',
               'ylabel':      'Frequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-' , 'alpha': 1},
               'axoffset':    0.1,
               'theme':       'open' }

    traj_legend_x  = 220
    traj_legend_y  = [0.65, 0.40]
    traj_legend_t  = ['Trait\nfrequency','Individual \nallele frequency']

    for k in range(len(traj_legend_y)):
        x1 = traj_legend_x-50
        x2 = traj_legend_x-10
        y1 = traj_legend_y[0] + (0.5-k) * 0.03
        y2 = traj_legend_y[1] + (0.5-k) * 0.03
        pprops['plotprops']['alpha'] = 1
        mp.line(ax=ax_tra2, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[k]], **pprops)
        pprops['plotprops']['alpha'] = 0.4
        mp.line(ax=ax_tra2, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[k]], **pprops)
        ax_tra2.text(traj_legend_x, traj_legend_y[k], traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    xdat = [range(0, n_gen, dg)]
    for i in range(len(trait_site)):
        for j in range(len(trait_site[i])):
            site = trait_site[i][j]
            ydat = [k for k in x[site:site+1]]
            pprops['plotprops']['alpha'] = 0.6
            mp.line(ax=ax_tra2, x=xdat, y=ydat, colors=[C_group[i]], **pprops)
        if i > 0:
            ydat = [k for k in y[i:i+1]]
            pprops['plotprops']['alpha'] = 1
            mp.line(ax=ax_tra2, x=xdat, y=ydat, colors = [C_group[i]], **pprops)

    ydat = [k for k in y[0:1]]
    pprops['plotprops']['alpha'] = 1
    mp.plot(type='line',ax=ax_tra2, x=xdat, y=ydat, colors = [C_group[0]], **pprops)

    ax_tra2.text(box_tra2['left']+dx, box_tra1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c -- individual beneficial/neutral/deleterious selection coefficients

    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

    pprops = { 'xlim':        [ -0.3,    4],
               'ylim':        [-0.05, 0.04],
               'yticks':      [-0.04, 0, 0.04],
               'yminorticks': [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
               'yticklabels': [-4, 0, 4],
               'xticks':      [],
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }

    n_coe1    = [n_ben, n_neu, n_del]
    c_coe1    = [C_BEN, C_NEU, C_DEL]
    c_coe1_lt = [C_BEN_LT, C_NEU_LT, C_DEL_LT]
    offset    = [0, n_ben, n_ben+n_neu]

    for k in range(len(n_coe1)):
        mp.line(ax=ax_coe1, x=[[k-0.35, k+0.35]], y=[[s_true[offset[k]], s_true[offset[k]]]], \
        colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
        plotprops = DEF_ERRORPROPS.copy()
        plotprops['alpha'] = 1
        for i in range(n_coe1[k]):
            xdat = [k + np.random.normal(0, 0.08)]
            ydat = [s_inf[offset[k]+i]]
            yerr = np.sqrt(ds[offset[k]+i][offset[k]+i])
            if i==n_coe1[k]-1 and k==len(n_coe1)-1:
                mp.plot(type='error', ax=ax_coe1, x=[xdat], y=[ydat], yerr=[yerr], \
                edgecolor=[c_coe1[k]], facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)
            else:
                mp.error(ax=ax_coe1, x=[xdat], y=[ydat], yerr=[yerr], edgecolor=[c_coe1[k]], \
                facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)

    coef_legend_x  =  2.8
    coef_legend_d  = -0.15
    coef_legend_dy = -0.011
    coef_legend_y  = [0.02, 0.02 + coef_legend_dy, 0.02 + (2*coef_legend_dy)]
    coef_legend_t  = ['Beneficial', 'Neutral', 'Deleterious']
    for k in range(len(coef_legend_y)):
        mp.error(ax=ax_coe1, x=[[coef_legend_x+coef_legend_d]], y=[[coef_legend_y[k]]], \
                 edgecolor=[c_coe1[k]], facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)
        ax_coe1.text(coef_legend_x, coef_legend_y[k], coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  0.02 + 3.5 * coef_legend_dy
    mp.line(ax=ax_coe1, x=[[coef_legend_x-0.21, coef_legend_x-0.09]], y=[[yy, yy]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_coe1.text(coef_legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe1.text(box_coe1['left']+dx, box_coe1['top']+0.04, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## d -- trait coefficients

    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

    pprops = { 'xlim':        [-0.3,   6],
               'ylim':        [   0,0.16],
               'yticks':      [   0,0.08,0.16],
               'yminorticks': [0.02,0.04,0.06,0.10,0.12,0.14],
               'yticklabels': [   0,   8,  16],
               'xticks':      [],
               'ylabel':      'Inferred trait\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }

    n_coe2    = []
    c_coe2    = []
    c_coe2_lt = []
    offset    = n_ben+n_neu+n_del
    for i in range(n_tra):
        c_coe2.append(C_group[i])
        c_coe2_lt.append(C_group[i])

    mp.line(ax=ax_coe2, x=[[0.15, 0.85]], y=[[s_true[-1], s_true[-1]]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['alpha'] = 1
    for i in range(n_tra):
        xdat = [np.random.normal(0.5, 0.18)]
        ydat = [s_inf[offset+i]]
        yerr = np.sqrt(ds[offset+i][offset+i])
        if i==n_tra-1:
            mp.plot(type='error', ax=ax_coe2, x=[xdat], y=[ydat], yerr=[yerr], \
            edgecolor=[c_coe2[i]], facecolor=[c_coe2_lt[i]], plotprops=plotprops, **pprops)
        else:
            mp.error(ax=ax_coe2, x=[xdat], y=[ydat], yerr=[yerr], \
            edgecolor=[c_coe2[i]], facecolor=[c_coe2_lt[i]], plotprops=plotprops, **pprops)

    coef_legend_d  = -0.15 * (6.3 / 4.3)
    coef_legend_x  =  1.8
    coef_legend_dy = -0.11 * (0.16 / 0.9)  #-0.02
    coef_legend_y  = []
    coef_legend_t  = []
    for i in range(len(trait_site)):
        coef_legend_y.append(0.135 + i * coef_legend_dy)
        coef_legend_t.append('Trait %d'%(i+1))
    for k in range(len(coef_legend_y)):
        mp.error(ax=ax_coe2, x=[[coef_legend_x+coef_legend_d+0.03]], y=[[coef_legend_y[k]]], \
                 edgecolor=[c_coe2[k]], facecolor=[c_coe2_lt[k]], plotprops=plotprops, **pprops)
        ax_coe2.text(coef_legend_x, coef_legend_y[k], coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  0.135 +(0.5+n_tra) * coef_legend_dy
    mp.line(ax=ax_coe2, x=[[coef_legend_x-0.24, coef_legend_x-0.09]], y=[[yy, yy]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_coe2.text(coef_legend_x, yy, 'True\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe2.text(box_coe2['left']+dx, box_coe2['top']+0.04, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    if show_fig:
        plt.savefig('%s/fig1.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        print('figure 1 done.')
    else:
        plt.savefig('%s/sim/%s.pdf' % (FIG_DIR,x_index), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.close()

def plot_histogram_sim(**pdata):
    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data

    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_tra = pdata['n_tra']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_tra = pdata['s_tra']

    # PLOT FIGURE
    ## set up figure grid

    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_se   = dict(left=0.10, right=0.62, bottom=0.65, top=0.95)
    box_tra   = dict(left=0.69, right=0.92, bottom=0.65, top=0.95)
    box_aur1 = dict(left=0.10, right=0.32, bottom=0.07, top=0.50)
    box_aur2 = dict(left=0.40, right=0.62, bottom=0.07, top=0.50)
    box_erro = dict(left=0.70, right=0.92, bottom=0.07, top=0.50)

    gs_se   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
    gs_tra   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra)
    gs_aur1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_aur1)
    gs_aur2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_aur2)
    gs_erro = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_erro)

    ax_se  = plt.subplot(gs_se[0, 0])
    ax_tra  = plt.subplot(gs_tra[0, 0])
    ax_aur1 = plt.subplot(gs_aur1[0, 0])
    ax_aur2 = plt.subplot(gs_aur2[0, 0])
    ax_erro = plt.subplot(gs_erro[0, 0])

    dx = -0.04
    dy =  0.03

    ### plot histogram

    df_all   = pd.read_csv('%s/mpl_collected_nsdt.csv' % SIM_DIR, memory_map=True)
    df       = df_all[(df_all.ns==1000) & (df_all.delta_t==1)]

    ben_cols = ['sc_%d' % i for i in range(n_ben)]
    neu_cols = ['sc_%d' % i for i in range(n_ben, n_ben+n_neu)]
    del_cols = ['sc_%d' % i for i in range(n_ben+n_neu, n_ben+n_neu+n_del)]
    tra_cols = ['tc_%d' % i for i in range(n_tra)]

    colors     = [C_BEN, C_NEU, C_DEL]
    tags       = ['beneficial', 'neutral', 'deleterious','trait']
    cols       = [ben_cols, neu_cols, del_cols, tra_cols]
    s_true_loc = [s_ben, s_neu, s_del,s_tra]

    ## a -- selection part

    dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.04,  0.04],
               'xticks'      : [ -0.04, -0.03, -0.02, -0.01,    0.,  0.01,  0.02,  0.03,  0.04],
               'xticklabels' : [    -4,    -3,    -2,    -1,     0,      1,    2,     3,     4],
               'ylim'        : [0., 0.10],
               'yticks'      : [0., 0.05, 0.10],
               'xlabel'      : 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.04, 0.04, 0.001),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    for i in range(len(tags)-1):
        x = [np.array(df[cols[i]]).flatten()]
        tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
        ax_se.text(s_true_loc[i], 0.106, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
        ax_se.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_se, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_se, x=x, colors=[colors[i]], **pprops)

    ax_se.text(  box_se['left']+dx,  box_se['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- trait part
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [   0, 0.12],
               'xticks'      : [   0, 0.04, 0.08, 0.10, 0.12],
               'xticklabels' : [   0,    4,    8,   10,   12],
               'ylim'        : [0., 0.15],
               'yticks'      : [0., 0.05, 0.10, 0.15],
               'xlabel'      : 'Inferred trait coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(0, 0.12, 0.003),
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    x = [np.array(df[cols[3]]).flatten()]
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_tra.text(s_true_loc[3], 0.159, r'$s_{%s}$' % (tags[3]), color=C_group[0], **tprops)
    ax_tra.axvline(x=s_true_loc[3], **dashlineprops)
    mp.plot(type='hist', ax=ax_tra, x=x, colors=[C_group[0]], **pprops)

    ax_tra.text( box_tra['left']+dx,  box_tra['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c,d  -- AUCs for inferring beneficial/deleterious mutations and error for trait part

    df   = pd.read_csv('%s/mpl_collected_extended.csv' % SIM_DIR, memory_map=True)

    ns_vals = [10, 20, 30, 40, 50, 80, 100]
    dt_vals = [1, 5, 10, 20, 50]

    AUC_matrix_ben = np.zeros((len(dt_vals), len(ns_vals)))
    AUC_matrix_del = np.zeros((len(dt_vals), len(ns_vals)))
    err_matrix_tra = np.zeros((len(dt_vals), len(ns_vals)))

    for i in range(len(dt_vals)):
        for j in range(len(ns_vals)):
            df_AUC = df[(df.delta_t==dt_vals[i]) & (df.ns==ns_vals[j])]
            AUC_matrix_ben[i, j] = np.mean(df_AUC.AUROC_ben)
            AUC_matrix_del[i, j] = np.mean(df_AUC.AUROC_del)
            err_matrix_tra[i, j] = np.mean(df_AUC.error_tra)

    pprops = { 'xlim'        : [0, len(dt_vals)],
               'xticks'      : np.arange(len(dt_vals))+0.5,
               'xticklabels' : [int(k) for k in dt_vals],
               'ylim'        : [0, len(ns_vals)],
               'yticks'      : np.arange(len(ns_vals))+0.5,
               'yticklabels' : [int(k) for k in ns_vals],
               'xlabel'      : 'Time between samples, '+r'$\Delta t$' + ' (generations)',
               'ylabel'      : 'Number of sequences per time point, '+r'$n_s$',
               'theme'       : 'boxed' }
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)

    ax_aur1.pcolor(AUC_matrix_ben.T, vmin=0.75, vmax=1.0, cmap='GnBu', alpha=0.75)
    for i in range(len(AUC_matrix_ben)):
        for j in range(len(AUC_matrix_ben[0])):
            tc = 'k'
            if AUC_matrix_ben[i,j]>0.96: tc = 'white'
            ax_aur1.text(i+0.5, j+0.5, '%.2f' % (AUC_matrix_ben[i,j]), color=tc, **tprops)
    mp.plot(type='scatter', ax=ax_aur1, x=[[-1]], y=[[-1]], colors=[BKCOLOR], **pprops)

    ax_aur2.pcolor(AUC_matrix_del.T, vmin=0.75, vmax=1.0, cmap='GnBu', alpha=0.75)
    for i in range(len(AUC_matrix_del)):
        for j in range(len(AUC_matrix_del[0])):
            tc = 'k'
            if AUC_matrix_del[i,j]>0.96: tc = 'white'
            ax_aur2.text(i+0.5, j+0.5, '%.2f' % (AUC_matrix_del[i,j]), color=tc, **tprops)
    mp.plot(type='scatter', ax=ax_aur2, x=[[-1]], y=[[-1]], colors=[BKCOLOR], **pprops)

    ax_erro.pcolor(err_matrix_tra.T, vmin=0.2, vmax=0.8, cmap='GnBu', alpha=0.75)
    for i in range(len(err_matrix_tra)):
        for j in range(len(err_matrix_tra[0])):
            tc = 'k'
            ax_erro.text(i+0.5, j+0.5, '%.2f' % (err_matrix_tra[i,j]), color=tc, **tprops)
    mp.plot(type='scatter', ax=ax_erro, x=[[-1]], y=[[-1]], colors=[BKCOLOR], **pprops)

    ## outside text labels

    tprops = dict(color=BKCOLOR, ha='center', va='center', family=FONTFAMILY, size=SIZELABEL,
                  clip_on=False, transform=fig.transFigure)

    ax_aur1.text((box_aur1['right']-box_aur1['left'])/2+box_aur1['left'], \
    box_aur1['top']+dy, 'Mean AUROC (beneficial)',  **tprops)
    ax_aur2.text((box_aur2['right']-box_aur2['left'])/2+box_aur2['left'], \
    box_aur2['top']+dy, 'Mean AUROC (deleterious)', **tprops)
    ax_erro.text((box_erro['right']-box_erro['left'])/2+box_erro['left'], \
    box_erro['top']+dy, 'NRMSE (trait)', **tprops)

    ax_aur1.text(box_aur1['left']+dx, box_aur1['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_aur2.text(box_aur2['left']+dx, box_aur2['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_erro.text(box_erro['left']+dx, box_erro['top']+dy, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig2.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 2 done.')

def plot_sc_escape(**pdata):
    """
    a. Histogram of selection coefficients for trait sites with group term
    b. Histogram of selection coefficients for trait sites without group term
    """

    # unpack passed data
    tags   = pdata['tags']

    # get all selection coefficients for trait sites
    sc_all         = []
    sc_all_notrait = []

    for tag in tags:
        df          = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        df_escape   = df[(df['epitope'].notna()) & (df['escape'] == True)]

        for i in range(len(df_escape)):
            sc_all.append(df_escape.iloc[i].sc_MPL)
            sc_all_notrait.append(df_escape.iloc[i].sc_old)

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 1.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_old = dict(left=0.15, right=0.92, bottom=0.61, top=0.95)
    box_new = dict(left=0.15, right=0.92, bottom=0.14, top=0.48)
    gs_old  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_old)
    gs_new  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_new)
    ax_old  = plt.subplot(gs_old[0, 0])
    ax_new  = plt.subplot(gs_new[0, 0])

    dx = -0.10
    dy =  0.02

    ### plot histogram - with and without escape term

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.05,  0.10],
               'xticks'      : [ -0.05,     0,  0.05, 0.1],
               'xticklabels' : [ ],
               'ylim'        : [0., 0.30],
               'yticks'      : [0., 0.10, 0.20, 0.30],
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.05, 0.10, 0.002),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }
    
    # without escape term
    x    = [np.array(sc_all_notrait)]
    mp.plot(type='hist', ax=ax_old, x=x, colors=[C_group[1]], **pprops)
    
    # with escape term
    pprops['xticklabels'] = [    -5,     0,    5,   10]
    pprops['xlabel'] = 'Inferred selection coefficient for escape sites, ' + r'$\hat{s}$ ' +'(%)'
    x    = [np.array(sc_all)]
    mp.plot(type='hist', ax=ax_new, x=x, colors=[C_group[0]], **pprops)

    # (with escape term)
    ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ax_old.text(0.02, 0.21, 'Without escape trait', **DEF_LABELPROPS)
    ax_new.text(0.02, 0.21, 'With escape trait', **DEF_LABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/sc_escape.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


def plot_tc_rec(**pdata):
    """
    a. Histogram of trait coefficients with recombination
    b. Histogram of trait coefficients without recombination
    """

    # unpack passed data
    tags   = pdata['tags']

    # get all trait coefficients
    tc_all     = []
    tc_all_noR = []

    for tag in tags:
        df_tc     = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        df_tc_noR = pd.read_csv('%s/noR/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        unique_epitopes = df_tc['epitope'].unique()
        for epitope in unique_epitopes:
            tc_all.append(df_tc[df_tc.epitope == epitope].iloc[0].tc_MPL)
            tc_all_noR.append(df_tc_noR[df_tc_noR.epitope == epitope].iloc[0].tc_MPL)

    # PLOT FIGURE
    ## set up figure grid
    w     = SINGLE_COLUMN
    goldh = w / 1.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_old = dict(left=0.15, right=0.92, bottom=0.61, top=0.95)
    box_new = dict(left=0.15, right=0.92, bottom=0.14, top=0.48)
    gs_old  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_old)
    gs_new  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_new)
    ax_old  = plt.subplot(gs_old[0, 0])
    ax_new  = plt.subplot(gs_new[0, 0])

    dx = -0.10
    dy =  0.02

    ### plot histogram - with and without escape term

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.05,  0.30],
               'xticks'      : [ -0.05,     0,  0.05, 0.1 , 0.15,  0.2, 0.25, 0.30],
               'xticklabels' : [ ],
               'ylim'        : [0., 0.20],
               'yticks'      : [0., 0.10, 0.20],
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.05, 0.30, 0.005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    # without escape term
    x    = [np.array(tc_all_noR)]
    mp.plot(type='hist', ax=ax_old, x=x, colors=[C_group[1]], **pprops)

    # with escape term
    pprops['xticklabels'] = [    -5,     0,    5,   10,   15,   20,   25,   30]
    pprops['xlabel'] = 'Inferred escape coefficient, ' + r'$\hat{s}$ ' +'(%)'
    x    = [np.array(tc_all)]
    mp.plot(type='hist', ax=ax_new, x=x, colors=[C_group[0]], **pprops)

    ax_old.text(0.113, 0.14, 'Without recombination', **DEF_LABELPROPS)
    ax_new.text(0.113, 0.14, 'With recombination', **DEF_LABELPROPS)

    ax_old.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_new.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/tc_rec.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


def plot_histogram_fraction_HIV(**pdata):
    """
    a. Histogram of selection coefficients and escape coefficients
    b. Fraction for escape part
    """

    # unpack passed data
    tags   = pdata['tags']
    ppts   = pdata['ppts']

    # get all escape coefficients
    tc = []

    for tag in tags:
        df_tc = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        tc_old = df_tc.iloc[0].tc_MPL
        for i in range(len(df_tc)):
            if df_tc.iloc[i].tc_MPL != tc_old:
                tc.append(df_tc.iloc[i].tc_MPL)
            tc_old = df_tc.iloc[i].tc_MPL

    # get all escape contribution
    fractions = []
    common_times = []
    for i in range(len(ppts)):
        ppt = ppts[i]
        tag_3 = ppt + '-' + str(3)
        tag_5 = ppt + '-' + str(5)

        t_3, f_3,n_3 = getFitness(tag_3)
        t_5, f_5,n_5 = getFitness(tag_5)

        common_t = np.intersect1d(t_3, t_5)
        fraction = np.zeros(len(common_t))
        for i in range(len(common_t)):
            index_3 = list(t_3).index(common_t[i])
            index_5 = list(t_5).index(common_t[i])
            FitAll    = (f_3[0,index_3]*n_3[index_3]+f_5[0,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
            FitEscape = (f_3[1,index_3]*n_3[index_3]+f_5[1,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
            if FitAll != 1:
                fraction[i] = FitEscape/(FitAll-1)
            if FitAll<1 and FitEscape<0:
                fraction[i] = 0
            if fraction[i] < 0:
                fraction[i] = 0
            if fraction[i] > 1:
                fraction[i] = 1

        common_times.append(common_t)
        fractions.append(fraction)

    max_times = [max(common_times[i]) for i in range(len(common_times))]
    max_time = int(max(max_times))

    whole_time = np.linspace(0,max_time,max_time+1)
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(0,0), bounds_error=False)

    IntFractions = np.zeros((len(common_times),len(whole_time)))
    IntNumber    = np.zeros((len(common_times),len(whole_time)))
    for i in range(len(common_times)):
        IntFractions[i] = interpolation(common_times[i], fractions[i])(whole_time)
        IntNumber[i] = interpolation(common_times[i], np.ones(len(common_times[i])))(whole_time)

    AveFraction = np.zeros(len(whole_time))
    for t in range(len(whole_time)):
        fraction_t = np.sum(IntFractions[:,t])
        number_t   = np.sum(IntNumber[:,t])
        AveFraction[t] = fraction_t/number_t

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 1.1
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tc   = dict(left=0.15, right=0.92, bottom=0.59, top=0.96)
    box_frac = dict(left=0.15, right=0.92, bottom=0.12, top=0.42)

    gs_frac  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_frac)
    gs_tc    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)

    ax_tc    = plt.subplot(gs_tc[0, 0])
    ax_frac  = plt.subplot(gs_frac[0, 0])

    dx = -0.10
    dy =  0.02

    ### plot histogram

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.05,  0.30],
               'xticks'      : [ -0.05,     0,  0.05, 0.1 , 0.15,  0.2, 0.25, 0.30],
               'xticklabels' : [    -5,     0,    5,   10,   15,    20,   25,   30],
               'ylim'        : [0., 0.30],
               'yticks'      : [0., 0.15, 0.30],
               'xlabel'      : 'Inferred escape coefficient, ' + r'$\hat{s}$ ' +'(%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.05, 0.30, 0.01),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    x    = [np.array(tc)]
    mp.plot(type='hist', ax=ax_tc, x=x, colors=[C_group[0]], **pprops)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ### plot fraction
    pprops = { #'xticks':      [ 0,  100, 200, 300,  400, 500, 600, 700],
               'xticks':      [ 0,  np.log(11),np.log(51),np.log(101), np.log(201),np.log(401),np.log(701)],
               'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
               'ylim'  :      [0., 1.01],
               'yticks':      [0., 1],
               'yminorticks': [0.25, 0.5, 0.75],
               'yticklabels': [0, '$\geq 1$'],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Fitness gain fraction \ndue to escape trait',
               'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.5},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    # individual fraction
    for i in range(len(common_times)):
        max_t_i = int(max(common_times[i]))
        time_i  = np.linspace(0,max_t_i,max_t_i+1)
        time    = np.log(time_i+1)
        mp.line(ax=ax_frac, x=[time], y=[IntFractions[i][:max_t_i+1]], colors=[C_group[0]], **pprops)

    # 0
    pprops['plotprops']['ls'] = '--'
    mp.line(ax=ax_frac,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)

    # average curve
    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*1.8
    pprops['plotprops']['ls'] = '-'
    time = np.log(whole_time+1)
    mp.plot(type='line', ax=ax_frac, x=[time], y=[AveFraction],colors=[C_NEU], **pprops)

    # legend
    traj_legend_x =  np.log(300)
    traj_legend_y = [1.0, 0.87]
    traj_legend_t = ['Individual', 'Average']

    x1 = traj_legend_x-0.4
    x2 = traj_legend_x-0.1
    y1 = traj_legend_y[0] + 0.015
    y2 = traj_legend_y[1] + 0.015
    pprops['plotprops']['alpha'] = 0.5
    pprops['plotprops']['lw'] = SIZELINE
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[0]], **pprops)
    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*1.8
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y2, y2]], colors=[C_NEU], **pprops)
    ax_frac.text(traj_legend_x, traj_legend_y[0], traj_legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_frac.text(traj_legend_x, traj_legend_y[1], traj_legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # label
    ax_frac.text(box_frac['left']+dx, box_frac['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig3.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 3 done.')


def plot_figure_4(**pdata):
    """
    epitope escape mutation frequencies, inferred escape coefficients.
    """
    # unpack data

    patient       = pdata['patient']
    region        = pdata['region']
    epitope       = pdata['epitope']
    traj_ticks    = pdata['traj_ticks']
    tag           = patient+'-'+region

    # process stored data

    df_poly = pd.read_csv('%s/group/escape_group-%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_tag   = []
    traj_poly = []
    poly_info = {}
    var_tc    = []
    var_traj  = []
    var_color = []
    for i in range(len(epitope)):
        df_esc  = df_poly[(df_poly.epitope==epitope[i])]
        df_row  = df_esc.iloc[0]
        epi_nuc = ''.join(epitope[i])
        p_tag = epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc))
        p_c   = df_esc.iloc[0].tc_MPL
        poly_info[p_tag] = p_c
        traj_poly.append([df_row['xp_at_%d' % t] for t in times])
        traj_indi = []
        for df_iter, df_entry in df_esc.iterrows():
            traj_indi.append([df_entry['f_at_%d' % t] for t in times])
        var_traj.append(traj_indi)

    var_c = sns.husl_palette(len(traj_poly))

    # Sort by the value of inferred escape coefficients
    poly_info_1 = sorted(poly_info.items(),key=lambda x: x[1],reverse=True)
    poly_key = list(poly_info.keys())
    for i in poly_info_1:
        var_tag.append(i[0])
        var_tc.append(i[1])
        index_i = poly_key.index(i[0])
        var_color.append(var_c[index_i])

    # PLOT FIGURE

    ## set up figure grid

    w       = SINGLE_COLUMN
    goldh   =  w /2
    fig     = plt.figure(figsize=(w, goldh),dpi=1000)

    box_traj = dict(left=0.17, right=0.95, bottom=0.57, top=0.90)
    box_coef = dict(left=0.17, right=0.44, bottom=0.15, top=0.38)
    box_lend = dict(left=0.55, right=0.95, bottom=0.15, top=0.38)

    gs_traj = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_coef = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coef)
    gs_lend = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lend)
    ax_traj = plt.subplot(gs_traj[0, 0])
    ax_coef = plt.subplot(gs_coef[0, 0])
    ax_lend = plt.subplot(gs_lend[0, 0])

    dx = -0.15
    dy =  0.06

    ## a -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
               'yticks':      [0, 1.0],
               'ylim':        [0, 0.33],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Variant frequency',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    for i in range(len(var_tag)):
        pprops['plotprops']['alpha'] = 1
        xdat = [times]
        ydat = [traj_poly[i]]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[var_c[i]], **pprops)
        pprops['plotprops']['alpha'] = 0.4
        for j in range(len(var_traj[i])):
            ydat = [var_traj[i][j]]
            mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[var_c[i]], **pprops)

    ax_traj.text(box_traj['left']+dx, box_traj['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- escape coefficients inferred by MPL

    hist_props = dict(lw=SIZELINE/2, width=0.4, align='center',alpha=0.8,
                      orientation='vertical',edgecolor=[BKCOLOR for i in range(len(var_tag))])

    bar_x  = [i+0.5 for i in range(len(var_tag))]
    var_epi = []
    for i in range(len(var_tag)):
        var_epi.append('r\'$'+var_tag[i]+'$\'')
    pprops = { 'colors':      [var_color],
               'xlim':        [0, len(var_tag)],
               'xticks'  :    bar_x,
               'ylim':        [    0,  0.16],
               'yticks':      [    0, 0.08, 0.16],
               'yminorticks': [ 0.04, 0.12],
               'yticklabels': [    0,    8,    16],
               'xticklabels': [eval(k) for k in var_epi],
               'ylabel':      'Inferred escape\ncoefficient, '+r'$\hat{s}$'+' (%)',
               'combine'     : True,
               'plotprops'   : hist_props,
               'axoffset'    : 0.1,
               'theme':       'open'}

    mp.plot(type='bar', ax=ax_coef,x=[bar_x], y=[var_tc], **pprops)

    ax_coef.text(box_coef['left']+dx, box_coef['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)


    # c -- legend

    pprops = { 'xlim': [0,  5],
               'ylim': [0,  5],
               'xticks': [],
               'yticks': [],
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'theme': 'open',
               'hide' : ['top', 'bottom', 'left', 'right'] }

    traj_legend_x  =  2
    traj_legend_y  = [1, 4]
    traj_legend_t  = ['Escape\nfrequency','Individual \nallele frequency']

    for k in range(len(traj_legend_y)):
        ax_lend.text(traj_legend_x, traj_legend_y[k], traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    for k in range(len(var_c)):
        x1 = traj_legend_x-0.7
        x2 = traj_legend_x-0.1
        y1 = traj_legend_y[0] + (1-k) * 0.4
        y2 = traj_legend_y[1] + (1-k) * 0.4
        pprops['plotprops']['alpha'] = 1
        mp.plot(type='line',ax=ax_lend, x=[[x1, x2]], y=[[y1, y1]], colors=[var_color[k]], **pprops)
        pprops['plotprops']['alpha'] = 0.4
        mp.plot(type='line',ax=ax_lend, x=[[x1, x2]], y=[[y2, y2]], colors=[var_color[k]], **pprops)

    #ax_lend.text(-1.5, 2.5, '3\' half-genome \nfor CH470', ha='left', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig4.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 4 done.')


def plot_figure_5(**pdata):
    """
    epitope escape mutation frequencies, inferred escape coefficientsd.
    """
    # unpack data

    patient       = pdata['patient']
    region        = pdata['region']
    epitope       = pdata['epitope']
    traj_ticks    = pdata['traj_ticks']
    tag           = patient+'-'+region

    # process stored data

    df_poly = pd.read_csv('%s/group/escape_group-%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_tag   = []
    var_snew  = []
    var_sold  = []
    var_tc    = []
    var_traj  = []
    traj_poly = []
    for i in range(len(epitope)):
        df_esc  = df_poly[(df_poly.epitope==epitope[i])]
        df_row  = df_esc.iloc[0]
        var_tc.append(df_esc.iloc[0].tc_MPL)
        traj_poly.append([df_row['xp_at_%d' % t] for t in times])
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide!='-':
                var_traj.append([df_entry['f_at_%d' % t] for t in times])
                var_tag.append(df_entry.HXB2_index+df_entry.nucleotide)
                var_sold.append(df_entry.sc_old)
                var_snew.append(df_entry.sc_MPL)

    #PLOT FIGURE

    # set up figure grid

    w       = SINGLE_COLUMN
    goldh   =  w /1.4
    fig     = plt.figure(figsize=(w, goldh),dpi=1000)
    sspace  = 0.02

    box_traj = dict(left=0.17, right=0.85, bottom=0.62, top=0.92)
    box_coef = dict(left=0.17, right=0.85, bottom=0.17, top=0.42)

    gs_traj = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_coef = gridspec.GridSpec(1, 2, width_ratios=[5,6], wspace = sspace,     **box_coef)

    ax_traj = plt.subplot(gs_traj[0, 0])
    ax_coef = [plt.subplot(gs_coef[0, i]) for i in range(2)]

    dx = -0.10
    dy =  0.04

    var_c = sns.husl_palette(len(var_traj)+1)

    ## a -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
               'ylim':        [0, 0.6],
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Variant frequency\nin EV11 epitope\n',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    xdat = [times]
    ydat = [traj_poly[0]]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[var_c[-1]], **pprops)
    pprops['plotprops']['alpha'] = 0.4
    for j in range(len(var_traj)):
        ydat = [var_traj[j]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[var_c[j]], **pprops)

    ax_traj.text(box_traj['left']+dx, box_traj['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b,c -- selection coefficients inferred by MPL (new and old)

    hist_props = dict(lw=SIZELINE/2, width=0.6, align='center',alpha=0.8,
                      orientation='vertical',edgecolor=[BKCOLOR for i in range(len(var_tag))])

    bar_x  = [i+0.5 for i in range(len(var_tag))]
    pprops = { 'colors':      [var_c],
               'xlim':        [0, len(var_tag)],
               'xticks'  :    bar_x,
               'xticklabels': var_tag,
               'ylim':        [ -0.05,  0.20],
               'yticks':      [ -0.05,     0,  0.05, 0.10, 0.15, 0.20],
               'yticklabels': [    -5,     0,     5,   10,   15,   20],
               'ylabel':      'Inferred \ncoefficient, '+r'$\hat{s}$'+' (%)',
               'combine'     : True,
               'plotprops'   : hist_props,
               'axoffset'    : 0.1,
               'theme':       'open'}

    mp.plot(type='bar', ax=ax_coef[0],x=[bar_x], y=[var_sold], **pprops)
    plt.setp(ax_coef[0].xaxis.get_majorticklabels(), rotation=90)

    transFigureInv = fig.transFigure.inverted()
    labelprops     = dict(color=BKCOLOR, ha='center', va='top', family=FONTFAMILY, size=SIZELABEL,
                          clip_on=False, transform=fig.transFigure)

    ax_coef[0].text((box_coef['right']+3*box_coef['left']-sspace)/4, box_coef['top'], 'Without escape trait', **labelprops)
    ax_coef[0].text(box_coef['left']+dx, box_coef['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    bar_x  = [i+0.5 for i in range(len(var_tag)+1)]
    var_tag.append(r'$EV11$')
    var_snew.append(var_tc[0])
    pprops = { 'colors':      [var_c],
               'xlim':        [0, len(var_tag)+1],
               'xticks'  :    bar_x,
               'xticklabels': var_tag,
               'ylim':        [ -0.03,  0.18],
               'yticks':      [],
               'ylabel':      None,
               'combine'     : True,
               'plotprops'   : hist_props,
               'axoffset'    : 0.1,
               'theme':       'open'}

    mp.plot(type='bar', ax=ax_coef[1],x=[bar_x], y=[var_snew], hide = ['left','right'], **pprops)
    plt.setp(ax_coef[1].xaxis.get_majorticklabels(), rotation=90)

    ax_coef[1].text((box_coef['right']*3+box_coef['left']-4*sspace)/4, box_coef['top'], 'With escape trait', **labelprops)
    ax_coef[1].text((box_coef['right']+box_coef['left']+sspace)/2+dx/2, box_coef['top']+dy, 'c'.lower(), \
    transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # add background

    cBG = '#F5F5F5'
    ddx = 0.01
    ddy = 0.01
    rec = matplotlib.patches.Rectangle(xy=((box_coef['right']+box_coef['left'])/2-2*ddx, box_coef['bottom']-(17.0*ddy)),
                                            width=box_coef['right']-(box_coef['right']+box_coef['left'])/2 - (2.2*ddx),
                                            height=box_coef['top']-box_coef['bottom']+(18.2*ddy), \
                                            transform=fig.transFigure, ec=None, fc=cBG, clip_on=False, zorder=-100)
    rec = ax_coef[1].add_patch(rec)

    # SAVE FIGURE
    plt.savefig('%s/fig5.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 5 done.')


def plot_figure_6(**pdata):

    # unpack data

    patient    = pdata['patient']
    region     = pdata['region']
    epitope    = pdata['epitope']
    traj_ticks = pdata['traj_ticks']
    variants   = pdata['variants']
    high_var   = pdata['high_var']
    seq_length = pdata['seq_length']
    note_var   = pdata['note_var']
    tag        = patient+'-'+region

    # process stored data

    df_poly = pd.read_csv('%s/analysis/%s-analyze.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
    df_sij  = pd.read_csv('%s/sij/%s-sij.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_sold  = []
    var_snew  = []
    epi_sold  = []
    epi_snew  = []
    var_traj  = []
    hig_traj  = []
    var_tag   = []
    var_note  = []
    ds_matrix = []
    new_var   = []
    new_note  = []

    for i in range(seq_length):
        df_esc  = df_poly[(df_poly.polymorphic_index==i)& (df_poly.sc_MPL != 0)& (df_poly.nucleotide != '-') ]
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide!='-' and df_entry.epitope not in epitope:
                var_sold.append(df_entry.sc_old)
                var_snew.append(df_entry.sc_MPL)

    for i in range(len(epitope)):
        df_esc  = df_poly[(df_poly.epitope==epitope[i]) & (df_poly.sc_MPL != 0)]
        epi_nuc = ''.join(epitope[i])
        var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))
        sold = []
        snew = []
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide!='-' and df_entry.tc_MPL > 0:
                sold.append(df_entry.sc_old)
                snew.append(df_entry.sc_MPL)
        epi_sold.append(sold)
        epi_snew.append(snew)

    for i in range(len(variants)):
        index_t = int(variants[i].split('_')[0])
        neucl_t = variants[i].split('_')[-1]

        df_esc  = df_poly[(df_poly.polymorphic_index==index_t)& (df_poly.nucleotide == neucl_t)]
        df_ds   = df_sij[(df_sij.target_polymorphic_index == str(index_t)) & (df_sij.target_nucleotide==neucl_t)]

        HXB2_in = df_esc.iloc[0].HXB2_index
        new_var.append(str(HXB2_in)+neucl_t)

        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.polymorphic_index not in high_var:
                var_traj.append([df_entry['f_at_%d' % t] for t in times])
            else:
                hig_traj.append([df_entry['f_at_%d' % t] for t in times])

        ds_vec = []
        for ii in range(len(variants)):
            index = int(variants[ii].split('_')[0])
            neucl = variants[ii].split('_')[-1]
            if ii == i:
                ds_vec.append(0)
            else:
                ds_vec.append(df_ds[(df_ds.mask_polymorphic_index==str(index)) & (df_ds.mask_nucleotide==neucl)].iloc[0].effect)
        for ii in range(len(epitope)):
            index_m = 'epi'+ str(ii)
            ds_vec.append(df_ds[(df_ds.mask_polymorphic_index==index_m) & (df_ds.mask_nucleotide=='polypart')].iloc[0].effect)
        ds_matrix.append(ds_vec)

    for i in range(len(note_var)):
        index   = int(note_var[i].split('_')[0])
        neucleo = note_var[i].split('_')[-1]

        df_esc  = df_poly[(df_poly.polymorphic_index==index)& (df_poly.nucleotide == neucleo)]

        HXB2_i  = df_esc.iloc[0].HXB2_index
        new_note.append(str(HXB2_i)+neucleo)

        for df_iter, df_entry in df_esc.iterrows():
            var_note.append([df_entry.sc_MPL,df_entry.sc_old])

    # PLOT FIGURE

    ## set up figure grid

    w       = DOUBLE_COLUMN
    goldh   = w/1.8
    fig     = plt.figure(figsize=(w, goldh),dpi=1000)

    box_ss   = dict(left=0.10, right=0.45, bottom=0.38, top=0.92)
    box_traj = dict(left=0.10, right=0.45, bottom=0.09, top=0.26)
    box_sij  = dict(left=0.53, right=0.95, bottom=0.12, top=0.75)

    gs_ss   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_ss)
    gs_traj = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_sij  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sij)

    ax_ss   =  plt.subplot(gs_ss[0, 0])
    ax_traj =  plt.subplot(gs_traj[0, 0])
    ax_sij  =  plt.subplot(gs_sij[0, 0])

    dx = -0.05
    dy =  0.04

    var_c = sns.husl_palette(len(var_tag)+len(hig_traj))
    traj_c = [var_c[-2],var_c[-1]]

    ## a -- inferred selection coefficients with VS. without poluygenic term

    s_min = -0.02
    s_max =  0.05
    scatterprops = dict(lw=0, s=SMALLSIZEDOT*0.6, marker='o', alpha=0.6)
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : ':', 'alpha' : 0.6}

    pprops = { 'xlim':         [ s_min, s_max],
               'ylim':         [ s_min, s_max],
               'xticks':       [-0.02,-0.01,    0,  0.01,  0.02,  0.03,  0.04,  0.05],
               'yticks':       [-0.02,-0.01,    0,  0.01,  0.02,  0.03,  0.04,  0.05],
               'xticklabels':  [-2,-1,0,1,2,3,4,5],
               'yticklabels':  [-2,-1,0,1,2,3,4,5],
               'xlabel':       'Inferred selection coefficients with escape traits',
               'ylabel':       'Inferred selection coefficients without escape traits',
               'theme':        'boxed'}

    mp.line(ax=ax_ss, x=[[s_min, s_max]], y=[[s_min,s_max]], colors=[C_NEU],plotprops=lineprops, **pprops)

    for i in range(len(var_snew)):
        mp.plot(type='scatter', ax=ax_ss, x=[[var_snew[i]]], y=[[var_sold[i]]], colors=[C_NEU],plotprops=scatterprops, **pprops)

    scatterprops['alpha'] = 0.8
    for i in range(len(epi_snew)):
        for j in range(len(epi_snew[i])):
            mp.plot(type='scatter', ax=ax_ss, x=[[epi_snew[i][j]]], y=[[epi_sold[i][j]]], colors=[var_c[i]],plotprops=scatterprops, **pprops)

    traj_legend_x  = 0.01
    traj_legend_dy = -0.003
    y0             = -0.002
    dx0            = 0.002
    traj_legend_y  = [y0, y0 + traj_legend_dy ,y0 + traj_legend_dy*2]
    scatterprops['s'] = SMALLSIZEDOT*0.8
    for k in range(len(var_tag)):
        traj_legend_k = 'Escape variants in epitope '+var_tag[k]
        mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[traj_legend_y[k]]], colors=[var_c[k]],plotprops=scatterprops, **pprops)
        ax_ss.text(traj_legend_x, traj_legend_y[k], traj_legend_k, ha='left', va='center', **DEF_LABELPROPS)

    mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[y0 + traj_legend_dy*3]], colors=[C_NEU],plotprops=scatterprops, **pprops)
    ax_ss.text(traj_legend_x, y0 + traj_legend_dy*3, 'Not escape variants', ha='left', va='center', **DEF_LABELPROPS)

    ddx = -0.0001
    ddy =  0.0025
    for i in range(len(var_note)):
        ax_ss.text(var_note[i][0]+ddx, var_note[i][1]+ddy, new_note[i], ha='center', va='center', **DEF_LABELPROPS)

    ax_ss.text(box_ss['left']+dx, box_ss['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Variant frequency',
               'plotprops':   {'lw': SIZELINE, 'ls': '-'},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    pprops['plotprops']['alpha'] = 0.4
    xdat = [times for k in range(len(var_traj))]
    ydat = [k for k in var_traj[:]]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[C_NEU for k in range(len(var_traj))], **pprops)

    pprops['plotprops']['alpha'] = 0.8
    for j in range(len(hig_traj)):
        xdat = [times]
        ydat = [hig_traj[j]]
        mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[traj_c[j]], **pprops)

    traj_legend_x  =  350
    traj_legend_dy = -0.2
    traj_legend_y  = [0.5, 0.5 + traj_legend_dy,0.5 + 2*traj_legend_dy]
    traj_legend_t  = ['Other variants','974A', '3951C']
    for k in range(len(traj_legend_y)):
        yy = traj_legend_y[k]
        if k != 0:
            mp.line(ax=ax_traj, x=[[traj_legend_x-35, traj_legend_x-10]], y=[[yy,yy]], colors=[traj_c[k-1]], **pprops)
        else:
            mp.line(ax=ax_traj, x=[[traj_legend_x-35, traj_legend_x-10]], y=[[yy,yy]], colors=[C_NEU], **pprops)
        ax_traj.text(traj_legend_x, traj_legend_y[k], traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    ax_traj.text(box_traj['left']+dx, box_traj['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c -- effects of linkage on selection for sites

    pprops = { 'colors': [BKCOLOR],
               'xlim': [0, len(ds_matrix[0]) + 1],
               'ylim': [1, len(ds_matrix)+2],
               'xticks': [],
               'yticks': [],
               'plotprops': dict(lw=0, s=0.1*SMALLSIZEDOT, marker='o', clip_on=False),
               'ylabel': '',
               'theme': 'open',
               'hide' : ['top', 'bottom', 'left', 'right'] }

    mp.plot(type='scatter', ax=ax_sij, x=[[5]], y=[[5]], **pprops)

    site_rec_props = dict(height=1, width=1, ec=None, lw=AXWIDTH/2, clip_on=False)
    rec_patches    = []

    for i in range(len(ds_matrix)):
        for j in range(len(ds_matrix[i])):
            temp_ds = ds_matrix[i][j]
            t       = temp_ds / 0.02
            if t>0:
                c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
            else:
                c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
            rec = matplotlib.patches.Rectangle(xy=(j + (j>=len(ds_matrix)) * 0.3, len(ds_matrix) - i), fc=c, **site_rec_props)
            rec_patches.append(rec)

    individal_rec_props = dict(height=len(ds_matrix), width=len(ds_matrix),
                             ec=BKCOLOR,fc='none', lw=AXWIDTH/2, clip_on=False)
    epitope_rec_props   = dict(height=len(ds_matrix), width=len(ds_matrix[0]) - len(ds_matrix),
                             ec=BKCOLOR, fc='none', lw=AXWIDTH/2, clip_on=False)
    rec_patches.append(matplotlib.patches.Rectangle(xy=(0, 1), **individal_rec_props))
    rec_patches.append(matplotlib.patches.Rectangle(xy=(len(ds_matrix)+0.3, 1), **epitope_rec_props))

    for i in range(9):
        t = i/4 - 1
        if t>0:
            c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
        else:
            c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
        rec = matplotlib.patches.Rectangle(xy=(i+2, 12.5), fc=c, **site_rec_props)
        rec_patches.append(rec)

    for patch in rec_patches:
        ax_sij.add_artist(patch)

    txtprops = dict(ha='right', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    for i in range(len(ds_matrix)):
        ax_sij.text(-0.2, len(ds_matrix) - i + 0.5, '%s' % new_var[i], **txtprops)

    txtprops = dict(ha='center', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    for i in range(len(ds_matrix)):
        ax_sij.text(i + 0.5, 0.8, '%s' % new_var[i], **txtprops)
    for i in range(len(var_tag)):
        ax_sij.text(len(ds_matrix) + i + 0.8, 0.8, '%s' % var_tag[i], **txtprops)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_sij.text(6.5, -1, 'Variant $i$', **txtprops)
    ax_sij.text(5.5,11.5, 'Most influential variants', **txtprops)
    ax_sij.text(11.8,11.5, 'Epitopes', **txtprops)

    y_label = 13.8
    ax_sij.text(  2.5, y_label,   2, **txtprops)
    ax_sij.text(  4.5, y_label,   1, **txtprops)
    ax_sij.text(  6.5, y_label,   0, **txtprops)
    ax_sij.text(  8.5, y_label,  -2, **txtprops)
    ax_sij.text( 10.5, y_label,  -2, **txtprops)
    ax_sij.text(  6.5, y_label+1, 'Effect of variant $i$ on inferred\nselection coefficient '+' $\hat{s}_j$'+\
                          ', $\Delta \hat{s}_{ij}$ (%)', ha='center', va='center', **DEF_LABELPROPS)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    ax_sij.text(-2.2, 7, 'Target variant $j$', **txtprops)

    ax_sij.text(box_sij['left']+dx, box_ss['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig6.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 6 done.')


def plotReversionFaction(**pdata):
    """
    Fraction for reversion part
    """

    # unpack passed data
    ppts   = pdata['ppts']

    # get all reversion contribution
    fractions = []
    common_times = []
    for i in range(len(ppts)):
        ppt = ppts[i]
        tag_3 = ppt + '-' + str(3)
        tag_5 = ppt + '-' + str(5)

        t_3, f_3,n_3 = getFitnessReversion(tag_3)
        t_5, f_5,n_5 = getFitnessReversion(tag_5)

        common_t = np.intersect1d(t_3, t_5)
        fraction = np.zeros(len(common_t))
        for i in range(len(common_t)):
            index_3 = list(t_3).index(common_t[i])
            index_5 = list(t_5).index(common_t[i])
            FitAll  = (f_3[0,index_3]*n_3[index_3]+f_5[0,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
            FitRev  = (f_3[1,index_3]*n_3[index_3]+f_5[1,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
            if FitAll != 1:
                fraction[i] = FitRev/(FitAll-1)
            if FitAll < 1 and FitRev<0:
                fraction[i] = 0

            if fraction[i] < 0:
                fraction[i] = 0
            if fraction[i] > 1:
                fraction[i] = 1

        common_times.append(common_t)
        fractions.append(fraction)

    max_times = [max(common_times[i]) for i in range(len(common_times))]
    max_time = int(max(max_times))

    whole_time = np.linspace(0,max_time,max_time+1)
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(0,0), bounds_error=False)

    IntFractions = np.zeros((len(common_times),len(whole_time)))
    IntNumber    = np.zeros((len(common_times),len(whole_time)))
    for i in range(len(common_times)):
        IntFractions[i] = interpolation(common_times[i], fractions[i])(whole_time)
        IntNumber[i] = interpolation(common_times[i], np.ones(len(common_times[i])))(whole_time)

    AveFraction = np.zeros(len(whole_time))
    for t in range(len(whole_time)):
        fraction_t = np.sum(IntFractions[:,t])
        number_t   = np.sum(IntNumber[:,t])
        AveFraction[t] = fraction_t/number_t

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 2.2
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_frac = dict(left=0.20, right=0.92, bottom=0.22, top=0.95)
    gs_frac  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_frac)
    ax_frac  = plt.subplot(gs_frac[0, 0])

    dx = -0.10
    dy =  0.02

    ### plot fraction
    pprops = { #'xticks':      [ 0,  100, 200, 300,  400, 500, 600, 700],
               'xticks':      [ 0,  np.log(10),np.log(50),np.log(100), np.log(200),np.log(400),np.log(700)],
               'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
               'ylim'  :      [0., 1.01],
               'yticks':      [0., 1],
               'yminorticks': [0.25, 0.5, 0.75],
               'yticklabels': [0, '$\geq 1$'],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Fitness gain fraction \ndue to reversion',
               'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.5},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    for i in range(len(common_times)):
        max_t_i = int(max(common_times[i]))
        time_i  = np.linspace(0,max_t_i,max_t_i+1)
        time    = np.log(time_i+1)
        mp.line(ax=ax_frac, x=[time], y=[IntFractions[i][:max_t_i+1]], colors=[C_group[0]], **pprops)

    pprops['plotprops']['ls'] = '--'
    mp.line(ax=ax_frac,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)

    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*1.8
    pprops['plotprops']['ls'] = '-'
    time = np.log(whole_time+1)
    mp.plot(type='line', ax=ax_frac, x=[time], y=[AveFraction],colors=[C_NEU], **pprops)

    # SAVE FIGURE
    plt.savefig('%s/reversion.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure reversion done.')

@dataclass
class Result:
    variants: 0
    seq_length: 0
    special_sites: []
    uniq_t:[]
    escape_group:[]
    escape_TF:[]

def AnalyzeData(tag):
    df_info = pd.read_csv('data/HIV/analysis/%s-analyze.csv' %tag, comment='#', memory_map=True)
    seq     = np.loadtxt('data/HIV/input/sequence/%s-poly-seq2state.dat'%tag)

    """get raw time points"""
    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])
    uniq_t = np.unique(times)

    """get variants number and sequence length"""
    df_poly  = df_info[df_info['nucleotide']!=df_info['TF']]
    variants = len(df_poly)
    seq_length = df_info.iloc[-1].polymorphic_index + 1

    """get special sites and escape sites"""
    # get all epitopes for one tag
    df_rows = df_info[df_info['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()

    min_n = 2 # the least escape sites a trait group should have (more than min_n)
    special_sites = [] # special site considered as time-varying site but not escape site
    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    escape_TF     = [] # corresponding wild type nucleotide
    for epi in unique_epitopes:
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()

        if len(unique_sites) <= min_n:
            special_sites.append(unique_sites)
        else:
            escape_group.append(list(unique_sites))
            tf_values = []
            for site in unique_sites:
                tf_value = df_e[df_e['polymorphic_index'] == site]['TF'].values
                tf_values.append(NUC.index(tf_value[0]))
            escape_TF.append(tf_values)

    special_sites = [item for sublist in special_sites for item in sublist]

    return Result(variants, seq_length,special_sites,uniq_t,escape_group,escape_TF)

def getSC(tag):

    df_info = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    xlength = df_info.iloc[-1].polymorphic_index+1
    SCMatrix = np.zeros((xlength,len(NUC)))
    for i in range(xlength):
        i_info = df_info[df_info.polymorphic_index == i]
        for j in range(len(i_info)):
            nucleotide = i_info.iloc[j].nucleotide
            n_index = NUC.index(nucleotide)
            SCMatrix[i,n_index] = i_info.iloc[j].sc_MPL
    return SCMatrix

def getEC(tag):

    try:
        df_info = pd.read_csv('%s/group/escape_group-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

        epitopes = []
        for i in range(len(df_info)):
            epitopes.append(df_info.iloc[i].epitope)
        uniq_e = np.unique(epitopes)

        elength = len(uniq_e)
        ECMatrix = np.zeros(elength)
        for i in range(elength):
            e_info = df_info[df_info.epitope == uniq_e[i]]
            ECMatrix[i] = e_info.iloc[0].tc_MPL

    except FileNotFoundError:
        ECMatrix = []

    return ECMatrix

def getFitness(tag):
    seq      = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    result   = AnalyzeData(tag)
    traitsite = result.escape_group
    polyseq  = result.escape_TF
    s_sites  = result.special_sites
    uniq_t   = result.uniq_t

    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])

    SCMatrix = getSC(tag)
    ECMatrix = getEC(tag)
    FitAll   = np.zeros((2,len(uniq_t)))# 0:total fitness, 1: fitness increase from escape (group+special sites)
    CountAll = np.zeros(len(uniq_t))    # population number at different times
    for t in range(len(uniq_t)):
        tid = times==uniq_t[t]

        cout_t = seq[tid][:,1]
        seq_t  = seq[tid][:,2:]
        counts = np.sum(cout_t)

        fit_a = 0 # total fitness
        fit_e = 0 # fitness for escape part

        for i in range(len(seq_t)):
            # fitness contribution from individual selection
            fitness = 1
            for j in range(len(seq_t[0])):
                fitness += SCMatrix[j,int(seq_t[i][j])]

            fitness_e = 0
            # fitness contribution from escape group
            for ii in range(len(traitsite)):
                poly_value = sum([abs(seq_t[i][int(traitsite[ii][jj])]-polyseq[ii][jj]) for jj in range(len(traitsite[ii]))])
                if poly_value > 0:
                    fitness   += ECMatrix[ii]
                    fitness_e += ECMatrix[ii]

            # fitness contribution from special sites
            for jj in s_sites:
                fitness_e += SCMatrix[jj,int(seq_t[i][jj])]

            fit_a += fitness*cout_t[i]
            fit_e += fitness_e*cout_t[i]

        FitAll[0,t]   = fit_a/counts
        FitAll[1,t]   = fit_e/counts
        CountAll[t] = counts

    return uniq_t,FitAll,CountAll

def getReversionSite(tag,s_sites):

    df_consensus = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)

    ReversionSites = []
    for i in range(len(df_consensus)):
        nucleotide_i = df_consensus.iloc[i].nucleotide
        TF_i         = df_consensus.iloc[i].TF
        consensus_i  = df_consensus.iloc[i].consensus
        if nucleotide_i != TF_i and nucleotide_i == consensus_i:
            site_i = df_consensus.iloc[i].polymorphic_index
            if site_i not in s_sites:
                ReversionSites.append(str(site_i)+str(nucleotide_i))
    ReversionSites = np.unique(ReversionSites)
    
    return ReversionSites

def getFitnessReversion(tag):
    seq      = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    result   = AnalyzeData(tag)
    traitsite = result.escape_group
    polyseq  = result.escape_TF
    s_sites  = result.special_sites
    uniq_t   = result.uniq_t

    ReversionSites = getReversionSite(tag,s_sites)

    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])

    SCMatrix = getSC(tag)
    ECMatrix = getEC(tag)
    FitAll   = np.zeros((2,len(uniq_t)))# 0:total fitness, 1: fitness increase from reversion (individual sites)
    CountAll = np.zeros(len(uniq_t))    # population number at different times
    for t in range(len(uniq_t)):
        tid = times==uniq_t[t]

        cout_t = seq[tid][:,1]
        seq_t  = seq[tid][:,2:]
        counts = np.sum(cout_t)

        fit_a = 0 # total fitness
        fit_r = 0 # fitness due to reversion

        for i in range(len(seq_t)):
            # fitness contribution from individual selection
            fitness = 1
            for j in range(len(seq_t[0])):
                fitness += SCMatrix[j,int(seq_t[i][j])]

            # fitness contribution from escape group
            for ii in range(len(traitsite)):
                poly_value = sum([abs(seq_t[i][int(traitsite[ii][jj])]-polyseq[ii][jj]) for jj in range(len(traitsite[ii]))])
                if poly_value > 0:
                    fitness   += ECMatrix[ii]

            fit_a += fitness*cout_t[i]

            fitness_r = 0
            for k in range(len(ReversionSites)):
                site_j = int(ReversionSites[k][:-1])
                allele_j = NUC.index(ReversionSites[k][-1])
                if seq_t[i][site_j] == allele_j:
                    fitness_r += SCMatrix[site_j,allele_j]
            fit_r += fitness_r*cout_t[i]

        FitAll[0,t]   = fit_a/counts
        FitAll[1,t]   = fit_r/counts
        CountAll[t] = counts

    return uniq_t,FitAll,CountAll

def plot_single_fraction(**pdata):
    """
    a. Histogram of selection coefficients and escape coefficients
    b. Fraction for escape part
    """

    # unpack passed data
    ppt   = pdata['ppt']

    # get fraction due to escape part or reversion part
    tag_3 = ppt + '-' + str(3)
    tag_5 = ppt + '-' + str(5)

    t_3, f_3,n_3 = getFitness(tag_3)
    t_5, f_5,n_5 = getFitness(tag_5)

    common_t = np.intersect1d(t_3, t_5)
    fraction_epi = np.zeros(len(common_t))
    for i in range(len(common_t)):
        index_3 = list(t_3).index(common_t[i])
        index_5 = list(t_5).index(common_t[i])
        FitAll    = (f_3[0,index_3]*n_3[index_3]+f_5[0,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
        FitEscape = (f_3[1,index_3]*n_3[index_3]+f_5[1,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
        if FitAll != 1:
            fraction_epi[i] = FitEscape/(FitAll-1)
        if FitAll<1 and FitEscape<0:
            fraction_epi[i] = 0
        if fraction_epi[i] < 0:
            fraction_epi[i] = 0
        if fraction_epi[i] > 1:
            fraction_epi[i] = 1

    t_3, f_3,n_3 = getFitnessReversion(tag_3)
    t_5, f_5,n_5 = getFitnessReversion(tag_5)
    fraction_rev = np.zeros(len(common_t))
    for i in range(len(common_t)):
        index_3 = list(t_3).index(common_t[i])
        index_5 = list(t_5).index(common_t[i])
        FitAll  = (f_3[0,index_3]*n_3[index_3]+f_5[0,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])
        FitRev  = (f_3[1,index_3]*n_3[index_3]+f_5[1,index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])

        if FitAll != 1:
            fraction_rev[i] = FitRev/(FitAll-1)
        if FitAll < 1 and FitRev<0:
            fraction_rev[i] = 0
        if fraction_rev[i] < 0:
            fraction_rev[i] = 0
        if fraction_rev[i] > 1:
            fraction_rev[i] = 1

    whole_time = np.linspace(0,int(common_t[-1]),int(common_t[-1]+1))
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(0,0), bounds_error=False)

    IntFractions = np.zeros((2,len(whole_time)))
    IntFractions[0] = interpolation(common_t, fraction_epi)(whole_time)
    IntFractions[1] = interpolation(common_t, fraction_rev)(whole_time)

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_frac   = dict(left=0.20, right=0.92, bottom=0.22, top=0.95)
    gs_frac    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_frac)
    ax_frac    = plt.subplot(gs_frac[0, 0])

    ### plot fraction for escape part
    pprops = { #'xticks':      [ 0,  100, 200, 300,  400, 500, 600, 700],
               'xticks':      [ 0,  np.log(11),np.log(51),np.log(101), np.log(201),np.log(401),np.log(701)],
               'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
               'ylim'  :      [0., 1.01],
               'yticks':      [0., 1],
               'yminorticks': [0.25, 0.5, 0.75],
               'yticklabels': [0, '$\geq 1$'],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Fitness gain fraction \ndue to escape (CH-%s)'%ppt[-3:],
               'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.5},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    time = np.log(whole_time+1)

    # due to escape
    mp.line(ax=ax_frac, x=[time], y=[IntFractions[0]], colors=[C_group[0]], **pprops)

    # due to reversion
    mp.line(ax=ax_frac, x=[time], y=[IntFractions[1]], colors=[C_group[1]], **pprops)

    # y=0 and y=1
    pprops['plotprops']['ls'] = '--'
    mp.line(ax=ax_frac,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)
    mp.line(ax=ax_frac,x=[[0,6.5]], y=[[1,1]],colors=[C_NEU], **pprops)

    # sum of due to escape and reversion
    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['ls'] = '-'
    mp.plot(type='line',ax=ax_frac, x=[time], y=[IntFractions[0]+IntFractions[1]], colors=[C_group[2]], **pprops)

    # legend
    traj_legend_x  =  np.log(2.2)
    traj_legend_y  = [0.85,0.70,0.55]
    traj_legend_t  = ['Escape','Reversion','Sum']

    x1 = traj_legend_x-0.6
    x2 = traj_legend_x-0.1
    y1 = traj_legend_y[0] + 0.015
    y2 = traj_legend_y[1] + 0.015
    y3 = traj_legend_y[2] + 0.015

    pprops['plotprops']['alpha'] = 0.5
    # escape
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[0]], **pprops)
    ax_frac.text(traj_legend_x, traj_legend_y[0], traj_legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    # reversion
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[1]], **pprops)
    ax_frac.text(traj_legend_x, traj_legend_y[1], traj_legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y3, y3]], colors=[C_group[2]], **pprops)
    ax_frac.text(traj_legend_x, traj_legend_y[2], traj_legend_t[2], ha='left', va='center', **DEF_LABELPROPS)

    plt.savefig('%s/reversion-CH%s.pdf' % (FIG_DIR,ppt[-3:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


def plot_histogram_sim_rec(**pdata):
    """
    a. Histogram of coefficients without recombination
    b. Histogram of coefficients with recombination
    """

    # unpack passed data

    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_tra = pdata['n_tra']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_tra = pdata['s_tra']

    # PLOT FIGURE
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w / 2.4
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_s_old   = dict(left=0.10, right=0.62, bottom=0.59, top=0.95)
    box_t_old   = dict(left=0.69, right=0.92, bottom=0.59, top=0.95)
    box_s_new   = dict(left=0.10, right=0.62, bottom=0.10, top=0.46)
    box_t_new   = dict(left=0.69, right=0.92, bottom=0.10, top=0.46)

    gs_s_old   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_s_old)
    gs_t_old   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_t_old)
    gs_s_new = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_s_new)
    gs_t_new = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_t_new)

    ax_s_old  = plt.subplot(gs_s_old[0, 0])
    ax_t_old  = plt.subplot(gs_t_old[0, 0])
    ax_s_new = plt.subplot(gs_s_new[0, 0])
    ax_t_new = plt.subplot(gs_t_new[0, 0])

    dx = -0.04
    dy =  0.03

    ### plot histogram

    df_noR = pd.read_csv('%s/mpl_collected_noR.csv' % SIM_DIR, memory_map=True)

    df_all = pd.read_csv('%s/mpl_collected_nsdt.csv' % SIM_DIR, memory_map=True)
    df     = df_all[(df_all.ns==1000) & (df_all.delta_t==1)]

    ben_cols = ['sc_%d' % i for i in range(n_ben)]
    neu_cols = ['sc_%d' % i for i in range(n_ben, n_ben+n_neu)]
    del_cols = ['sc_%d' % i for i in range(n_ben+n_neu, n_ben+n_neu+n_del)]
    tra_cols = ['tc_%d' % i for i in range(n_tra)]

    colors     = [C_BEN, C_NEU, C_DEL]
    tags       = ['beneficial', 'neutral', 'deleterious','trait']
    cols       = [ben_cols, neu_cols, del_cols, tra_cols]
    s_true_loc = [s_ben, s_neu, s_del,s_tra]

    ## a,c -- selection part

    dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.04,  0.04],
               'xticks'      : [ -0.04, -0.03, -0.02, -0.01,    0.,  0.01,  0.02,  0.03,  0.04],
               'xticklabels' : [    -4,    -3,    -2,    -1,     0,      1,    2,     3,     4],
               'ylim'        : [0., 0.10],
               'yticks'      : [0., 0.05, 0.10],
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.04, 0.04, 0.001),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    for i in range(len(tags)-1):
        x = [np.array(df_noR[cols[i]]).flatten()]
        tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
        ax_s_old.text(s_true_loc[i], 0.106, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
        ax_s_old.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_s_old, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_s_old, x=x, colors=[colors[i]], **pprops)

    pprops['xlabel'] = 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)'
    for i in range(len(tags)-1):
        x = [np.array(df[cols[i]]).flatten()]
        tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
        ax_s_new.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_s_new, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_s_new, x=x, colors=[colors[i]], **pprops)

    ax_s_old.text(-0.035, 0.08, 'Without\nrecombination', **DEF_LABELPROPS)
    ax_s_new.text(-0.035, 0.08, 'With\nrecombination', **DEF_LABELPROPS)

    ax_s_old.text(box_s_old['left']+dx, box_s_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_s_new.text(box_s_new['left']+dx, box_s_new['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ## b,d -- trait part
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [   0, 0.12],
               'xticks'      : [   0, 0.04, 0.08, 0.10, 0.12],
               'xticklabels' : [   0,    4,    8,   10,   12],
               'ylim'        : [0., 0.15],
               'yticks'      : [0., 0.05, 0.10, 0.15],
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(0, 0.12, 0.003),
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    x = [np.array(df_noR[cols[3]]).flatten()]
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_t_old.text(s_true_loc[3], 0.159, r'$s_{%s}$' % (tags[3]), color=C_group[0], **tprops)
    ax_t_old.axvline(x=s_true_loc[3], **dashlineprops)
    mp.plot(type='hist', ax=ax_t_old, x=x, colors=[C_group[0]], **pprops)

    pprops['xlabel'] = 'Inferred trait coefficient, ' + r'$\hat{s}$' + ' (%)'
    x = [np.array(df[cols[3]]).flatten()]
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_t_new.axvline(x=s_true_loc[3], **dashlineprops)
    mp.plot(type='hist', ax=ax_t_new, x=x, colors=[C_group[0]], **pprops)

    ax_t_old.text(0.01, 0.12, 'Without\nrecombination', **DEF_LABELPROPS)
    ax_t_new.text(0.01, 0.12, 'With\nrecombination', **DEF_LABELPROPS)

    ax_t_old.text(box_t_old['left']+dx,  box_t_old['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_t_new.text(box_t_new['left']+dx,  box_t_new['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # SAVE FIGURE
    plt.savefig('%s/sim_his_rec.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

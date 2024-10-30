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
from typing import List

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
    result = []
    with open('%s/%s'%(SIM_DIR,name), 'r') as file:
        for line in file:
            line_data = []
            for item in line.split():
                line_data.append(int(item))
            result.append(line_data)
    return result

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
    trait_site = read_file('/jobs/traitsite/traitsite-%s.dat'%(x_index))

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

    box_tra1 = dict(left=0.10, right=0.48, bottom=0.55, top=0.98)
    box_tra2 = dict(left=0.58, right=0.96, bottom=0.55, top=0.98)
    box_coe1 = dict(left=0.10, right=0.48, bottom=0.05, top=0.42)
    box_coe2 = dict(left=0.58, right=0.70, bottom=0.05, top=0.42)
    box_fit  = dict(left=0.70, right=0.99, bottom=0.05, top=0.42)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_coe1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe1)
    gs_coe2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe2)
    gs_fit  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_fit)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_coe1 = plt.subplot(gs_coe1[0, 0])
    ax_coe2 = plt.subplot(gs_coe2[0, 0])
    ax_fit  = plt.subplot(gs_fit[0, 0])

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

    pprops = { 'xlim':        [-0.3,   3],
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

    # ture value
    mp.line(ax=ax_coe2, x=[[0.00, 0.70]], y=[[s_true[-1], s_true[-1]]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)

    # inferred value
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['alpha'] = 1
    for i in range(n_tra):
        xdat = [np.random.normal(0.35, 0.18)]
        ydat = [s_inf[offset+i]]
        yerr = np.sqrt(ds[offset+i][offset+i])
        mp.error(ax=ax_coe2, x=[xdat], y=[ydat], yerr=[yerr], \
        edgecolor=[c_coe2[i]], facecolor=[c_coe2_lt[i]], plotprops=plotprops, **pprops)

    # legend
    coef_legend_d  = -0.15 * (6.3 / 4.3)
    coef_legend_x  =  1.65
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
    mp.plot(type='line',ax=ax_coe2, x=[[coef_legend_x-0.24, coef_legend_x-0.09]], y=[[yy, yy]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_coe2.text(coef_legend_x, yy, 'True\ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe2.text(box_coe2['left']+dx, box_coe2['top']+0.04, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # # e -- inset fitness model picture
    # use mpimg.imread to read the image
    img = mpimg.imread('%s/fitness_model.png'%FIG_DIR)
    # ax_fit.imshow(img,aspect='equal') # display the image
    # ax_fit.axis('off') # no axis

    # obtain the width and height of the image
    height, width, _ = img.shape
    x_min, x_max = 0, width
    y_min, y_max = -height / 2, height / 2

    # show the image and set the extent
    ax_fit.imshow(img, extent=[x_min, x_max, y_min, y_max])

    # adjust the x-axis range to make the image fit the left side of the figure
    ax_fit.set_xlim(0, width*1.6)  # the image fit the left side of the figure
    ax_fit.set_ylim(-height / 2, height / 2)  # the image fit the middle side of the figure

    # close the axis
    ax_fit.axis('off')

    ax_fit.text(box_fit['left']+0.02, box_fit['top']-0.02, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # SAVE FIGURE
    if show_fig:
        plt.savefig('%s/fig1-sim.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        print('Figure saved as fig1-sim.pdf')
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
    plt.savefig('%s/sim-his.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as sim-his.pdf')

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
        try:
            df_epitope = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        except FileNotFoundError:
            continue
        df_epitope = df_epitope[(df_epitope.escape == True)]

        for i in range(len(df_epitope)):
            sc_all_notrait.append(df_epitope.iloc[i].sc_old)
            sc_all.append(df_epitope.iloc[i].sc_MPL)

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

    # add labels
    ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ax_old.text(0.02, 0.21, 'Without escape trait', **DEF_LABELPROPS)
    ax_new.text(0.02, 0.21, 'With escape trait', **DEF_LABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/sc_escape.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as sc_escape.pdf')


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
        try:
            df_tc     = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
            df_tc_noR = pd.read_csv('%s/noR/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        except FileNotFoundError:
            continue
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
    print('Figure saved as tc_rec.pdf')


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
        try:
            df_tc     = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        except FileNotFoundError:
            continue
        unique_traits = df_tc['epitope'].unique()
        for epitope in unique_traits:
            tc.append(df_tc[df_tc.epitope == epitope].iloc[0].tc_MPL)

    print(f'There are {len(tc)} epitopes can be seen as the binary traits.')
    print(f'Highest escape coefficient: {max(tc)}, average escape coefficient: {np.mean(tc)}')
    print(f'Median escape coefficient: {np.median(tc)}, lowest escape coefficient: {min(tc)}')

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
               'yticklabels': [0, 1],
               'nudgey':      1.1,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Fitness gain fraction \ndue to escape',
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
    plt.savefig('%s/fig2-HIV.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as fig2-HIV.pdf')


def plot_CH470_3(**pdata):
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
               'xlabel':      'Days after Fiebig I/II',
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
        var_epi.append(var_tag[i])
    pprops = { 'colors':      [var_color],
               'xlim':        [0, len(var_tag)],
               'xticks'  :    bar_x,
               'ylim':        [    0,  0.16],
               'yticks':      [    0, 0.08, 0.16],
               'yminorticks': [ 0.04, 0.12],
               'yticklabels': [    0,    8,    16],
               'xticklabels': [k for k in var_epi],
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
    plt.savefig('%s/fig3-CH470-3.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as fig3-CH470-3.pdf')

def plot_CH131_3(**pdata):
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
               'xlabel':      'Days after Fiebig I/II',
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
    var_tag.append('EV11')
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
    plt.savefig('%s/fig4-CH131-3_escape.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as fig4-CH131-3_escape.pdf')

def plot_CH470_5(**pdata):

    # unpack data

    tag        = pdata['tag']

    traj_ticks = pdata['traj_ticks']
    variants   = pdata['variants']   # variants have strong linkage with 17A (HXB2 index 974)
    high_var   = pdata['high_var']   # variants that need to be highlighted
    note_var   = pdata['note_var']   # variants that need to be texted

    # process stored data

    df_poly = pd.read_csv('%s/analysis/%s-analyze.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
    df_epi  = df_poly[df_poly['tc_MPL'].notna()]
    epitopes = df_epi['epitope'].unique()
    seq_length = df_poly.iloc[-1]['polymorphic_index']+1

    df_sij  = pd.read_csv('%s/sij/%s-sij.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_sold  = [] # selection coefficient for all variants without binary traits
    var_snew  = [] # selection coefficient for all variants with binary traits
    esc_sold  = [] # selection coefficient for escape mutations without binary traits
    esc_snew  = [] # selection coefficient for escape mutations with binary traits
    epi_name  = [] # the name for each independent epitopes
    var_traj  = [] # frequency trajectory for strong linked variants (non-hightlighted one)
    hig_traj  = [] # frequency trajectory for strong linked variants (hightlighted one)
    ds_matrix = [] # s_ij matrix between linked variants
    var_name  = [] # name for each variants (HXB2 index + nucleotide)
    text_sc   = [] # selection coefficient for variants that need to be texted
    text_name = [] # name for each variants that need to be texted

    '''Get the selection coefficients w/ vs. w/o binary traits for all variants'''
    for i in range(seq_length): 
        df_esc  = df_poly[(df_poly.polymorphic_index==i)& (df_poly.sc_MPL != 0) & (df_poly.nucleotide != '-') ]
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.epitope not in epitopes:
                var_sold.append(df_entry.sc_old)
                var_snew.append(df_entry.sc_MPL)

    for i in range(len(epitopes)): # escape mutations
        df_esc  = df_poly[(df_poly.epitope==epitopes[i]) & (df_poly.sc_MPL != 0) & (df_poly.nucleotide != '-') ]
        epi_nuc = ''.join(epitopes[i])
        epi_name.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))
        sold = []
        snew = []
        for df_iter, df_entry in df_esc.iterrows():
            if pd.notna(df_entry.tc_MPL):
                sold.append(df_entry.sc_old)
                snew.append(df_entry.sc_MPL)
        esc_sold.append(sold)
        esc_snew.append(snew)

    '''Find the frequencies and s_ij values for variants that have strong linkage with 17A'''
    for i in range(len(variants)):
        site_i = int(variants[i].split('_')[0])
        nuc_i  = variants[i].split('_')[-1]

        df_esc  = df_poly[(df_poly.polymorphic_index==site_i)& (df_poly.nucleotide == nuc_i)]
        df_ds   = df_sij[(df_sij.target_polymorphic_index == str(site_i)) & (df_sij.target_nucleotide==nuc_i)]

        HXB2_in = df_esc.iloc[0].HXB2_index
        var_name.append(str(HXB2_in)+nuc_i)

        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.polymorphic_index not in high_var:
                var_traj.append([df_entry['f_at_%d' % t] for t in times])
            else:
                hig_traj.append([df_entry['f_at_%d' % t] for t in times])

        ds_vec = [] # s_ij values for each linked variant for index i
        for j in range(len(variants)):
            site_j = int(variants[j].split('_')[0])
            nuc_j  = variants[j].split('_')[-1]
            if j == i:
                ds_vec.append(0)
            else:
                ds_vec.append(df_ds[(df_ds.mask_polymorphic_index==str(site_j)) & (df_ds.mask_nucleotide==nuc_j)].iloc[0].effect)
        
        for ii in range(len(epitopes)):
            index_m = 'epi'+ str(ii)
            ds_vec.append(df_ds[(df_ds.mask_polymorphic_index==index_m) & (df_ds.mask_nucleotide=='polypart')].iloc[0].effect)
        ds_matrix.append(ds_vec)

    '''Find sc values for variants that need to be texted'''
    for i in range(len(note_var)):
        index   = int(note_var[i].split('_')[0])
        neucleo = note_var[i].split('_')[-1]

        df_esc  = df_poly[(df_poly.polymorphic_index==index)& (df_poly.nucleotide == neucleo)]

        HXB2_i  = df_esc.iloc[0].HXB2_index
        text_name.append(str(HXB2_i)+neucleo)

        for df_iter, df_entry in df_esc.iterrows():
            text_sc.append([df_entry.sc_MPL,df_entry.sc_old])

    # PLOT FIGURE
    ## set up figure grid
    w     = DOUBLE_COLUMN
    goldh = w/1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_l  = 0.10
    box_r  = 0.95
    box_t  = 0.94
    box_b  = 0.09
    box_x  = 0.30
    box_y  = (w/goldh) * box_x
    box_dy = 0.10
    box_dx = 0.08


    box_ss   = dict(left=box_l, right=box_l+box_x, bottom=box_t-box_y, top=box_t)
    box_traj = dict(left=box_l, right=box_l+box_x, bottom=box_b, top=box_t-box_y-box_dy)
    box_sij  = dict(left=box_l+box_x+box_dx, right=box_r, bottom=box_t-box_y-0.013, top=box_t)
    box_bar  = dict(left=box_l+box_x+box_dx, right=box_r, bottom=box_b, top=box_t-box_y-1.5*box_dy)

    gs_ss   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_ss)
    gs_traj = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_sij  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sij)
    gs_bar  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_bar)

    ax_ss   =  plt.subplot(gs_ss[0, 0])
    ax_traj =  plt.subplot(gs_traj[0, 0])
    ax_sij  =  plt.subplot(gs_sij[0, 0])
    ax_bar  =  plt.subplot(gs_bar[0, 0])

    dx = -0.05
    dy =  0.04

    var_c = sns.husl_palette(len(epi_name)+len(hig_traj))
    traj_c = [var_c[-2],var_c[-1]]

    ## a -- inferred selection coefficients with VS. without binary trait term

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
    for i in range(len(esc_snew)):
        for j in range(len(esc_snew[i])):
            mp.plot(type='scatter', ax=ax_ss, x=[[esc_snew[i][j]]], y=[[esc_sold[i][j]]], colors=[var_c[i]],plotprops=scatterprops, **pprops)

    traj_legend_x  = 0.008
    traj_legend_dy = -0.0035
    y0             = -0.0025
    dx0            = 0.002
    traj_legend_y = [y0 + traj_legend_dy*k for k in range(len(epi_name))]
    scatterprops['s'] = SMALLSIZEDOT*0.8
    for k in range(len(epi_name)):
        traj_legend_k = 'Escape variants in epitope '+epi_name[k]
        mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[traj_legend_y[k]]], colors=[var_c[k]],plotprops=scatterprops, **pprops)
        ax_ss.text(traj_legend_x, traj_legend_y[k], traj_legend_k, ha='left', va='center', **DEF_LABELPROPS)

    mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[y0 + traj_legend_dy*(len(epi_name))]], colors=[C_NEU],plotprops=scatterprops, **pprops)
    ax_ss.text(traj_legend_x, y0 + traj_legend_dy*(len(epi_name)), 'Not escape variants', ha='left', va='center', **DEF_LABELPROPS)

    ddx = -0.0001
    ddy =  0.0025
    for i in range(len(text_sc)):
        ax_ss.text(text_sc[i][0]+ddx, text_sc[i][1]+ddy, text_name[i], ha='center', va='center', **DEF_LABELPROPS)

    ax_ss.text(box_ss['left']+dx, box_ss['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Days after Fiebig I/II',
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
    traj_legend_dy = -0.12
    traj_legend_y  = [0.42, 0.42 + traj_legend_dy, 0.42 + 2*traj_legend_dy]
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
               'ylim': [1, len(ds_matrix)+1],
               'xticks': [],
               'yticks': [],
               'plotprops': dict(lw=0, s=0.1*SMALLSIZEDOT, marker='o', clip_on=False),
               'ylabel': '',
               'theme': 'open',
               'hide' : ['top', 'bottom', 'left', 'right'] }

    mp.plot(type='scatter', ax=ax_sij, x=[[5]], y=[[5]], **pprops)

    site_rec_props = dict(height=1, width=1, ec=None, lw=AXWIDTH/2, clip_on=False)
    rec_patches    = []

    '''Get color and set the rectangle position''' 
    '''s_ij part'''
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

    '''box for heatmap'''
    individal_rec_props = dict(height=len(ds_matrix), width=len(ds_matrix),
                             ec=BKCOLOR,fc='none', lw=AXWIDTH/2, clip_on=False)
    epitope_rec_props   = dict(height=len(ds_matrix), width=len(ds_matrix[0]) - len(ds_matrix),
                             ec=BKCOLOR, fc='none', lw=AXWIDTH/2, clip_on=False)
    rec_patches.append(matplotlib.patches.Rectangle(xy=(0, 1), **individal_rec_props))
    rec_patches.append(matplotlib.patches.Rectangle(xy=(len(ds_matrix)+0.3, 1), **epitope_rec_props))

    for patch in rec_patches:
        ax_sij.add_artist(patch)

    '''name for each variant and epitope'''
    txtprops = dict(ha='right', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    for i in range(len(ds_matrix)):
        ax_sij.text(-0.2, len(ds_matrix) - i + 0.5, '%s' % var_name[i], **txtprops)

    txtprops = dict(ha='center', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    for i in range(len(ds_matrix)):
        ax_sij.text(i + 0.5, 0.8, '%s' % var_name[i], **txtprops)
    for i in range(len(epi_name)):
        ax_sij.text(len(ds_matrix) + i + 0.8, 0.8, '%s' % epi_name[i], **txtprops)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_sij.text(6.5, -1.4, 'Variant $i$', **txtprops)
    ax_sij.text(5.5,11.5, 'Most influential variants', **txtprops)
    ax_sij.text(12.3,11.5, 'Epitopes', **txtprops)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    ax_sij.text(-2.2, 7, 'Target variant $j$', **txtprops)

    ax_sij.text(box_sij['left']+dx, box_ss['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ## d -- bar plot for s_ij heatmap
    pprops = { 'colors': [BKCOLOR],
               'xlim': [0, len(ds_matrix[0]) + 1],
               'ylim': [1, 4],
               'xticks': [],
               'yticks': [],
               'plotprops': dict(lw=0, s=0.1*SMALLSIZEDOT, marker='o', clip_on=False),
               'ylabel': '',
               'theme': 'open',
               'hide' : ['top', 'bottom', 'left', 'right'] }

    mp.plot(type='scatter', ax=ax_bar, x=[[5]], y=[[2.4]], **pprops)

    '''legend part, ranged from -0.02 to 0.02'''
    rec_patches    = []
    for i in range(9):
        t = i/4 - 1 # the first i correspond to coefficient -0.02
        if t>0:
            c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
        else:
            c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
        rec = matplotlib.patches.Rectangle(xy=(i+2.5, 2.2), fc=c, **site_rec_props)
        rec_patches.append(rec)

    for patch in rec_patches:
        ax_bar.add_artist(patch)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, clip_on=False)

    y_label = 1.7
    ax_bar.text(  3, y_label,  -2, **txtprops)
    ax_bar.text(  5, y_label,  -1, **txtprops)
    ax_bar.text(  7, y_label,   0, **txtprops)
    ax_bar.text(  9, y_label,   1, **txtprops)
    ax_bar.text( 11, y_label,   2, **txtprops)
    ax_bar.text(  7, y_label-1.3, 'Effect of variant $i$ on inferred\nselection coefficient '+' $\hat{s}_j$'+\
                          ', $\Delta \hat{s}_{ij}$ (%)', ha='center', va='center', **DEF_LABELPROPS)


    # SAVE FIGURE
    plt.savefig('%s/fig5-CH470-5_linkage.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as fig5-CH470-5_linkage.pdf')

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
               'yticklabels': [0, 1],
               'nudgey':      1.1,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Fitness gain fraction \ndue to reversions',
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

    # legend
    traj_legend_x =  np.log(200)
    traj_legend_y = [0.8, 0.65]
    traj_legend_t = ['Individual', 'Average']
    x1 = traj_legend_x-0.4
    x2 = traj_legend_x-0.1
    y1 = traj_legend_y[0] + 0.012
    y2 = traj_legend_y[1] + 0.012
    pprops['plotprops']['alpha'] = 0.5
    pprops['plotprops']['lw'] = SIZELINE
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[0]], **pprops)
    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*1.8
    mp.line(ax=ax_frac, x=[[x1, x2]], y=[[y2, y2]], colors=[C_NEU], **pprops)
    ax_frac.text(traj_legend_x, traj_legend_y[0], traj_legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_frac.text(traj_legend_x, traj_legend_y[1], traj_legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/reversion.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as reversion.pdf')

@dataclass
class Result:
    seq_length: int
    special_sites: List[int]
    uniq_t: List[float]
    escape_group: List[int]
    escape_TF: List[bool]

def AnalyzeData(tag):
    df_info  = pd.read_csv('data/HIV/analysis/%s-analyze.csv' %tag, comment='#', memory_map=True)
    seq      = np.loadtxt('data/HIV/input/sequence/%s-poly-seq2state.dat'%tag)

    """Get raw time points"""
    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])
    uniq_t = np.unique(times)

    """Get variants number and sequence length"""
    seq_length = len(seq[0])-2

    """Get escape information"""
    escape_group  = [] # binary trait group, the corresponding epitope is independent
    escape_TF     = [] # corresponding wild type or synonymous mutant nucleotide

    try:
        df_trait = pd.read_csv('data/HIV/group/escape_group-%s.csv' %tag, comment='#', memory_map=True)
        # get all binary traits for one tag
        df_rows = df_trait[df_trait['epitope'].notna()]
        unique_traits = df_rows['epitope'].unique()

        """Get binary sites"""
        for epi in unique_traits:
            # collect all escape sites for one binary trait
            df_e = df_rows[(df_rows['epitope'] == epi)] # find all escape mutation for this epitope
            unique_sites = df_e['polymorphic_index'].unique()
            unique_sites = [int(site) for site in unique_sites]
            escape_group.append(list(unique_sites))
        
        """Get special sites and TF sequence"""
        df_epi = df_info[(df_info['epitope'].notna()) & (df_info['escape'] == True)]
        nonsy_sites = df_epi['polymorphic_index'].unique() # all sites can contribute to epitope
        
        for n in range(len(escape_group)):
            escape_TF_n = []
            for site in escape_group[n]:
                # remove escape sites to find special sites
                index = np.where(nonsy_sites == site)
                nonsy_sites = np.delete(nonsy_sites, index)

                # find the corresponding TF
                escape_TF_site = []
                df_TF = df_info[(df_info['polymorphic_index'] == site) & (df_info['escape'] == False)]
                for i in range(len(df_TF)):
                    TF = df_TF.iloc[i].nucleotide
                    escape_TF_site.append(int(NUC.index(TF)))
                escape_TF_n.append(escape_TF_site)
            escape_TF.append(escape_TF_n)
        
    except FileNotFoundError:
        df_epi = df_info[(df_info['epitope'].notna()) & (df_info['escape'] == True)]
        # since no traits, all nonsy_sites contribute to dependent epitopes
        nonsy_sites = df_epi['polymorphic_index'].unique() 

    return Result(seq_length,nonsy_sites,uniq_t,escape_group,escape_TF)

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
    seq       = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    result    = AnalyzeData(tag)
    traitsite = result.escape_group
    polyseq   = result.escape_TF
    s_sites   = result.special_sites
    uniq_t    = result.uniq_t

    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])

    SCMatrix = getSC(tag)
    ECMatrix = getEC(tag)
    FitAll   = np.zeros((2,len(uniq_t)))# 0:total fitness, 1: fitness increase due to escape (group+special sites)
    CountAll = np.zeros(len(uniq_t))    # population number at different times
    for t in range(len(uniq_t)):
        tid = times==uniq_t[t]

        cout_t = seq[tid][:,1]
        seq_t  = seq[tid][:,2:]
        counts = np.sum(cout_t)

        fit_a = 0 # total fitness
        fit_e = 0 # fitness contribution due to escape

        for i in range(len(seq_t)):
            # Average fitness (individual term + escape term)
            fitness   = 1  
            
            # Average fitness contribution due to escape (independent epitopes + dependent epitopes)
            # Independent epitopes : trait coefficients for that traits
            # Dependent epitopes   : selection coefficients for special sites
            fitness_e = 0  

            # Fitness contribution from individual selection
            for j in range(len(seq_t[0])):
                fitness += SCMatrix[j,int(seq_t[i][j])]

            # Fitness contribution from escape sites
            # Escape fitness contribution due to escape sites
            for ii in range(len(traitsite)):

                Mutant = False
                for jj in range(len(traitsite[ii])):
                    if seq_t[i][int(traitsite[ii][jj])] not in polyseq[ii][jj]:
                        Mutant = True
                        break
                                
                if Mutant:
                    fitness   += ECMatrix[ii]
                    fitness_e += ECMatrix[ii]

            # Fitness contribution from special sites
            # Escape fitness contribution due to special sites
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
                Mutant = False
                for jj in range(len(traitsite[ii])):
                    if seq_t[i][int(traitsite[ii][jj])] not in polyseq[ii][jj]:
                        Mutant = True
                        break
                                
                if Mutant:
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
    Fraction for escape part, reversion part and the sum of them
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
               'yticklabels': [0, 1],
               'nudgey':      1.1,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Fitness gain fraction', # \ndue to escape (CH-%s)'%ppt[-3:],
               'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.5},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    time = np.log(whole_time+1)

    # due to escape
    mp.line(ax=ax_frac, x=[time], y=[IntFractions[0]], colors=[C_group[0]], **pprops)

    # due to reversion
    mp.line(ax=ax_frac, x=[time], y=[IntFractions[1]], colors=[C_group[1]], **pprops)

    # # y=0 and y=1
    # pprops['plotprops']['ls'] = '--'
    # mp.line(ax=ax_frac,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)
    # mp.line(ax=ax_frac,x=[[0,6.5]], y=[[1,1]],colors=[C_NEU], **pprops)

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
    print('Figure saved as reversion-CH%s.pdf' % ppt[-3:])


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
    print('Figure saved as sim_his_rec.pdf')

def plot_sum_fraction(**pdata):
    """
    Fraction for escape part, reversion part and the sum of them
    """

    # unpack passed data
    ppts   = pdata['ppts']

    # get all escape/reversion contribution
    fractions_epi = []
    fractions_rev = []
    common_times = []
    for i in range(len(ppts)):
        ppt = ppts[i]
        tag_3 = ppt + '-' + str(3)
        tag_5 = ppt + '-' + str(5)

        # get fraction due to escape part
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
        
        # get fraction due to reversion part
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
        
        # record the information
        common_times.append(common_t)
        fractions_epi.append(fraction_epi)
        fractions_rev.append(fraction_rev)

    max_times = [max(common_times[i]) for i in range(len(common_times))]
    max_time = int(max(max_times))

    whole_time = np.linspace(0,max_time,max_time+1)
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(0,0), bounds_error=False)

    IntFractions_epi = np.zeros((len(common_times),len(whole_time)))
    IntFractions_rev = np.zeros((len(common_times),len(whole_time)))
    IntNumber    = np.zeros((len(common_times),len(whole_time)))
    for i in range(len(common_times)):
        IntFractions_epi[i] = interpolation(common_times[i], fractions_epi[i])(whole_time)
        IntFractions_rev[i] = interpolation(common_times[i], fractions_rev[i])(whole_time)
        IntNumber[i] = interpolation(common_times[i], np.ones(len(common_times[i])))(whole_time)
    IntFractions_sum = IntFractions_epi + IntFractions_rev

    AveFraction_epi = np.zeros(len(whole_time))
    AveFraction_rev = np.zeros(len(whole_time))
    for t in range(len(whole_time)):
        fraction_t_epi = np.sum(IntFractions_epi[:,t])
        fraction_t_rev = np.sum(IntFractions_rev[:,t])
        number_t   = np.sum(IntNumber[:,t])
        AveFraction_epi[t] = fraction_t_epi/number_t
        AveFraction_rev[t] = fraction_t_rev/number_t
    
    AveFraction_sum = AveFraction_epi + AveFraction_rev

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_frac   = dict(left=0.20, right=0.92, bottom=0.22, top=0.95)
    gs_frac    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_frac)
    ax_frac    = plt.subplot(gs_frac[0, 0])

    ### plot fraction
    pprops = { #'xticks':      [ 0,  100, 200, 300,  400, 500, 600, 700],
               'xticks':      [ 0,  np.log(11),np.log(51),np.log(101), np.log(201),np.log(401),np.log(701)],
               'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
               'ylim'  :      [0., 1.2],
               'yticks':      [0., 1],
               'yminorticks': [0.25, 0.5, 0.75],
               'yticklabels': [0, 1],
               'nudgey':      1.0,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Fitness gain fraction', # \ndue to escape trait',
               'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.3},
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    # individual fraction
    for i in range(len(common_times)):
        max_t_i = int(max(common_times[i]))
        time_i  = np.linspace(0,max_t_i,max_t_i+1)
        time    = np.log(time_i+1)
        # mp.line(ax=ax_frac, x=[time], y=[IntFractions_epi[i][:max_t_i+1]], colors=[C_group[0]], **pprops)
        # mp.line(ax=ax_frac, x=[time], y=[IntFractions_rev[i][:max_t_i+1]], colors=[C_group[1]], **pprops)
        mp.line(ax=ax_frac, x=[time], y=[IntFractions_sum[i][:max_t_i+1]], colors=[C_group[2]], **pprops)

    # # 0
    # pprops['plotprops']['ls'] = '--'
    # mp.line(ax=ax_frac,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)
    # mp.line(ax=ax_frac,x=[[0,6.5]], y=[[1,1]],colors=[C_NEU], **pprops)

    # average curve
    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*1.8
    pprops['plotprops']['ls'] = '-'
    time = np.log(whole_time+1)
    mp.line(ax=ax_frac,x=[time], y=[AveFraction_epi],colors=[C_group[0]], **pprops)
    mp.line(ax=ax_frac,x=[time], y=[AveFraction_rev],colors=[C_group[1]], **pprops)
    mp.plot(type='line', ax=ax_frac, x=[time], y=[AveFraction_sum],colors=[C_group[2]], **pprops)

    # legend
    traj_legend_x  = np.log(1.8)
    traj_legend_y  = [0.90, 0.80, 0.70]
    traj_legend_t  = ['Escape','Reversion','Sum']

    x1 = traj_legend_x-0.4
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

    plt.savefig('%s/fraction_sum.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as fraction_sum.pdf')


def plot_trait_site_reversion(**pdata):
    """
    Histogram of selection coefficients for trait sises
    Use different color to represent reversion mutation and non-reversion mutation
    """

    # unpack passed data
    tags   = pdata['tags']

    # get all selection coefficients for escape mutations
    sc_all_old = [] # escape mutations
    sc_rev_old = [] # escape mutations that are also reversions
    sc_all_new = [] # escape mutations
    sc_rev_new = [] # escape mutations that are also reversions
    for tag in tags:
        try:
            df_epitope = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        except FileNotFoundError:
            continue
    
        df_reversion = df_epitope[(df_epitope['nucleotide'] == df_epitope['consensus'])]

        for i in range(len(df_epitope)):
            sc_all_old.append(df_epitope.iloc[i].sc_old)
            sc_all_new.append(df_epitope.iloc[i].sc_MPL)

        for i in range(len(df_reversion)):
            sc_rev_old.append(df_reversion.iloc[i].sc_old)
            sc_rev_new.append(df_reversion.iloc[i].sc_MPL)

    pos_old = [i for i in sc_all_old if i > 0.01]
    pos_new = [i for i in sc_all_new if i > 0.01]

    rev_pos_old = [i for i in sc_rev_old if i > 0]
    rev_pos_new = [i for i in sc_rev_new if i > 0]

    print(f'Totally {len(sc_all_old)} escape mutations and', end=' ')
    print(f'{len(sc_rev_old)} ({len(sc_rev_old)/len(sc_all_old)*100:.2f}%) reversion mutations')
    
    print('For all escape mutations:')
    print(f'Before counting escape term, {len(pos_old)}', end='')
    print(f' ({len(pos_old)/len(sc_all_old)*100:.2f}%) escape mutations are substantially beneficial (s>1%).')
    print(f'After counting escape term, only {len(pos_new)}', end='')
    print(f' ({len(pos_new)/len(sc_all_new)*100:.2f}%) escape mutations are substantially beneficial.')

    # print('For reversion mutations:')
    # print(f'Before counting escape term, {len(rev_pos_old)}', end='')
    # print(f' ({len(rev_pos_old)/len(sc_all_old)*100:.2f}%) reversion mutations are positive.')
    # print(f'After counting escape term, only {len(rev_pos_new)}', end='')
    # print(f' ({len(rev_pos_new)/len(sc_all_new)*100:.2f}%)reversion mutations are positive.')


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

    ### plot histogram of selection coefficients
    histprops = dict(lw=SIZELINE/2, width=0.004, align='center', orientation='vertical',alpha=0.5, edgecolor='none')
    lineprops = dict(lw=SIZELINE*6, linestyle='-', alpha=1)

    pprops = { 'xlim'       : [ -0.05,  0.10],
               'xticks'     : [ -0.05,     0,  0.05, 0.1],
               'xminorticks': [ -0.025, 0.025, 0.075],
               'xticklabels': [ ],
               'ylim'       : [0, 3.5], #[0., 0.50],
               'yticks'     : [1+np.log10(1), 1+np.log10(10), 1+np.log10(100)], #, np.log10(50)], #[0., 0.25, 0.50],
               'yticklabels': [            1,             10,             100], #,         50], #[0,25, 50],
               'ylabel'     : 'Counts',
               'theme'      : 'boxed' }
    
    bins = np.arange(-0.05, 0.10, 0.005)

    #### a. without escape terms
    # all escape mutations
    all_old, bin_edges = np.histogram(sc_all_old, bins=bins)
    all_old            = [1+np.log10(val) if val>0 else 0 for val in all_old]
    # all_old            = all_old/len(sc_all_old)
    bar_x = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_new, bin_edges = np.histogram(sc_all_new, bins=bins)
    all_new            = [1+np.log10(val) if val>0 else 0 for val in all_new]
    # all_new            = all_new/len(sc_all_new)

    mp.bar(ax=ax_old, x=[bar_x], y=[all_old], colors=[C_group[0]], plotprops=histprops, **pprops)
    mp.bar(ax=ax_new, x=[bar_x], y=[all_new], colors=[C_group[0]], plotprops=histprops, **pprops)

    # escape mutations that are also reversions
    rev_old, bin_edges = np.histogram(sc_rev_old, bins=bins)
    rev_old            = [1+np.log10(val) if val>0 else 0 for val in rev_old]
    # rev_old            = rev_old/len(sc_all_old)
    
    rev_new, bin_edges = np.histogram(sc_rev_new, bins=bins)
    rev_new            = [1+np.log10(val) if val>0 else 0 for val in rev_new]
    # rev_new            = rev_new/len(sc_all_new)

    histprops['alpha'] = 1
    mp.bar(ax=ax_old, x=[bar_x], y=[rev_old], colors=[C_group[1]], plotprops=histprops, **pprops)
    mp.bar(ax=ax_new, x=[bar_x], y=[rev_new], colors=[C_group[1]], plotprops=histprops, **pprops)

    # legend
    legend_x  =  0.065
    title_y   =  1+np.log10(100) # 0.40
    legend_y  = [1+np.log10(100/2.2), 1+np.log10(100/(2.2**2))]#[0.25,0.32]
    legend_t  = ['Reversion', 'Not reversion']

    x1 = legend_x-0.01
    x2 = legend_x-0.004
    y1 = legend_y[0] + 0.0025
    y2 = legend_y[1] + 0.0025

    # escape mutations that are also reversions
    ax_old.text(x1, title_y, 'Without escape trait', **DEF_LABELPROPS)
    
    mp.line(            ax=ax_old, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[1]], plotprops=lineprops, **pprops)
    lineprops['alpha'] = 0.5
    mp.plot(type='line',ax=ax_old, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)

    ax_old.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_old.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # other escape mutations
    pprops['xticks']      = [ -0.05,     0,  0.05, 0.1]
    pprops['xticklabels'] = [    -5,     0,    5,   10]
    pprops['xlabel']      = 'Inferred selection coefficient for escape mutations, ' + r'$\hat{s}$ ' +'(%)'

    ax_new.text(x1, title_y, 'With escape trait', **DEF_LABELPROPS)
    
    lineprops['alpha'] = 1
    mp.line(            ax=ax_new, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[1]], plotprops=lineprops, **pprops)
    lineprops['alpha'] = 0.5
    mp.plot(type='line',ax=ax_new, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)
    
    ax_new.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_new.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # label
    ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/sc_escape_reversion.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as sc_escape_reversion.pdf')

def get_participation_ratio(tag):

    df_analysis = pd.read_csv('%s/analysis/%s-analyze.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
    times = [int(i.split('_')[-1]) for i in df_analysis.columns if 'f_at_' in i]
    
    seq      = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    CountAll = np.zeros(len(times))    # population number at different times
    for t in range(len(times)):
        seq_t = seq[seq[:,0] == times[t]]
        CountAll[t] = np.sum(seq_t[:,1])
        
    try:
        df_epitope  = pd.read_csv('%s/group/escape_group-%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
        df_rows  = df_epitope[df_epitope['epitope'].notna()]
        epitopes = df_rows['epitope'].unique()
    except FileNotFoundError:
        epitopes = []    

    f     = np.zeros(len(times))
    for t in range(len(times)):
        time = times[t]
        for i in range(len(df_analysis)):
            f_t      = df_analysis.iloc[i]['f_at_%d' % time]
            f[t]    += df_analysis.iloc[i].sc_MPL * f_t

        for n in range(len(epitopes)):
            df_epi_n = df_epitope[df_epitope['epitope'] == epitopes[n]]
            tf_t     = df_epi_n.iloc[0]['xp_at_%d' % time]
            f[t]    += df_epi_n.iloc[0].tc_MPL * tf_t

    ff     = np.zeros(len(times))
    for t in range(len(times)):
        time = times[t]
        if f[t] != 0:
            for i in range(len(df_analysis)):
                f_t      = df_analysis.iloc[i]['f_at_%d' % time]
                ff[t]   += (df_analysis.iloc[i].sc_MPL * f_t / f[t])**2

            for n in range(len(epitopes)):
                df_epi_n = df_epitope[df_epitope['epitope'] == epitopes[n]]
                tf_t     = df_epi_n.iloc[0]['xp_at_%d' % time]
                ff[t]   += (df_epi_n.iloc[0].tc_MPL * tf_t / f[t])**2

        # else:
        #     print(f'fitness gain for CH{tag[-5:]} at time point {time} is 0')
    
    return times, ff, CountAll

# def plotParticipationRatio(**pdata):
#     """

#     """

#     # unpack passed data
#     ppts   = pdata['ppts']

#     # get all 
#     ratio_all    = []
#     common_times = []
#     for i in range(len(ppts)):
#         ppt = ppts[i]
#         tag_3 = ppt + '-' + str(3)
#         tag_5 = ppt + '-' + str(5)

#         t_3, f_3, n_3 = get_participation_ratio(tag_3)
#         t_5, f_5, n_5 = get_participation_ratio(tag_5)

#         common_t  = np.intersect1d(t_3, t_5)
#         ratio_tag = np.zeros(len(common_t))
#         for i in range(len(common_t)):
#             index_3 = list(t_3).index(common_t[i])
#             index_5 = list(t_5).index(common_t[i])
#             ratio_tag[i]  = (f_3[index_3]*n_3[index_3]+f_5[index_5]*n_5[index_5])/(n_3[index_3]+n_5[index_5])

#         common_times.append(common_t)
#         ratio_all.append(ratio_tag)

#     max_times = [max(common_times[i]) for i in range(len(common_times))]
#     max_time = int(max(max_times))

#     whole_time = np.linspace(0,max_time,max_time+1)
#     interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(0,0), bounds_error=False)

#     IntRatio  = np.zeros((len(common_times),len(whole_time)))
#     IntNumber    = np.zeros((len(common_times),len(whole_time)))

#     for i in range(len(common_times)):
#         IntRatio[i]  = interpolation(common_times[i], ratio_all[i])(whole_time)
#         IntNumber[i] = interpolation(common_times[i], np.ones(len(common_times[i])))(whole_time)

#     AveRatio = np.zeros(len(whole_time))
#     for t in range(len(whole_time)):
#         ratio_t = np.sum(IntRatio[:,t])
#         number_t   = np.sum(IntNumber[:,t])
#         AveRatio[t] = ratio_t/number_t

#     # PLOT FIGURE
#     ## set up figure grid

#     w     = SINGLE_COLUMN
#     goldh = w / 2.2
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)

#     box_ratio = dict(left=0.20, right=0.92, bottom=0.22, top=0.95)
#     gs_ratio  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_ratio)
#     ax_ratio  = plt.subplot(gs_ratio[0, 0])

#     ### plot fraction
#     pprops = { #'xticks':      [ 0,  100, 200, 300,  400, 500, 600, 700],
#                'xticks':      [ 0,  np.log(10),np.log(50),np.log(100), np.log(200),np.log(400),np.log(700)],
#                'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
#                'ylim'  :      [0., 1.01],
#                'yticks':      [0., 1],
#                'yminorticks': [0.25, 0.5, 0.75],
#                'yticklabels': [0, 1],
#                'nudgey':      1.1,
#                'xlabel':      'Days after Fiebig I/II',
#                'ylabel':      'Participation ratio', # \sum (\frac{x_i s_i}{\Delta f})^2
#                'plotprops':   {'lw': SIZELINE, 'ls': '-','alpha':0.5},
#                'axoffset':    0.1,
#                'theme':       'open',
#                'combine'     : True}

#     for i in range(len(common_times)):
#         max_t_i = int(max(common_times[i]))
#         time_i  = np.linspace(0,max_t_i,max_t_i+1)
#         time    = np.log(time_i+1)
#         mp.line(ax=ax_ratio, x=[time], y=[IntRatio[i][:max_t_i+1]], colors=[C_group[0]], **pprops)

#     pprops['plotprops']['ls'] = '--'
#     mp.line(ax=ax_ratio,x=[[0,6.5]], y=[[0,0]],colors=[C_NEU], **pprops)

#     pprops['plotprops']['alpha'] = 1
#     pprops['plotprops']['lw'] = SIZELINE*1.8
#     pprops['plotprops']['ls'] = '-'
#     time = np.log(whole_time+1)
#     mp.plot(type='line', ax=ax_ratio, x=[time], y=[AveRatio],colors=[C_NEU], **pprops)



def plot_site_reversion(**pdata):
    """
    Histogram of all selection coefficients
    Use different color to represent reversion mutation and non-reversion mutation
    """

    # unpack passed data
    tags   = pdata['tags']

    # get all selection coefficients for escape mutations
    sc_all_old = [] # mutations
    sc_rev_old = [] # reversion mutations 
    sc_all_new = [] # mutations
    sc_rev_new = [] # reversion mutations
    for tag in tags:

        df = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        df_variant   = df[df['nucleotide'] != df['TF']] # all mutation
        df_reversion = df_variant[(df_variant['nucleotide'] == df_variant['consensus'])]

        for i in range(len(df_variant)):
            sc_all_old.append(df_variant.iloc[i].sc_old)
            sc_all_new.append(df_variant.iloc[i].sc_MPL)

        for i in range(len(df_reversion)):
            sc_rev_old.append(df_reversion.iloc[i].sc_old)
            sc_rev_new.append(df_reversion.iloc[i].sc_MPL)

    # rev_pos_old = [i for i in sc_rev_old if i > 0]
    # rev_pos_new = [i for i in sc_rev_new if i > 0]

    # print(f'Totally {len(sc_all_old)} mutations and', end=' ')
    # print(f'{len(sc_rev_old)} ({len(sc_rev_old)/len(sc_all_old)*100:.2f}%) reversion mutations')
    # print('For reversion mutations:')
    # print(f'Before counting escape term, {len(rev_pos_old)} reversion mutations are positive', end='')
    # print(f', which is {len(rev_pos_old)/len(sc_all_old)*100:.2f}% compared to all mutations')
    # print(f'After counting escape term, {len(rev_pos_new)} reversion mutations are positive', end='')
    # print(f', {len(rev_pos_new)/len(sc_all_new)*100:.2f}% compared to all mutations')

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

    ### plot histogram of selection coefficients
    histprops = dict(lw=SIZELINE/2, width=0.0015, align='center', orientation='vertical',alpha=0.5, edgecolor='none')
    lineprops = dict(lw=SIZELINE*3, linestyle='-', alpha=1)

    pprops = { 'xlim'       : [ -0.06,  0.12],
               'xticks'     : [ -0.06,     0,  0.06, 0.12],
               'xminorticks': [ -0.03, 0.03, 0.09],
               'xticklabels': [ ],
               'ylim'       : [0., 4.8],
               'yticks'     : [1+np.log10(1), 1+np.log10(10), 1+np.log10(100), 1+np.log10(1000)],
               'yticklabels': [            1,             10,             100,             1000],
               'ylabel'     : 'Counts',
               'theme'      : 'boxed' }
    
    bins   = np.arange(-0.05, 0.10, 0.002)

    #### a. without escape terms
    # all mutations
    all_old, bin_edges = np.histogram(sc_all_old, bins=bins)
    all_old            = [1+np.log10(val) if val>0 else 0 for val in all_old]
    # all_old            = all_old/len(sc_all_old)
    bar_x = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_new, bin_edges = np.histogram(sc_all_new, bins=bins)
    all_new            = [1+np.log10(val) if val>0 else 0 for val in all_new]
    # all_new            = all_new/len(sc_all_new)

    # log_all_old    = np.log10(all_old*100+1)
    # log_all_new    = np.log10(all_new*100+1)

    mp.bar(ax=ax_old, x=[bar_x], y=[all_old], colors=[C_group[0]], plotprops=histprops, **pprops)
    mp.bar(ax=ax_new, x=[bar_x], y=[all_new], colors=[C_group[0]], plotprops=histprops, **pprops)

    # reversions mutations 
    rev_old, bin_edges = np.histogram(sc_rev_old, bins=bins)
    rev_old            = [1+np.log10(val) if val>0 else 0 for val in rev_old]
    # rev_old            = rev_old/len(sc_all_old)
    
    rev_new, bin_edges = np.histogram(sc_rev_new, bins=bins)
    rev_new            = [1+np.log10(val) if val>0 else 0 for val in rev_new]
    # rev_new            = rev_new/len(sc_all_new)

    # log_rev_old    = np.log10(rev_old*100+1)
    # log_rev_new    = np.log10(rev_new*100+1)

    histprops['alpha'] = 1
    mp.bar(ax=ax_old, x=[bar_x], y=[rev_old], colors=[C_group[1]], plotprops=histprops, **pprops)
    mp.bar(ax=ax_new, x=[bar_x], y=[rev_new], colors=[C_group[1]], plotprops=histprops, **pprops)

    # legend
    legend_x  =  0.07
    title_y   = 1+np.log10(900)
    legend_y  = [1+np.log10(300), 1+np.log10(100)]
    legend_t  = ['Reversion', 'Not reversion']

    x1 = legend_x-0.01
    x2 = legend_x-0.004
    y1 = legend_y[0]+0.025
    y2 = legend_y[1]+0.025

    # reversions mutations
    ax_old.text(x1, title_y, 'Without escape trait', **DEF_LABELPROPS)
    
    mp.line(            ax=ax_old, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[1]], plotprops=lineprops, **pprops)
    lineprops['alpha'] = 0.5
    mp.plot(type='line',ax=ax_old, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)

    ax_old.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_old.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # other mutations
    pprops['xticks']      = [ -0.06,     0,  0.06, 0.12]
    pprops['xticklabels'] = [    -6,     0,     6,   12]
    pprops['xlabel']      = 'Inferred selection coefficient, ' + r'$\hat{s}$ ' +'(%)'

    ax_new.text(x1, title_y, 'With escape trait', **DEF_LABELPROPS)
    
    lineprops['alpha'] = 1
    mp.line(            ax=ax_new, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[1]], plotprops=lineprops, **pprops)
    lineprops['alpha'] = 0.5
    mp.plot(type='line',ax=ax_new, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)
    
    ax_new.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_new.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

    # label
    ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/sc_reversion.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as sc_reversion.pdf')

# def plot_site_escape(**pdata):
#     """
#     Histogram of all selection coefficients
#     Use different color to represent escape mutation on trait site and other mutations
#     """

#     # unpack passed data
#     tags   = pdata['tags']

#     # get all selection coefficients for escape mutations
#     sc_all_old = [] # mutations
#     sc_esc_old = [] # escape mutations on escape sites
#     sc_all_new = [] # mutations
#     sc_esc_new = [] # escape mutations on escape sites
#     for tag in tags:

#         df = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
#         df_variant   = df[df['nucleotide'] != df['TF']] # all mutation

#         for i in range(len(df_variant)):
#             sc_all_old.append(df_variant.iloc[i].sc_old)
#             sc_all_new.append(df_variant.iloc[i].sc_MPL)

#         try:   
#             df_epitope = pd.read_csv('%s/group/escape_group-%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
#             df_epitope = df_epitope[df_epitope['escape']==True]
#             for i in range(len(df_epitope)):
#                 sc_esc_old.append(df_epitope.iloc[i].sc_old)
#                 sc_esc_new.append(df_epitope.iloc[i].sc_MPL)
#         except FileNotFoundError:
#             pass

#     # PLOT FIGURE
#     ## set up figure grid

#     w     = SINGLE_COLUMN
#     goldh = w / 1.5
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)

#     box_old = dict(left=0.15, right=0.92, bottom=0.61, top=0.95)
#     box_new = dict(left=0.15, right=0.92, bottom=0.14, top=0.48)
#     gs_old  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_old)
#     gs_new  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_new)
#     ax_old  = plt.subplot(gs_old[0, 0])
#     ax_new  = plt.subplot(gs_new[0, 0])

#     dx = -0.10
#     dy =  0.02

#     ### plot histogram of selection coefficients
#     histprops = dict(lw=SIZELINE/2, width=0.0015, align='center', orientation='vertical',alpha=0.5, edgecolor='none')
#     lineprops = dict(lw=SIZELINE*3, linestyle='-', alpha=1)

#     pprops = { 'xlim'        : [ -0.06,  0.12],
#                'xticks'      : [ -0.06,     0,  0.06, 0.12],
#                'xticklabels' : [ ],
#                'ylim'        : [0., 1.6],
#                'yticks'      : [0., np.log10(1+1), np.log10(5+1), np.log10(10+1), np.log10(20+1), np.log10(40+1)],
#                'yticklabels' : [0.,          0.01,          0.05,           0.10,           0.20,            0.4],
#                #'ylim'        : [0., 0.40],
#                #'yticks'      : [0., 0.20, 0.40],
#                'ylabel'      : 'Frequency',
#                'theme'       : 'boxed' }
    
#     bins   = np.arange(-0.05, 0.10, 0.002)

#     #### a. without escape terms
#     # all escape mutations
#     all_old, bin_edges = np.histogram(sc_all_old, bins=bins)
#     all_old            = all_old/len(sc_all_old)
#     bar_x = (bin_edges[:-1] + bin_edges[1:]) / 2

#     all_new, bin_edges = np.histogram(sc_all_new, bins=bins)
#     all_new            = all_new/len(sc_all_new)

#     mp.bar(ax=ax_old, x=[bar_x], y=[np.log10(all_old*100+1)], colors=[C_group[0]], plotprops=histprops, **pprops)
#     mp.bar(ax=ax_new, x=[bar_x], y=[np.log10(all_new*100+1)], colors=[C_group[0]], plotprops=histprops, **pprops)

#     # escape mutations that are also reversions
#     rev_old, bin_edges = np.histogram(sc_esc_old, bins=bins)
#     rev_old            = rev_old/len(sc_all_old)
    
#     rev_new, bin_edges = np.histogram(sc_esc_new, bins=bins)
#     rev_new            = rev_new/len(sc_all_new)

#     histprops['alpha'] = 1
#     mp.bar(ax=ax_old, x=[bar_x], y=[np.log10(rev_old*100+1)], colors=[C_group[2]], plotprops=histprops, **pprops)
#     mp.bar(ax=ax_new, x=[bar_x], y=[np.log10(rev_new*100+1)], colors=[C_group[2]], plotprops=histprops, **pprops)

#     # legend
#     legend_x  =  0.04
#     title_y   =  np.log10(21) # 0.25
#     legend_y  = [np.log10(12),np.log10(7)] #[0.15,0.20]
#     legend_t  = ['Escape mutations', 'Other mutations']

#     x1 = legend_x-0.01
#     x2 = legend_x-0.004
#     y1 = legend_y[0] + 0.025 # 0.0025
#     y2 = legend_y[1] + 0.025 # 0.0025

#     # escape mutations that are also reversions
#     ax_old.text(x1, title_y, 'Without escape trait', **DEF_LABELPROPS)
    
#     mp.line(            ax=ax_old, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[2]], plotprops=lineprops, **pprops)
#     lineprops['alpha'] = 0.5
#     mp.plot(type='line',ax=ax_old, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)

#     ax_old.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
#     ax_old.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

#     # other escape mutations
#     pprops['xticks']      = [ -0.06,     0,  0.06, 0.12]
#     pprops['xticklabels'] = [    -6,     0,     6,   12]
#     pprops['xlabel']      = 'Inferred selection coefficient, ' + r'$\hat{s}$ ' +'(%)'

#     ax_new.text(x1, title_y, 'With escape trait', **DEF_LABELPROPS)
    
#     lineprops['alpha'] = 1
#     mp.line(            ax=ax_new, x=[[x1, x2]], y=[[y1, y1]], colors=[C_group[2]], plotprops=lineprops, **pprops)
#     lineprops['alpha'] = 0.5
#     mp.plot(type='line',ax=ax_new, x=[[x1, x2]], y=[[y2, y2]], colors=[C_group[0]], plotprops=lineprops, **pprops)
    
#     ax_new.text(legend_x, legend_y[0], legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
#     ax_new.text(legend_x, legend_y[1], legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

#     # label
#     ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#     ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)


# def plot_reversion_scatter(**pdata):
#     """
#     Scatter of selection coefficients for trait sises
#     Use different color to represent reversion mutation and non-reversion mutation
#     """

#     # unpack passed data
#     tags   = pdata['tags']

#     # get all selection coefficients for escape mutations
#     sc_all_old = [] # escape mutations
#     sc_rev_old = [] # escape mutations that are also reversions
#     sc_all_new = [] # escape mutations
#     sc_rev_new = [] # escape mutations that are also reversions
#     for tag in tags:
#         df_epitope = pd.read_csv('%s/group/escape_group-%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
#         df_epitope = df_epitope[df_epitope['escape'] == True]
#         df_reversion = df_epitope[(df_epitope['nucleotide'] == df_epitope['consensus'])]

#         for i in range(len(df_epitope)):
#             if df_epitope.iloc[i].nucleotide != df_epitope.iloc[i].consensus:
#                 sc_all_old.append(df_epitope.iloc[i].sc_old)
#                 sc_all_new.append(df_epitope.iloc[i].sc_MPL)

#         for i in range(len(df_reversion)):
#             sc_rev_old.append(df_reversion.iloc[i].sc_old)
#             sc_rev_new.append(df_reversion.iloc[i].sc_MPL)

#     # PLOT FIGURE
#     ## set up figure grid

#     w     = SINGLE_COLUMN
#     goldh = w / 1.5
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)

#     box_old = dict(left=0.15, right=0.92, bottom=0.61, top=0.95)
#     box_new = dict(left=0.15, right=0.92, bottom=0.14, top=0.48)
#     gs_old  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_old)
#     gs_new  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_new)
#     ax_old  = plt.subplot(gs_old[0, 0])
#     ax_new  = plt.subplot(gs_new[0, 0])

#     dx = -0.10
#     dy =  0.02

#     ### plot histogram of selection coefficients
#     sprops = dict(lw=AXWIDTH, s=2., marker='o', alpha=0.5)
#     lineprops = dict(lw=SIZELINE, linestyle='--', alpha=0.5)

#     pprops = { 'xlim'        : [ -0.3,  0.6],
#                'xticks'      : [ ],
#                'ylim'        : [ -0.05,  0.10],
#                'yticks'      : [ -0.05,     0,  0.05, 0.1],
#             #    'ylabel'      : 'Frequency',
#                'theme'       : 'open',
#                 'hide'       : ['bottom'] }

#     #### a. without escape terms
#     # all escape mutations
#     x_all = np.random.normal(0, 0.06,len(sc_all_old))
#     mp.scatter(ax=ax_old, x=[x_all], y=[sc_all_old], colors=[C_group[0]],plotprops=sprops, **pprops)
#     mp.scatter(ax=ax_new, x=[x_all], y=[sc_all_new], colors=[C_group[0]],plotprops=sprops, **pprops)

#     # escape mutations that are also reversions
#     sprops['alpha'] = 1
#     x_rev = np.random.normal(0, 0.03,len(sc_rev_old))
#     mp.scatter(ax=ax_old, x=[x_rev], y=[sc_rev_old], colors=[C_group[1]],plotprops=sprops, **pprops)
#     mp.scatter(ax=ax_new, x=[x_rev], y=[sc_rev_new], colors=[C_group[1]],plotprops=sprops, **pprops)

#     mp.line(ax=ax_old, x=[[-0.20, 0.20]], y=[[0, 0]], colors=[BKCOLOR], plotprops=lineprops, **pprops)
#     mp.line(ax=ax_new, x=[[-0.20, 0.20]], y=[[0, 0]], colors=[BKCOLOR], plotprops=lineprops, **pprops)

#     # escape mutations that are also reversions
#     # legend
#     traj_legend_x  =  0.30
#     traj_dot_x     = traj_legend_x-0.03
#     traj_legend_y1 = 0.04
#     traj_legend_y2 = 0.06
#     traj_legend_t  = ['Reversion mutation', 'Not reversion mutation']

#     # escape mutations that are also reversions
#     mp.scatter(ax=ax_old, x=[[traj_dot_x]], y=[[traj_legend_y1]], colors=[C_group[1]], plotprops=sprops, **pprops)
#     mp.scatter(ax=ax_new, x=[[traj_dot_x]], y=[[traj_legend_y1]], colors=[C_group[1]], plotprops=sprops, **pprops)

#     sprops['alpha'] = 0.5
#     mp.plot(type='scatter',ax=ax_old, x=[[traj_dot_x]], y=[[traj_legend_y2]], colors=[C_group[0]], plotprops=sprops, **pprops)
#     mp.plot(type='scatter',ax=ax_new, x=[[traj_dot_x]], y=[[traj_legend_y2]], colors=[C_group[0]], plotprops=sprops, **pprops)

#     ax_old.text(traj_legend_x, 0.08, 'Without escape trait', **DEF_LABELPROPS)
#     ax_new.text(traj_legend_x, 0.08, 'With escape trait', **DEF_LABELPROPS)

#     ax_old.text(traj_legend_x, traj_legend_y1, traj_legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
#     ax_old.text(traj_legend_x, traj_legend_y2, traj_legend_t[1], ha='left', va='center', **DEF_LABELPROPS)
    
#     ax_new.text(traj_legend_x, traj_legend_y1, traj_legend_t[0], ha='left', va='center', **DEF_LABELPROPS)
#     ax_new.text(traj_legend_x, traj_legend_y2, traj_legend_t[1], ha='left', va='center', **DEF_LABELPROPS)

#     # label
#     ax_old.text(box_old['left']+dx, box_old['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#     ax_new.text(box_new['left']+dx, box_new['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     # SAVE FIGURE
#     # plt.savefig('%s/sc_escape_new.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_example_tv(**pdata):
    """
    Example evolutionary trajectory for a 50-site system and inferred selection coefficients
    and trait coefficients, together with aggregate properties for different levels of sampling..
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    dg       = pdata['dg']
    pop_size = pdata['N']
    xfile    = pdata['xfile']
    xpath    = pdata['xpath']

    n_ben    = pdata['n_ben']
    n_neu    = pdata['n_neu']
    n_del    = pdata['n_del']
    n_tra    = pdata['n_tra']
    s_ben    = pdata['s_ben']
    s_neu    = pdata['s_neu']
    s_del    = pdata['s_del']
    s_tra    = pdata['s_tra']
    s_t_a    = pdata['s_t_a']

    r_seed = pdata['r_seed']
    np.random.seed(r_seed)

    # load and process data files

    data  = np.loadtxt('%s/%s/sequences/example-%s.dat' % (SIM_DIR, xpath, xfile))

    x_index = xfile.split('_')[0]
    trait_site = read_file('/%s/traitsite/traitsite-%s.dat'%(xpath, x_index))

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

    s_true  = [s_ben for i in range(n_ben)] + [0 for i in range(n_neu)] + [s_del for i in range(n_del)]
    
    # binary case
    s_inf   = np.loadtxt('%s/%s/output/sc-%s.dat' %(SIM_DIR,xpath,x_index))
    cov     = np.loadtxt('%s/%s/covariance/covariance-%s.dat' %(SIM_DIR,xpath,x_index))
    ds      = np.linalg.inv(cov) / pop_size

    ### distribution of inferred selection coefficients for multiple simulaitons

    df       = pd.read_csv('%s/mpl_collected_tv.csv' % SIM_DIR, memory_map=True)

    ben_cols = ['sc_%d' % i for i in range(n_ben)]
    neu_cols = ['sc_%d' % i for i in range(n_ben, n_ben+n_neu)]
    del_cols = ['sc_%d' % i for i in range(n_ben+n_neu, n_ben+n_neu+n_del)]
    tra_cols = ['tc_%d' % i for i in range(n_tra)]

    colors     = [C_BEN, C_NEU, C_DEL]
    tags_tv    = ['beneficial', 'neutral', 'deleterious', 'trait']
    cols_tv    = [ben_cols, neu_cols, del_cols, tra_cols]
    s_true_loc = [s_ben, s_neu, s_del, s_t_a]

    # PLOT FIGURE
    ## set up figure grid
    w     = DOUBLE_COLUMN
    goldh = w/1.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tra1 = dict(left=0.10, right=0.48, bottom=0.72, top=0.98)
    box_tra2 = dict(left=0.58, right=0.96, bottom=0.72, top=0.98)
    box_coe1 = dict(left=0.10, right=0.48, bottom=0.37, top=0.63)
    box_coe2 = dict(left=0.58, right=0.96, bottom=0.42, top=0.63)
    box_dis1 = dict(left=0.10, right=0.48, bottom=0.07, top=0.30)
    box_dis2 = dict(left=0.58, right=0.96, bottom=0.07, top=0.30)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_coe1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe1)
    gs_coe2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe2)
    gs_dis1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_dis1)
    gs_dis2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_dis2)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_coe1 = plt.subplot(gs_coe1[0, 0])
    ax_coe2 = plt.subplot(gs_coe2[0, 0])
    ax_dis1 = plt.subplot(gs_dis1[0, 0])
    ax_dis2 = plt.subplot(gs_dis2[0, 0])

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

    traj_legend_x  = 265
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
    pprops = { 'xlim':        [0, 1000],
               'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [   0,0.16],
               'yticks':      [   0,0.08,0.16],
               'yminorticks': [0.02,0.04,0.06,0.10,0.12,0.14],
               'yticklabels': [   0,   8,  16],
               'xlabel':      'Generation (days)',
               'ylabel':      'Inferred trait\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': ':', 'alpha': 1 },
               'theme':       'open'}

    times       = np.linspace(0,n_gen,int(n_gen+1))

    n_coe2    = []
    c_coe2    = []
    c_coe2_lt = []
    offset    = n_ben+n_neu+n_del
    for i in range(n_tra):
        c_coe2.append(C_group[i])
        c_coe2_lt.append(C_group[i])

    mp.line(ax=ax_coe2, x=[times], y=[s_tra], colors=[BKCOLOR], **pprops)
    pprops['plotprops']['ls'] = '-'
    for i in range(n_tra):
        xdat = [0,1000]
        ydat = [s_inf[offset+i],s_inf[offset+i]]
        yerr = np.sqrt(ds[offset+i][offset+i])
        mp.line(ax=ax_coe2, x=[xdat], y=[ydat], colors=[c_coe2[i]], **pprops)
    
    # legend
    coef_legend_x  = 600
    coef_legend_y  = 0.145
    coef_legend_dy = -0.025
    coef_legend_t  = []
    for i in range(len(trait_site)):
        coef_legend_t.append('Trait %d'%(i+1))

    for k in range(len(trait_site)):
        yy = coef_legend_y + k * coef_legend_dy
        mp.line(ax=ax_coe2, x=[[coef_legend_x-100, coef_legend_x-50]], y=[[yy, yy]], colors=[c_coe2[k]], **pprops)
        ax_coe2.text(coef_legend_x, yy, coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  coef_legend_y + len(trait_site) * coef_legend_dy
    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_coe2, x=[[coef_legend_x-100, coef_legend_x-50]], y=[[yy, yy]], colors=[BKCOLOR], **pprops)
    ax_coe2.text(coef_legend_x, yy, 'True coefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe2.text(box_coe2['left']+dx, box_coe2['top']+0.04, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # e -- distribution of inferred selection coefficients for individual loci
    dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)

    pprops = { 'xlim'     : [ -0.06,  0.06],
            'xticks'      : [ -0.06, -0.04, -0.02,    0.,  0.02,  0.04,  0.06],
            'xticklabels' : [    -6,    -4,    -2,     0,     2,     4,     6],
            'ylim'        : [0., 0.10],
            'yticks'      : [0., 0.05, 0.10],
            'xlabel'      : 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
            'ylabel'      : 'Frequency',
            'bins'        : np.arange(-0.06, 0.06, 0.001),
            'combine'     : True,
            'plotprops'   : histprops,
            'axoffset'    : 0.1,
            'theme'       : 'boxed' }

    for i in range(len(tags_tv)-1):
        x = [np.array(df[cols_tv[i]]).flatten()]
        ax_dis1.text(s_true_loc[i], 0.106, r'$s_{%s}$' % (tags_tv[i]), color=colors[i], **tprops)
        ax_dis1.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags_tv)-2: mp.hist(             ax=ax_dis1, x=x, colors=[colors[i]], **pprops)
        else:                mp.plot(type='hist', ax=ax_dis1, x=x, colors=[colors[i]], **pprops)

    ax_dis1.text(  box_dis1['left']+dx,  box_dis1['top']+0.04, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # f -- distribution of inferred selection coefficients for binary traits
    pprops = { 'xlim'        : [   0, 0.12],
               'xticks'      : [   0, 0.04, 0.08, 0.12],
               'xticklabels' : [   0,    4,    8,   12],
               'ylim'        : [0., 0.15],
               'yticks'      : [0., 0.05, 0.10, 0.15],
               'xlabel'      : 'Inferred trait coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(0, 0.12, 0.003),
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    ax_dis2.text(s_true_loc[3], 0.159, r'$s_{%s}$' % (tags_tv[3]), color=BKCOLOR, **tprops)
    ax_dis2.axvline(x=s_true_loc[3], **dashlineprops)
    
    for i in range(n_tra):
        x = [np.array(df[cols_tv[3][i]]).flatten()]
        mp.plot(type='hist', ax=ax_dis2, x=x, colors=[c_coe2[i]], **pprops)

    ax_dis2.text( box_dis2['left']+dx, box_dis2['top']+0.04, 'f'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/sufig-sim-tv.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as sufig-sim-tv.pdf')


def plot_different_r(**pdata):

    # unpack data

    tags    = pdata['tags']
    r_rates = pdata['r_rates']
    lim_sc  = pdata['lim_sc']
    lim_tc  = pdata['lim_tc']
    tick_sc = pdata['tick_sc']
    tick_tc = pdata['tick_tc']

    var_sc = [[] for _ in range(len(r_rates))]
    var_tc = [[] for _ in range(len(r_rates))]
    
    for tag in tags:
        df_poly = pd.read_csv('%s/rx/different r/%s-analyze-R.csv' % (HIV_DIR, tag), comment='#', memory_map=True)
        for df_iter, df_entry in df_poly.iterrows():
            if df_entry.sc_MPL != 0 and df_entry.nucleotide != '-':
                for j in range(len(r_rates)):
                    var_sc[j].append(df_entry['sc_%d' %j])

        df_trait    = pd.read_csv('%s/rx/different r/escape_group-R-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        var_tc_0    = 0
        for df_iter, df_entry in df_trait.iterrows():
            if pd.isna(df_entry.tc_MPL) ==  False:
                if df_entry.tc_MPL != var_tc_0:
                    for j in range(len(r_rates)):
                        var_tc[j].append(df_entry['tc_%d' %j])
                var_tc_0 = df_entry.tc_MPL

    var_sc_array = np.array(var_sc)
    var_tc_array = np.array(var_tc)

    sc_var = var_sc_array.T
    tc_var = var_tc_array.T

    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/3.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_sc = dict(left=0.10, right=0.38, bottom=0.15, top=0.90)
    box_tc = dict(left=0.48, right=0.76, bottom=0.15, top=0.90)
    box_label = dict(left=0.80, right=0.96, bottom=0.15, top=0.92)

    gs_sc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)
    gs_tc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_label = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_label)

    ax_sc = plt.subplot(gs_sc[0, 0])
    ax_tc = plt.subplot(gs_tc[0, 0])
    ax_label = plt.subplot(gs_label[0, 0])

    dx = -0.05
    dy =  0.04
    
    colors = []
    lights = np.linspace(0.2,0.8,len(r_rates)-2)
    for i in range(len(r_rates)-2):
        colors.append(hls_to_rgb(0.58, lights[i], 0.60))

    ## a -- inferred selection coefficients with VS. without binary trait term
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : '-', 'alpha' : 0.8}

    pprops = { 'xlim':         lim_sc,
               'ylim':         lim_sc,
               'xticks':       tick_sc,
               'yticks':       tick_sc,
               'xticklabels':  [int(i*100) for i in tick_sc],
               'yticklabels':  [int(i*100) for i in tick_sc],
               'xlabel':       'Inferred selection coefficients for individual locus, '+ r'$\hat{s}$' + ' (%)',
               'ylabel':       'Inferred selection coefficients with \na different recombination rate, '+ r'$\hat{s}$' + ' (%)',
               'theme':        'boxed'}
    
    for i in range(len(sc_var)):

        for j in range(len(r_rates)-2):
            x_i = [sc_var[i][j],   sc_var[i][j+1]]
            y_i = [sc_var[i][j+1], sc_var[i][j+2]]

            if i == len(sc_var)-1 and j == len(r_rates)-3:
                mp.plot(type='line', ax=ax_sc, x=[x_i], y=[y_i], colors=[colors[j]],plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax_sc, x=[x_i], y=[y_i], colors=[colors[j]],plotprops=lineprops, **pprops)

    ax_sc.text(box_sc['left']+dx, box_sc['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- trajectory plot

    pprops = { 'xlim':         lim_tc,
               'ylim':         lim_tc,
               'xticks':       tick_tc,
               'yticks':       tick_tc,
               'xticklabels':  [int(i*100) for i in tick_tc],
               'yticklabels':  [int(i*100) for i in tick_tc],
               'xlabel':       'Inferred selection coefficients for binary trait, '+ r'$\hat{s}$' + ' (%)',
               'ylabel':       'Inferred selection coefficients with \na different recombination rate, '+ r'$\hat{s}$' + ' (%)',
               'theme':        'boxed'}
    
    for i in range(len(tc_var)):
        for j in range(len(r_rates)-2):
            x_i = [tc_var[i][j],   tc_var[i][j+1]]
            y_i = [tc_var[i][j+1], tc_var[i][j+2]]

            if i == len(tc_var)-1 and j == len(r_rates)-3:
                mp.plot(type='line', ax=ax_tc, x=[x_i], y=[y_i], colors=[colors[j]],plotprops=lineprops, **pprops)
            else:
                mp.line(             ax=ax_tc, x=[x_i], y=[y_i], colors=[colors[j]],plotprops=lineprops, **pprops)
        
    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # c -- label
    pprops = { 'xlim':         [ 0, 10],
               'ylim':         [ 0, 10],
               'xticks':       [],
               'yticks':       [],
               'xlabel':       '',
               'ylabel':       '',
               'theme':       'open',
               'hide':        ['bottom','left']}

    ax_label.text(1, 9, 'start', ha='left', va='center', **DEF_LABELPROPS)
    ax_label.text(1, 9 - 0.65*(len(r_rates)-1), 'end', ha='left', va='center', **DEF_LABELPROPS)
    ax_label.text(4, 9, r'$r = %.2e$' % r_rates[0], ha='left', va='center', **DEF_LABELPROPS)
    ax_label.text(4, 9 - 0.65*(len(r_rates)-1), r'$r = %.2e$' % r_rates[-1], ha='left', va='center', **DEF_LABELPROPS)

    for i in range(len(r_rates)-2):
        yy_i = 9.0 - 0.65 * (i + 1) 

        ax_label.text(4, yy_i, r'$r = %.2e$' % r_rates[i+1], ha='left', va='center', **DEF_LABELPROPS)

        if i == len(r_rates) - 3:
            mp.plot(type='line', ax=ax_label, x=[[1,3]], y=[[yy_i, yy_i]], colors=[colors[i]],plotprops=lineprops, **pprops)
        else:
            mp.line(             ax=ax_label, x=[[1,3]], y=[[yy_i, yy_i]], colors=[colors[i]],plotprops=lineprops, **pprops)

    # SAVE FIGURE
    plt.savefig('%s/rxfig-different-r.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as rxfig-different-r.pdf')


def plot_different_r_tc(**pdata):

    # unpack data

    tags    = pdata['tags']
    r_rates = pdata['r_rates']   # all recombination rates
    index   = pdata['index']     # index for normal recombination rate 
    lim_tc_x  = pdata['lim_tc_x']    # limit for trait coefficient with a normal recombination rate
    lim_tc_y  = pdata['lim_tc_y']    # limit for trait coefficient with all recombination rates
    tick_tc_x = pdata['tick_tc_x']   # ticks for trait coefficient with a normal recombination rate
    tick_tc_y = pdata['tick_tc_y']   # ticks for trait coefficient with all recombination rates

    var_tc = [[] for _ in range(len(r_rates))]
    
    for tag in tags:
        df_trait    = pd.read_csv('%s/rx/different r/escape_group-R-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        var_tc_0    = 0
        for df_iter, df_entry in df_trait.iterrows():
            if pd.isna(df_entry.tc_MPL) ==  False:
                if df_entry.tc_MPL != var_tc_0:
                    for j in range(len(r_rates)):
                        var_tc[j].append(df_entry['tc_%d' %j])
                var_tc_0 = df_entry.tc_MPL

    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/3.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tc = dict(left=0.10, right=0.66, bottom=0.15, top=0.90)
    box_label = dict(left=0.70, right=0.90, bottom=0.15, top=0.90)

    gs_tc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_label = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_label)

    ax_tc = plt.subplot(gs_tc[0, 0])
    ax_label = plt.subplot(gs_label[0, 0])

    dx = -0.05
    dy =  0.04

    colors = []
    lights = np.linspace(0.2,0.8,len(r_rates))
    for i in range(len(r_rates)):
        colors.append(hls_to_rgb(0.58, lights[i], 0.60))

    ## a -- distribution of inferred trait coefficients with different recombination rates

    scatterprops = dict(lw=0, s=SMALLSIZEDOT*0.6, marker='o', alpha=0.6, clip_on=False)
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : ':', 'alpha' : 0.8}
    
    pprops = { 'xlim':         lim_tc_x,
               'ylim':         lim_tc_y,
               'xticks':       tick_tc_x,
               'yticks':       tick_tc_y,
               'xticklabels':  [int(i*100) for i in tick_tc_x],
               'yticklabels':  [int(i*100) for i in tick_tc_y],
               'xlabel':       'Inferred selection coefficients for binary trait, '+ r'$\hat{s}$' + ' (%)',
               'ylabel':       'Inferred selection coefficients with \na different recombination rate, '+ r'$\hat{s}$' + ' (%)',
               'theme':        'boxed'}
    
    mp.plot(type='line', ax=ax_tc, x=[lim_tc_x], y=[lim_tc_x], colors=[C_NEU],plotprops=lineprops, **pprops)

    for i in range(len(r_rates)):
        if i == len(r_rates)-1:
            mp.plot(type='scatter', ax=ax_tc, x=[var_tc[index]], y=[var_tc[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:
            mp.scatter(             ax=ax_tc, x=[var_tc[index]], y=[var_tc[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    # b -- label
    pprops = { 'xlim':         [ 0, 10],
               'ylim':         [ 0, 10],
               'xticks':       [],
               'yticks':       [],
               'xlabel':       '',
               'ylabel':       '',
               'theme':       'open',
               'hide':        ['bottom','left']}

    for i in range(len(r_rates)):
        yy_i = 9.5 - 0.65 * i

        ax_label.text(4, yy_i, r'$r = %.2e$' % r_rates[i], ha='left', va='center', **DEF_LABELPROPS)

        if i == (len(r_rates) - 1):
            mp.plot(type='scatter', ax=ax_label, x=[[2]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:
            mp.scatter(             ax=ax_label, x=[[2]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    # SAVE FIGURE
    plt.savefig('%s/rxfig-different-r-tc.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as rxfig-different-r-tc.pdf')
 

def plot_virus_load(**pdata):

    # unpack data

    tags    = pdata['tags']
    
    def get_times(tag):
        seq      = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
        times = []
        for i in range(len(seq)):
            times.append(seq[i][0])
        return np.unique(times)

    times_all = []
    vl_all    = []

    times_sample_all = [[] for _ in range(len(tags))]
    vl_sample_all    = [[] for _ in range(len(tags))]

    for i in range(len(tags)):
        tag = tags[i]
        ppt = tag[:9]

        df_vl_raw    = pd.read_csv('%s/virus load/%s.csv' %(HIV_DIR,ppt), header=None)
        df_vl_raw.columns = ['time', 'virus_load']
        df_vl = df_vl_raw.sort_values(by='time', ascending=True)

        sample_times = get_times(tag)

        times = [int(i) for i in df_vl['time'].values]
        virus_load = [np.power(10, i) for i in df_vl['virus_load'].values]

        time_min = np.min([int(times[0]),sample_times[0]])
        time_max = np.max([int(times[-1]),sample_times[-1]])
        whole_time = np.linspace(time_min,time_max,int(time_max-time_min+1))

        interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear',fill_value=(virus_load[0], virus_load[-1]), bounds_error=False)
        AllVL = interpolation(times, virus_load)(whole_time)

        for t in sample_times:
            if t <= whole_time[-1] and t >= whole_time[0]:
                index = list(whole_time).index(t)
                vl_sample_all[i].append(AllVL[index])
                times_sample_all[i].append(t)   # remove the time points outside the range

        times_all.append(whole_time)
        vl_all.append(AllVL)


    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tc = dict(left=0.10, right=0.66, bottom=0.15, top=0.90)
    box_label = dict(left=0.70, right=0.90, bottom=0.15, top=0.90)

    gs_tc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_label = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_label)

    ax_tc = plt.subplot(gs_tc[0, 0])
    ax_label = plt.subplot(gs_label[0, 0])

    dx = -0.05
    dy =  0.04

    # colors = []
    # lights = np.linspace(0.2,0.8,len(tags))
    # for i in range(len(tags)):
    #     colors.append(hls_to_rgb(0.58, lights[i], 0.60))

    colors = sns.color_palette("husl", len(tags))
    ## a -- distribution of inferred trait coefficients with different recombination rates

    scatterprops = dict(lw=0, s=SMALLSIZEDOT*0.6, marker='o', alpha=1.0, clip_on=False)
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : ':', 'alpha' : 0.6}
    fillprops = { 'lw': 0, 'alpha': 0.15, 'interpolate': True, 'step': 'mid' }

    pprops = { 'xlim':         [0,800],
               'ylim':         [1,7],
               'xticks':       [0,200,400,600,800],
               'yticks':       [1,2,3,4,5,6,7],
               'yticklabels':  [r'$10^{1}$',r'$10^{2}$',r'$10^{3}$',r'$10^{4}$',r'$10^{5}$',r'$10^{6}$',r'$10^{7}$'],
               'xlabel':       'Time (days) ',
               'ylabel':       'viral load (copies/ml)',
               'theme':        'boxed'}

    for i in range(len(tags)):
        vl_i = np.log10(vl_all[i])
        mp.line(ax=ax_tc, x=[times_all[i]], y=[vl_i], colors=[colors[i]],plotprops=lineprops, **pprops)
    
    for i in range(len(tags)):
        vl_sample_i = np.log10(vl_sample_all[i])
        if i == len(tags)-1:
            mp.plot(type='scatter', ax=ax_tc, x=[times_sample_all[i]], y=[vl_sample_i], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:
            mp.scatter(             ax=ax_tc, x=[times_sample_all[i]], y=[vl_sample_i], colors=[colors[i]],plotprops=scatterprops, **pprops)
    
    # x_tc = [0,800,800,0]
    # y_tc = [np.log10(100),np.log10(26800),np.log10(82000),np.log10(165500),np.log10(10000000)]
    # c_tc = ['#0000FF','#808080','#FFFF00','#FFA500']
    # for i in range(len(y_tc)-2):
    #     y_tc_i = [y_tc[i],y_tc[i],y_tc[i+1],y_tc[i+1]]
    #     mp.fill(ax=ax_tc, x=[x_tc], y=[y_tc_i], colors=[c_tc[i]],plotprops=fillprops, **pprops)

    # y_tc_i = [y_tc[-2],y_tc[-2],y_tc[-1],y_tc[-1]]
    # mp.plot(type='fill',ax=ax_tc, x=[x_tc], y=[y_tc_i], colors=[c_tc[-1]],plotprops=fillprops, **pprops)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- label
    pprops = { 'xlim':         [ 0, 10],
               'ylim':         [ 0, 10],
               'xticks':       [],
               'yticks':       [],
               'xlabel':       '',
               'ylabel':       '',
               'theme':       'open',
               'hide':        ['bottom','left']}

    for i in range(len(tags)):
        yy_i = 9.5 - 0.55 * i

        ax_label.text(4, yy_i, '%s' % tags[i], ha='left', va='center', **DEF_LABELPROPS)

        if i == (len(tags) - 1):
            mp.plot(type='scatter', ax=ax_label, x=[[2]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:
            mp.scatter(             ax=ax_label, x=[[2]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    # SAVE FIGURE
    plt.savefig('%s/rxfig-vl.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('Figure saved as rxfig-vl.pdf')
 
def plot_sc_with_VL_r(**pdata):

    # unpack data

    tags    = pdata['tags']
    
    var_sc     = [[] for _ in range(len(tags))]
    var_sc_new = [[] for _ in range(len(tags))]
    var_tc     = [[] for _ in range(len(tags))]
    var_tc_new = [[] for _ in range(len(tags))]

    # process stored data
    for i in range(len(tags)):
        tag = tags[i]
        df_sc = pd.read_csv('%s/rx/constant r/%s-analyze.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

        tc_tag = -0.1
        for ii in range(len(df_sc)):
            if df_sc.iloc[ii].sc_MPL != 0 and df_sc.iloc[ii].nucleotide != '-':
                var_sc[i].append(df_sc.loc[ii,'sc_cR'])
                var_sc_new[i].append(df_sc.loc[ii,'sc_MPL'])

            if pd.isna(df_sc.loc[ii,'tc_MPL']) == False and df_sc.loc[ii,'tc_MPL'] != tc_tag:
                var_tc[i].append(df_sc.loc[ii,'tc_cR'])
                var_tc_new[i].append(df_sc.loc[ii,'tc_MPL'])
                tc_tag = df_sc.loc[ii,'tc_MPL']

    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/1.6
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_sc = dict(left=0.07, right=0.57, bottom=0.10, top=0.90)
    box_tc = dict(left=0.65, right=0.95, bottom=0.10, top=0.58)
    box_la = dict(left=0.62, right=0.95, bottom=0.60, top=0.90)
    
    gs_sc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)
    gs_tc = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_la = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_la)

    ax_sc = plt.subplot(gs_sc[0, 0])
    ax_tc = plt.subplot(gs_tc[0, 0])
    ax_la = plt.subplot(gs_la[0, 0])

    dx = -0.05
    dy =  0.04

    dx = -0.05
    dy =  0.04

    colors = sns.color_palette("husl", len(tags))

    ## a -- inferred selection coefficients with constant r VS. VL-dependent r

    s_min = -0.06
    s_max =  0.08
    scatterprops = dict(lw=0, s=SMALLSIZEDOT*0.6, marker='o', alpha=0.6)
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : ':', 'alpha' : 0.6}

    pprops = { 'xlim':         [ s_min, s_max],
               'ylim':         [ s_min, s_max],
               'xticks':       [ -0.06, -0.04, -0.02,   0,  0.02,  0.04,  0.06,  0.08],
               'yticks':       [ -0.06, -0.04, -0.02,   0,  0.02,  0.04,  0.06,  0.08],
               'xticklabels':  [ -6, -4, -2, 0, 2, 4, 6, 8],
               'yticklabels':  [ -6, -4, -2, 0, 2, 4, 6, 8],
               'xlabel':       'Inferred selection coefficients with constant r',
               'ylabel':       'Inferred selection coefficients with VL-dependent r',
               'theme':        'boxed'}

    mp.line(ax=ax_sc, x=[[s_min, s_max]], y=[[s_min,s_max]], colors=[C_NEU],plotprops=lineprops, **pprops)

    
    for i in range(len(var_sc)):
        if i == len(var_sc)-1:
            mp.plot(type='scatter', ax=ax_sc, x=[var_sc[i]], y=[var_sc_new[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:  
            mp.scatter(             ax=ax_sc, x=[var_sc[i]], y=[var_sc_new[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    ## b -- inferred selection coefficients with VS. without binary trait term

    s_min = -0.05
    s_max =  0.30
    scatterprops = dict(lw=0, s=SMALLSIZEDOT*0.6, marker='o', alpha=0.6)
    lineprops   = { 'lw' : SIZELINE, 'linestyle' : ':', 'alpha' : 0.6}

    pprops = { 'xlim':         [ s_min, s_max],
               'ylim':         [ s_min, s_max],
               'xticks':       [ -0.05,   0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
               'yticks':       [ -0.05,   0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
               'xticklabels':  [ -5,  0,  5,  10,  15,  20, 25, 30],
               'yticklabels':  [ -5,  0,  5,  10,  15,  20, 25, 30],
               'xlabel':       'Inferred trait coefficients with constant r',
               'ylabel':       'Inferred trait coefficients with VL-dependent r',
               'theme':        'boxed'}

    mp.line(ax=ax_tc, x=[[s_min, s_max]], y=[[s_min,s_max]], colors=[C_NEU],plotprops=lineprops, **pprops)
    
    for i in range(len(var_tc)):
        if i == len(var_tc)-1:
            mp.plot(type='scatter', ax=ax_tc, x=[var_tc[i]], y=[var_tc_new[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:  
            mp.scatter(             ax=ax_tc, x=[var_tc[i]], y=[var_tc_new[i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    ## c -- label
    pprops = { 'xlim':         [ 0, 10],
               'ylim':         [ 0, 10],
               'xticks':       [],
               'yticks':       [],
               'xlabel':       '',
               'ylabel':       '',
               'theme':       'open',
               'hide':        ['bottom','left']}

    for i in range(len(tags)):
        
        if i%3 == 0:
            yy_i = 8 - 1.2 * i / 3
            xx_i = 1
        elif i%3 == 1:
            yy_i = 8 - 1.2 * (i - 1) / 3
            xx_i = 4
        else:
            yy_i = 8 - 1.2 * (i - 2) / 3
            xx_i = 7

        if i == (len(tags) - 1):
            ax_la.text(xx_i, yy_i, '%s' % tags[i], ha='left', va='center', **DEF_LABELPROPS)
            mp.plot(type='scatter', ax=ax_la, x=[[xx_i-0.3]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)
        else:
            ax_la.text(xx_i, yy_i, '%s' % tags[i], ha='left', va='center', **DEF_LABELPROPS)
            mp.scatter(ax=ax_la, x=[[xx_i-0.3]], y=[[yy_i]], colors=[colors[i]],plotprops=scatterprops, **pprops)

    # SAVE FIGURE
    plt.savefig('%s/rxfig-HIV-r-VL.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.show()
    print('Figure saved as rxfig-HIV-r-VL.pdf')

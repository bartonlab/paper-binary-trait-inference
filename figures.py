#############  PACKAGES  #############
import numpy as np

import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import seaborn as sns

from colorsys import hls_to_rgb

import mplot as mp

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
def plot_figure_1(**pdata):
    """
    Example evolutionary trajectory for a 50-site system and inferred selection coefficients
    and trait coefficients, together with aggregate properties for different levels of sampling..
    """

    # unpack passed data

    n_gen   = pdata['n_gen']
    dg      = pdata['dg']
    N       = pdata['N']
    xfile   = pdata['xfile']
    name    = pdata['name']
    po_site = pdata['po_site']

    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_pol = pdata['n_pol']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']

    r_seed = pdata['r_seed']
    np.random.seed(r_seed)

    # load and process data files

    data  = np.loadtxt('%s/%s.dat' % (SIM_DIR, xfile))
    times = np.unique(data.T[0]) # get all time point

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
        t_fre     = [];
        for i in range(len(po_site)):
            t_data_i  = t_num*0;
            for j in range(len(po_site[i])):
                site = po_site[i][j];
                t_data_i += data[idx].T[site+2]
            for k in range(len(t_data_i)):
                if t_data_i[k] != 0:
                    t_data_i[k] = 1;
            t_freq_i = np.einsum('i,i', t_num, t_data_i) / float(np.sum(t_num))
            t_fre.append(t_freq_i)
        y.append(t_fre)
    y = np.array(y).T # get trait frequency

    s_true  = [s_ben for i in range(n_ben)] + [0 for i in range(n_neu)]
    s_true += [s_del for i in range(n_del)] + [s_pol for i in range(n_pol)]
    s_inf   = np.loadtxt('%s/sc_example.dat' %SIM_DIR)
    cov     = np.loadtxt('%s/example_covariance.dat' %SIM_DIR)
    ds      = np.linalg.inv(cov) / N

    # PLOT FIGURE

    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tra1 = dict(left=0.10, right=0.45, bottom=0.50, top=0.90)
    box_tra2 = dict(left=0.55, right=0.90, bottom=0.50, top=0.90)
    box_coe1 = dict(left=0.10, right=0.45, bottom=0.10, top=0.40)
    box_coe2 = dict(left=0.55, right=0.90, bottom=0.10, top=0.40)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_coe1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe1)
    gs_coe2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_coe2)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_coe1 = plt.subplot(gs_coe1[0, 0])
    ax_coe2 = plt.subplot(gs_coe2[0, 0])


    dx = -0.03
    dy =  0.02

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
               'ylabel':      'trait\nfrequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-' , 'alpha': 1},
               'axoffset':    0.1,
               'theme':       'open' }

    var_c = sns.husl_palette(len(po_site))

    traj_legend_x  =  700
    traj_legend_y  = [0.65, 0.40]
    traj_legend_t  = ['trait\nfrequency','individual \nallele frequency']

    for k in range(len(traj_legend_y)):
        x1 = traj_legend_x-90
        x2 = traj_legend_x-10
        y1 = traj_legend_y[0] + (0.5-k) * 0.03
        y2 = traj_legend_y[1] + (0.5-k) * 0.03
        pprops['plotprops']['alpha'] = 1
        mp.line(ax=ax_tra2, x=[[x1, x2]], y=[[y1, y1]], colors=[var_c[k]], **pprops)
        pprops['plotprops']['alpha'] = 0.4
        mp.line(ax=ax_tra2, x=[[x1, x2]], y=[[y2, y2]], colors=[var_c[k]], **pprops)
        ax_tra2.text(traj_legend_x, traj_legend_y[k], traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    xdat = [range(0, n_gen, dg)]
    for i in range(len(po_site)):
        for j in range(len(po_site[i])):
            site = po_site[i][j]
            ydat = [k for k in x[site:site+1]]
            pprops['plotprops']['alpha'] = 0.6
            mp.line(ax=ax_tra2, x=xdat, y=ydat, colors=[var_c[i]], **pprops)
        if i > 0:
            ydat = [k for k in y[i:i+1]]
            pprops['plotprops']['alpha'] = 1
            mp.line(ax=ax_tra2, x=xdat, y=ydat, colors = [var_c[i]], **pprops)

    ydat = [k for k in y[0:1]]
    pprops['plotprops']['alpha'] = 1
    mp.plot(type='line',ax=ax_tra2, x=xdat, y=ydat, colors = [var_c[0]], **pprops)

    ax_tra2.text(box_tra2['left']+dx, box_tra1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c -- individual beneficial/neutral/deleterious selection coefficients

    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

    pprops = { 'xlim':        [ -0.3,    4],
               'ylim':        [-0.04, 0.04],
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
        mp.line(ax=ax_coe1, x=[[k-0.35, k+0.35]], y=[[s_true[offset[k]], s_true[offset[k]]]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
        plotprops = DEF_ERRORPROPS.copy()
        plotprops['alpha'] = 1
        for i in range(n_coe1[k]):
            xdat = [k + np.random.normal(0, 0.08)]
            ydat = [s_inf[offset[k]+i]]
            yerr = np.sqrt(ds[offset[k]+i][offset[k]+i])
            if i==n_coe1[k]-1 and k==len(n_coe1)-1:
                mp.plot(type='error', ax=ax_coe1, x=[xdat], y=[ydat], yerr=[yerr], edgecolor=[c_coe1[k]], facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)
            else:
                mp.error(ax=ax_coe1, x=[xdat], y=[ydat], yerr=[yerr], edgecolor=[c_coe1[k]], facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)

    coef_legend_x  =  2.8
    coef_legend_d  = -0.15
    coef_legend_dy = -0.011
    coef_legend_y  = [0.02, 0.02 + coef_legend_dy, 0.02 + (2*coef_legend_dy)]
    coef_legend_t  = ['Beneficial', 'Neutral', 'Deleterious']
    for k in range(len(coef_legend_y)):
        mp.error(ax=ax_coe1, x=[[coef_legend_x+coef_legend_d]], y=[[coef_legend_y[k]]], edgecolor=[c_coe1[k]], facecolor=[c_coe1_lt[k]], plotprops=plotprops, **pprops)
        ax_coe1.text(coef_legend_x, coef_legend_y[k], coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  0.02 + 3.5 * coef_legend_dy
    mp.line(ax=ax_coe1, x=[[coef_legend_x-0.21, coef_legend_x-0.09]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_coe1.text(coef_legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe1.text(box_coe1['left']+dx, box_coe1['top']+dy*1.8, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c2 -- trait coefficients

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
    for i in range(n_pol):
        c_coe2.append(var_c[i])
        c_coe2_lt.append(var_c[i])

    mp.line(ax=ax_coe2, x=[[0.15, 0.85]], y=[[s_true[-1], s_true[-1]]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['alpha'] = 1
    for i in range(n_pol):
        xdat = [np.random.normal(0.5, 0.08)]
        ydat = [s_inf[offset+i]]
        yerr = np.sqrt(ds[offset+i][offset+i])
        if i==n_pol-1:
            mp.plot(type='error', ax=ax_coe2, x=[xdat], y=[ydat], yerr=[yerr], edgecolor=[c_coe2[i]], facecolor=[c_coe2_lt[i]], plotprops=plotprops, **pprops)
        else:
            mp.error(ax=ax_coe2, x=[xdat], y=[ydat], yerr=[yerr], edgecolor=[c_coe2[i]], facecolor=[c_coe2_lt[i]], plotprops=plotprops, **pprops)

    coef_legend_x  =  1.8
    coef_legend_dy = -0.025
    coef_legend_y  = []
    coef_legend_t  = []
    for i in range(len(po_site)):
        coef_legend_y.append(0.135 + i * coef_legend_dy)
        coef_legend_t.append('trait-%d'%(i+1))
    for k in range(len(coef_legend_y)):
        mp.error(ax=ax_coe2, x=[[coef_legend_x+coef_legend_d]], y=[[coef_legend_y[k]]], edgecolor=[c_coe2[k]], facecolor=[c_coe2_lt[k]], plotprops=plotprops, **pprops)
        ax_coe2.text(coef_legend_x, coef_legend_y[k], coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  0.135 +(0.5+n_pol) * coef_legend_dy
    mp.line(ax=ax_coe2, x=[[coef_legend_x-0.21, coef_legend_x-0.09]], y=[[yy, yy]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_coe2.text(coef_legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_coe2.text(box_coe2['left']+dx, box_coe2['top']+dy*1.8, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig1.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

    print('figure 1 done.')

def plot_figure_2(**pdata):
    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data

    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_pol = pdata['n_pol']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']

    # PLOT FIGURE
    ## set up figure grid

    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_se   = dict(left=0.10, right=0.64, bottom=0.65, top=0.95)
    box_po   = dict(left=0.69, right=0.92, bottom=0.65, top=0.95)
    box_aur1 = dict(left=0.10, right=0.32, bottom=0.07, top=0.50)
    box_aur2 = dict(left=0.40, right=0.62, bottom=0.07, top=0.50)
    box_erro = dict(left=0.70, right=0.92, bottom=0.07, top=0.50)

    gs_se   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
    gs_po   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_po)
    gs_aur1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_aur1)
    gs_aur2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_aur2)
    gs_erro = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_erro)

    ax_se  = plt.subplot(gs_se[0, 0])
    ax_po  = plt.subplot(gs_po[0, 0])
    ax_aur1 = plt.subplot(gs_aur1[0, 0])
    ax_aur2 = plt.subplot(gs_aur2[0, 0])
    ax_erro = plt.subplot(gs_erro[0, 0])

    dx = -0.04
    dy =  0.03

    ### plot histogram

    df_all = pd.read_csv('%s/mpl_collected.csv' % SIM_DIR, memory_map=True)
    df     = df_all[(df_all.ns==1000) & (df_all.delta_t==1)]

    ben_cols = ['sc_%d' % i for i in range(n_ben)]
    neu_cols = ['sc_%d' % i for i in range(n_ben, n_ben+n_neu)]
    del_cols = ['sc_%d' % i for i in range(n_ben+n_neu, n_ben+n_neu+n_del)]
    pol_cols = ['pc_%d' % i for i in range(n_pol)]

    colors     = [C_BEN, C_NEU, C_DEL]
    tags       = ['beneficial', 'neutral', 'deleterious','trait']
    cols       = [ben_cols, neu_cols, del_cols, pol_cols]
    s_true_loc = [s_ben, s_neu, s_del,s_pol]

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
        ax_se.text(s_true_loc[i], 0.11, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
        ax_se.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_se, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_se, x=x, colors=[colors[i]], **pprops)

    ax_se.text(  box_se['left']+dx,  box_se['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- trait part
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [   0, 0.12],
               'xticks'      : [   0, 0.04, 0.08, 0.10, 0.12],
               'xticklabels' : [   0,    4,    8,   10,   12],
               'ylim'        : [0., 0.20],
               'yticks'      : [0., 0.1, 0.2],
               'xlabel'      : 'Inferred trait coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(0, 0.12, 0.003),
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    c_po   = sns.husl_palette(1)

    x = [np.array(df[cols[3]]).flatten()]
    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_po.text(s_true_loc[3], 0.212, r'$s_{%s}$' % (tags[3]), color=c_po[0], **tprops)
    ax_po.axvline(x=s_true_loc[3], **dashlineprops)
    mp.plot(type='hist', ax=ax_po, x=x, colors=[c_po[0]], **pprops)

    ax_po.text( box_po['left']+dx,  box_po['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c,d  -- AUCs for inferring beneficial/deleterious mutations and error for trait part

    df   = pd.read_csv('%s/mpl_collected_extended.csv' % SIM_DIR, memory_map=True)

    ns_vals = [10, 20, 30, 40, 50, 80, 100]
    dt_vals = [1, 5, 10, 20, 50]

    AUC_matrix_ben = np.zeros((len(dt_vals), len(ns_vals)))
    AUC_matrix_del = np.zeros((len(dt_vals), len(ns_vals)))
    err_matrix_pol = np.zeros((len(dt_vals), len(ns_vals)))

    for i in range(len(dt_vals)):
        for j in range(len(ns_vals)):
            df_AUC = df[(df.delta_t==dt_vals[i]) & (df.ns==ns_vals[j])]
            AUC_matrix_ben[i, j] = np.mean(df_AUC.AUROC_ben)
            AUC_matrix_del[i, j] = np.mean(df_AUC.AUROC_del)
            err_matrix_pol[i, j] = np.mean(df_AUC.error_pol)

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

    ax_erro.pcolor(err_matrix_pol.T, vmin=0.2, vmax=0.8, cmap='GnBu', alpha=0.75)
    for i in range(len(err_matrix_pol)):
        for j in range(len(err_matrix_pol[0])):
            tc = 'k'
            ax_erro.text(i+0.5, j+0.5, '%.2f' % (err_matrix_pol[i,j]), color=tc, **tprops)
    mp.plot(type='scatter', ax=ax_erro, x=[[-1]], y=[[-1]], colors=[BKCOLOR], **pprops)

    ## outside text labels

    tprops = dict(color=BKCOLOR, ha='center', va='center', family=FONTFAMILY, size=SIZELABEL,
                  clip_on=False, transform=fig.transFigure)

    ax_aur1.text((box_aur1['right']-box_aur1['left'])/2+box_aur1['left'], box_aur1['top']+dy, 'Mean AUROC (beneficial)',  **tprops)
    ax_aur2.text((box_aur2['right']-box_aur2['left'])/2+box_aur2['left'], box_aur2['top']+dy, 'Mean AUROC (deleterious)', **tprops)
    ax_erro.text((box_erro['right']-box_erro['left'])/2+box_erro['left'], box_erro['top']+dy, 'NRMSE (trait)', **tprops)

    ax_aur1.text(box_aur1['left']+dx, box_aur1['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_aur2.text(box_aur2['left']+dx, box_aur2['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_erro.text(box_erro['left']+dx, box_erro['top']+dy, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig2.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 2 done.')

def plot_figure_3(**pdata):
    """
    histogram of selection coefficients and escape coefficients
    """

    # unpack passed data
    tags   = pdata['tags']

    # get all escape coefficients

    pc = []

    for tag in tags:
        df_pc = pd.read_csv('%s/poly/%s.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
        pc_old = df_pc.iloc[0].pc_MPL
        for i in range(len(df_pc)):
            if df_pc.iloc[i].pc_MPL != pc_old:
                pc.append(df_pc.iloc[i].pc_MPL)
            pc_old = df_pc.iloc[i].pc_MPL

    # PLOT FIGURE
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_pc  = dict(left=0.15, right=0.85, bottom=0.15, top=0.85)
    gs_pc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_pc)
    ax_pc   = plt.subplot(gs_pc[0, 0])

    dx = -0.04
    dy =  0.03

    ### plot histogram

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim'        : [ -0.05,  0.30],
               'xticks'      : [ -0.05,     0,  0.05, 0.1 , 0.15,  0.2, 0.25, 0.30],
               'xticklabels' : [    -5,     0,    5,   10,   15,    20,   25,   30],
               'ylim'        : [0., 0.30],
               'yticks'      : [0., 0.15, 0.30],
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.05, 0.30, 0.01),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    c_po = sns.husl_palette(1)
    x    = [np.array(pc)]
    mp.plot(type='hist', ax=ax_pc, x=x, colors=[c_po[0]], **pprops)

    tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
    ax_pc.text( 0.15, 0.32, 'All inferred escape coefficient, ' + r'$\hat{s}$' +'(%)', color=BKCOLOR, **tprops)

    # SAVE FIGURE
    plt.savefig('%s/fig3.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 3 done.')

def plot_figure_4(**pdata):
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

    df_poly = pd.read_csv('%s/poly/%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_tag   = []
    traj_poly = []
    poly_info = {}
    var_pc    = []
    var_traj  = []
    var_color = []
    for i in range(len(epitope)):
        df_esc  = df_poly[(df_poly.epitope==epitope[i])]
        df_row  = df_esc.iloc[0]
        epi_nuc = ''.join(epitope[i])
        p_tag = epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc))
        p_c   = df_esc.iloc[0].pc_MPL
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
        var_pc.append(i[1])
        index_i = poly_key.index(i[0])
        var_color.append(var_c[index_i])
    #
    # poly_key = list(poly_info.keys())
    # for i in poly_info:
    #     var_tag.append(i)
    #     var_pc.append(poly_info[i])
    #     index_i = poly_key.index(i)
    #     var_color.append(var_c[index_i])

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
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Variant frequency\nfor CH%s-%s\n'%(patient[-3:],region),
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
    var_epi = [];
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

    mp.plot(type='bar', ax=ax_coef,x=[bar_x], y=[var_pc], **pprops)

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
    traj_legend_t  = ['escape\nfrequency','individual \nallele frequency']

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

    # SAVE FIGURE
    plt.savefig('%s/fig4.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    # plt.savefig('%s/epi/%s-%s.jpg' % (FIG_DIR,patient[-3:],region), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
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

    df_poly = pd.read_csv('%s/poly/%s.csv' % (HIV_DIR, tag), comment='#', memory_map=True)

    times = [int(i.split('_')[-1]) for i in df_poly.columns if 'f_at_' in i]
    times.sort()

    var_tag   = []
    var_snew  = []
    var_sold  = []
    var_pc    = []
    var_traj  = []
    traj_poly = []
    for i in range(len(epitope)):
        df_esc  = df_poly[(df_poly.epitope==epitope[i])]
        df_row  = df_esc.iloc[0]
        epi_nuc = ''.join(epitope[i])
        var_pc.append(df_esc.iloc[0].pc_MPL)
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

    box_traj = dict(left=0.17, right=0.85, bottom=0.65, top=0.95)
    box_coef = dict(left=0.17, right=0.85, bottom=0.20, top=0.45)

    gs_traj = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_coef = gridspec.GridSpec(1, 2, width_ratios=[5,6], wspace = sspace,     **box_coef)

    ax_traj = plt.subplot(gs_traj[0, 0])
    ax_coef = [plt.subplot(gs_coef[0, i]) for i in range(2)]

    dx = -0.10
    dy =  0.04

    var_c = sns.husl_palette(len(var_traj)+1)

    ## a -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
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
               'ylim':        [ -0.03,  0.18],
               'yticks':      [ -0.03,     0,  0.03, 0.06, 0.09, 0.12, 0.15, 0.18],
               'yticklabels': [    -3,     0,     3,    6,    9,   12,   15,   18],
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

    ax_coef[0].text((box_coef['right']+3*box_coef['left']-sspace)/4, box_coef['top'], 'old', **labelprops)
    ax_coef[0].text(box_coef['left']+dx, box_coef['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    bar_x  = [i+0.5 for i in range(len(var_tag)+1)]
    var_tag.append(r'$EV11$')
    var_snew.append(var_pc[0])
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

    ax_coef[1].text((box_coef['right']*3+box_coef['left']-4*sspace)/4, box_coef['top'], 'new', **labelprops)
    ax_coef[1].text((box_coef['right']+box_coef['left']+sspace)/2+dx/2, box_coef['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # add background

    cBG = '#F5F5F5'
    ddx = 0.01
    ddy = 0.01
    rec = matplotlib.patches.Rectangle(xy=((box_coef['right']+box_coef['left'])/2-3*ddx, box_coef['bottom']-(0.7*ddy)),
                                       width=box_coef['right']-(box_coef['right']+box_coef['left'])/2+(0.2*ddx),
                                       height=box_coef['top']-box_coef['bottom']+(1.7*ddy), transform=fig.transFigure, ec=None, fc=cBG, clip_on=False, zorder=-100)
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

    var_sold  = [];
    var_snew  = [];
    epi_sold  = [];
    epi_snew  = [];
    var_traj  = [];
    hig_traj  = [];
    var_tag   = [];
    var_note  = [];
    ds_matrix = [];
    new_var   = [];
    new_note  = [];

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
            if df_entry.nucleotide!='-' and df_entry.pc_MPL > 0:
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
    sspace  = 0.02

    box_ss   = dict(left=0.10, right=0.45, bottom=0.32, top=0.92)
    box_traj = dict(left=0.53, right=0.91, bottom=0.77, top=0.92)
    box_sij  = dict(left=0.53, right=0.88, bottom=0.12, top=0.59)

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
               'xlabel':       'Inferred selection coefficients with escape terms',
               'ylabel':       'Inferred selection coefficients without escape terms',
               'theme':        'boxed'}

    mp.line(ax=ax_ss, x=[[s_min, s_max]], y=[[s_min,s_max]], colors=[C_NEU],plotprops=lineprops, **pprops)

    for i in range(len(var_snew)):
        mp.plot(type='scatter', ax=ax_ss, x=[[var_snew[i]]], y=[[var_sold[i]]], colors=[C_NEU],plotprops=scatterprops, **pprops)

    scatterprops['alpha'] = 0.8
    for i in range(len(epi_snew)):
        for j in range(len(epi_snew[i])):
            mp.plot(type='scatter', ax=ax_ss, x=[[epi_snew[i][j]]], y=[[epi_sold[i][j]]], colors=[var_c[i]],plotprops=scatterprops, **pprops)

    traj_legend_x  = 0.016
    traj_legend_dy = -0.003
    y0             = 0.003
    dx0            = 0.002
    traj_legend_y  = [y0, y0 + traj_legend_dy ,y0 + traj_legend_dy*2]
    scatterprops['s'] = SMALLSIZEDOT*0.8
    for k in range(len(var_tag)):
        traj_legend_k = 'escape sites in epitope '+var_tag[k]
        mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[traj_legend_y[k]]], colors=[var_c[k]],plotprops=scatterprops, **pprops)
        ax_ss.text(traj_legend_x, traj_legend_y[k], traj_legend_k, ha='left', va='center', **DEF_LABELPROPS)

    mp.plot(type='scatter', ax=ax_ss, x=[[traj_legend_x-dx0]], y=[[y0 + traj_legend_dy*3]], colors=[C_NEU],plotprops=scatterprops, **pprops)
    ax_ss.text(traj_legend_x, y0 + traj_legend_dy*3, 'not escape sites', ha='left', va='center', **DEF_LABELPROPS)

    ddx = -0.0001
    ddy =  0.0025
    for i in range(len(var_note)):
        ax_ss.text(var_note[i][0]+ddx, var_note[i][1]+ddy, new_note[i], ha='center', va='center', **DEF_LABELPROPS)

    ax_ss.text(box_ss['left']+dx/2, box_ss['top']+dy/2, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- trajectory plot

    pprops = { 'xticks':      traj_ticks,
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'xlabel':      'Time (days)',
               'ylabel':      'Variant frequency\nfor CH 470-5\n',
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

    traj_legend_x  =  380
    traj_legend_dy = -0.2
    traj_legend_y  = [0.4, 0.4 + traj_legend_dy]
    traj_legend_t  = ['974A', '3951C']
    for k in range(len(traj_legend_y)):
        yy = traj_legend_y[k]
        mp.line(ax=ax_traj, x=[[traj_legend_x-35, traj_legend_x-10]], y=[[yy,yy]], colors=[traj_c[k]], **pprops)
        ax_traj.text(traj_legend_x, traj_legend_y[k], traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    ax_traj.text(box_traj['left']+dx, box_traj['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c -- effects of linkage on selection for sites

    pprops = { 'colors': [BKCOLOR],
               'xlim': [0, len(ds_matrix[0]) + 1],
               'ylim': [1, len(ds_matrix)],
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

    txtprops = dict(ha='right', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    for i in range(len(ds_matrix)):
        ax_sij.text(-0.2, len(ds_matrix) - i + 0.5, '%s' % new_var[i], **txtprops)

    txtprops = dict(ha='center', va='top', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    for i in range(len(ds_matrix)):
        ax_sij.text(i + 0.5, 0.8, '%s' % new_var[i], **txtprops)
    for i in range(len(var_tag)):
        ax_sij.text(len(ds_matrix) + i + 0.8, 0.8, '%s' % var_tag[i], **txtprops)

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    ax_sij.text(15.5, 9,  2, **txtprops)
    ax_sij.text(15.5, 7.5,  1, **txtprops)
    ax_sij.text(15.5, 6,  0, **txtprops)
    ax_sij.text(15.5, 4.5,  -1, **txtprops)
    ax_sij.text(15.5, 3, -2, **txtprops)
    ax_sij.text(7.5, -1, 'Variant i', **txtprops)
    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=90)
    ax_sij.text(-2, 7, 'Target variant j', **txtprops)


    for i in range(-3, 3+1, 1):
        c = ''
        t = i/3
        if t>0:
            c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
        else:
            c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
        rec = matplotlib.patches.Rectangle(xy=(14, 5.5+i), fc=c, **site_rec_props)
        rec_patches.append(rec)


    for patch in rec_patches:
        ax_sij.add_artist(patch)

    ax_sij.text(7, 11.5, 'Linkage effects on inferred coefficients for 974A', ha='center', va='center', **DEF_LABELPROPS)

    ax_sij.text(box_traj['left']+dx, box_sij['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/fig6.pdf' % FIG_DIR, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    print('figure 6 done.')

""" 
UTILITY DESCRIBTION:
--------------------
As the name states this is a utility help the user to make nice figures for esspecially timeseries analysis, and make your code look more simple. Most routines calls the 'setting option', which include everything how the data should be presented.
"""

# Numpy:
import numpy as np
# Others: 
import time, sys
# Plot tools:
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

###########################################################################################################
#                                                SETTINGS                                                 #
###########################################################################################################
    
def plot_settings(xlabel, ylabel, title=None, legend=None):
    FS = 15; FT = 15
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'xlabel',fontsize=FS); plt.xlabel(xlabel) 
    plt.ylabel(r'ylabel',fontsize=FS); plt.ylabel(ylabel)
    plt.rc('xtick', labelsize=FS)
    plt.rc('ytick', labelsize=FS)
    plt.minorticks_on()
    #plt.tight_layout()
    if legend!=None:
        if legend==1: legend = 'upper right'
        if legend==2: legend = 'upper left'
        if legend==3: legend = 'lower left'
        if legend==4: legend = 'lower right'
        plt.legend(loc=legend,numpoints=1,frameon=False,fontsize=FT)
    if title!=None:
        plt.title(r'\bfseries {}'.format(title), fontsize=FT)    


def plot_axis(x, y, bx, by):
    # White spaces around figures:
    wx = (np.nanmax(x) - np.nanmin(x))*0.01*bx   # bx is in procent
    wy = (np.nanmax(y) - np.nanmin(y))*0.01*by   # by is in procent
    # Apply limits:
    plt.xlim(x[0]-wx, x[-1]+wx)
    plt.ylim(np.nanmin(y)-wy, np.nanmax(y)+wy)
        

def plot_subplot(n):
    from matplotlib import gridspec as gs
    if n==1: # TRANSIT 
        g  = gs.GridSpec(4, 1)
        ax = plt.subplot(g[1:3]) 
    if n==2: # Window
        g  = gs.GridSpec(3, 7)
        ax = plt.subplot(g[0:2, 4:7])   
    if n==3: # Thin and broad
        g  = gs.GridSpec(5, 1)
        ax = plt.subplot(g[0:2])
    if n==4: # Timeseries
        g  = gs.GridSpec(1, 4)
        ax = plt.subplot(g[1]) 

###########################################################################################################
#                                              GENERAL PLOTS                                              #
###########################################################################################################

def PLOT(data, mark, xlab, ylab, title=None, subplot=0, legend=1,  axis=[1,1]):
    """
    General function to make fast plots in one command line:
    --------INPUT:
    data       (array)  : Data structure e.g. [data0, data1];  data0 and data1 have a x, y coloumn.
    mark       (list)   : If one have 2 datasets use e.g. ['b-', 'k.']
    xlab, ylab (string) : Labels on x and y
    title      (string) : Title
    legpos     (float)  : This can be 1, 2, 3, and 4 corresponding to each quadrant.
    subplot    (float)  : Different types of subplots.
    axis       (list)   : Procentage edge-space in x and y. E.g. [1, 5] to 1% in x and 5% in y. 
    """
    # Type of subplot:
    if subplot is not 0: plot_subplot(subplot)

    # Plot data:
    if legend is 1:
        for i in range(len(data)):
            plt.plot(data[i][:,0], data[i][:,1], mark[i])
        plot_settings(xlab, ylab, title)
    if legend is not 1:
        for i in range(len(data)):
            plt.plot(data[i][:,0], data[i][:,1], mark[i], label=legend[i+1])
        plot_settings(xlab, ylab, title, legend[0])

    # Axes setting:
    plot_axis(data[0][:,0], data[0][:,1], axis[0], axis[1])
    plt.show()

    
def SURF(x, y, z, xlab, ylab, zlab, title):
    # Find (x, y) value for maximum peak:
    z_max   = np.max(z, axis=0)
    z_max_i = np.where(z==z_max)
    print 'Best Period: {:.6f} days'.format(x[z_max_i[0][0]])
    print 'Best Phase : {:.6f} days'.format(y[z_max_i[1][0]])
    # 3D plot:
    y, x = np.meshgrid(y, x)
    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    # Axes labels and title:
    ax.set_xlabel(xlab, fontsize=12); ax.tick_params(axis='x', labelsize=10)
    ax.set_ylabel(ylab, fontsize=12); ax.tick_params(axis='y', labelsize=10)
    ax.set_zlabel(zlab, fontsize=12); ax.tick_params(axis='z', labelsize=10)
    plt.title(title,    fontsize=15)
    # Extra settings:
    # ax.invert_xaxis()                          # Invert x-axis
    # ax.view_init(30, 45)                       # Viewing angle 
    # fig.colorbar(surf, shrink=0.5, aspect=8)   # Colorbar
    plt.show()

    
def HIST(hist, bins, xlab, ylab, x_int=None, y_int=None, title=None):
    plt.hist(hist, bins, edgecolor='k', alpha=1, log=True)
    plot_settings(xlab, ylab, title)
    if x_int!=None:
        plt.xlim(x_int[0], x_int[1])
        plt.ylim(y_int[0], y_int[1])


def TRANSIT(data, axes, zoombox, xlabel, ylabel):
    # Subplot first:
    from matplotlib import gridspec as gs
    g  = gs.GridSpec(4, 1)
    ax = plt.subplot(g[0:2])
    # Plot data:
    plt.plot(data[:,0], data[:,1], 'r-')
    plot_settings(xlabel, ylabel)
    plot_axis(data[:,0], data[:,1], 1, 2)
    # Fancy subplot:
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, zoombox, loc=3) #bbox_to_anchor=(380, 345)) # customized position 
    axins.plot(data[:,0], data[:,1], 'r-')
    x1, x2, y1, y2 = axes[0], axes[1], axes[2], axes[3] # specify the limits
    axins.set_xlim(x1, x2); plt.xticks(visible=False)
    axins.set_ylim(y1, y2); plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")

    
def PHASEFOLD(data, P, phi, dT, depth, mark):
    """
    This routine takes your data a phasefold it by the period (P [days])
    """
    # Time, P, phi, dT is given in [days]
    t     = data[:,0]
    S     = data[:,1]
    phase = np.mod(t, P)
    # Whole phase plot:
    plt.plot(phase, S, mark)
    xpos = min(phase)+(max(phase)-min(phase))*0.70
    ypos = max(S)-(max(S)-min(S))*0.98
    plt.text(xpos, ypos,'$P={:.4f}$ days'.format(P), fontsize=18)
    plt.title('Phase folded', fontsize=18)
    plot_axis([min(phase), max(phase)], S, 1, 1)
    plot_settings('$t$ mod $P$ [days]', 'Signal')
    plt.show()
    # Zoom-in on transit:
    plt.plot(phase, S, mark)
    plot_axis([phi-dT, phi+dT], [depth, max(S)], 1, 1)
    xpos  = phi-dT+((phi+dT)-(phi-dT))*0.70
    ypos1 = depth+0.0000
    ypos2 = depth+0.0002 
    plt.text(xpos, ypos2, '$P={:.4f}$ days'.format(P), fontsize=18)
    plt.text(xpos, ypos1, '$\phi={:.4f}$ days'.format(phi), fontsize=18)
    plt.title('Phase  folded', fontsize=18)
    plot_settings('$t$ mod $P$ [days]', 'Signal')
    plt.show()


###########################################################################################################
#                                          PLOTS TOOLS - SPECIAL                                          #
###########################################################################################################

    
    
def plot_locate(t, dif0, dif1, cut_off, n, xlabel, ylabel):
    cut_up = np.ones(len(t))*cut_off
    cut_dw = -cut_up
    plt.plot(t, dif0, 'b-', label='Outliers')
    plt.plot(t, dif1, 'k-', label='Corrected')
    plt.plot(t, cut_up, 'r-', t, cut_dw, 'r-')
    # Settings:
    plot_settings(xlabel, ylabel, 3)
    plot_axis(t, dif0, 1, 2)
    # Print used n:
    xpos = t[0]+(t[-1]-t[0])*0.1
    ypos = max(dif0)-(max(dif0)-min(dif0))*0.1
    plt.text(xpos, ypos, '$n={}$'.format(n), fontsize=25)
    plt.show()

    
def plot_gapsize(S_diff, xlabel, ylabel):
    binwidth = 5
    plt.hist(S_diff, bins=range(int(min(S_diff)), int(max(S_diff))+binwidth, binwidth), log=True)
    # Settings:
    plt.xlim(-1e3, 1e3); plt.ylim(5e-1, 1e3)
    plot_settings(xlabel, ylabel, 3, 'Threshold for Jumps')
    plt.show()

    
def plot_jumps(t, S0, S1, xlabel, ylabel):
    plot_subplot(0)
    plt.plot(t, S0, 'k.', label='Old data')
    plt.plot(t, S1, 'm.', label='Jump correction')
    # Settings:
    plot_settings(xlabel, ylabel, 3)
    plot_axis(t, S0, 1, 1)
    plt.show()

    
def plot_stellarnoise_power(data0, data1, fx_int, fy_int, xlabel, ylabel):
    plot_subplot(0)
    plt.plot(data0[:,0], data0[:,1], 'k-', label='Old data')
    plt.plot(data1[:,0], data1[:,1], 'b-', label='Cleaned data')
    # Settings:
    plot_settings(xlabel, ylabel, 1)
    plot_axis(fx_int, fy_int, 1, 1)
    plt.show()

    
def plot_stellarnoise_time(data0, data1, xlabel, ylabel):
    plot_subplot(0)
    plt.plot(data0[:,0], data0[:,1], 'k-', label='Old data')
    plt.plot(data1[:,0], data1[:,1], 'b-', label='Cleaned data')
    # Settings:
    plot_settings(xlabel, ylabel, 3)
    plot_axis(data0[:,0], data0[:,1], 1, 1)
    plt.show()

    
def plot_slowtrend(data0, data1, data2, data3, m1, m2, xlabel, ylabel):
    plot_subplot(0)
    plt.plot(data0[:,0], data0[:,1], 'k-', label='No filter')
    plt.plot(data1[:,0], data1[:,1], 'b-', label='Median filter: $m={}$'.format(m1))
    plt.plot(data2[:,0], data2[:,1], 'r-', label='Mean   filter: $m={}$'.format(m2))
    # Settings:
    plot_settings(xlabel, ylabel, 3)
    plot_axis(data0[:,0], data0[:,1], 1, 1)
    plt.show()
    
def plot_period_cc(P, cc, mark):
    # Extract maximum cc amplitude found for each period:
    A_cc = np.zeros(len(P))
    for i in range(len(A_cc)):
        A_cc[i] = np.max(cc[i,:])
    # Find best period:
    cc_max   = np.max(cc)
    cc_max_i = np.where(cc==cc_max)
    P_best   = P[cc_max_i[0][0]]
    # Plot:
    plt.plot(P, A_cc, mark)
    plt.xlim(P[0], P[-1])
    plt.ylim(0, max(A_cc)+max(A_cc)*0.1)
    plt.title('Projected max CC-Amplitude', fontsize=16)
    plot_settings('$P$ [days]', 'A')
    # Print period::
    xpos = min(P)+(max(P)-min(P))*0.65
    ypos = max(A_cc)-(max(A_cc)-min(A_cc))*0.05
    plt.text(xpos, ypos,'$P={:.4f}$ days'.format(P_best), fontsize=16)
    plt.show()

    
def plot_interpolation(data0, data1, xlabel, ylabel):
    plt.plot(data0[:,0], data0[:,1], 'k-', label='Old data')
    plt.plot(data1[:,0], data1[:,1], 'b.', label='Interpolation')
    # Settings:
    plot_axis(data0[:,0], data0[:,1], 1, 1)
    plot_settings(xlabel, ylabel, 3)
    plt.show()

    
def plot_autocor(P, ac):
    plt.plot(P, ac, 'b-', label='Old data')
    # Settings: 
    plot_settings('Period (days)', 'AC Amplitude', 3, 'Auto-Correlation')
    plot_axis(P, ac, 1, 1)
    plt.show()

    
def plot_period_ac(P_mea, P_cal):
    # Linear regression between P_mea vs. P_cal:
    index = where(P_mea > 0)[0][:]
    coef, stats = np.polynomial.polyfit(P_cal[index], P_mea[index], 1, full=True)
    b = coef[0]
    a = coef[1]
    p = arange(0, P_cal[-1]+120, 1)
    P_fit = a*p+b
    # Plot:
    plt.plot(P_cal, P_mea, 'ko', label='Data')
    plt.plot(p, P_fit, 'r-', label='Linear fit')
    # Settings:
    plot_settings('Calculated $N\times P-P$ (days)', 'Measured $N\times P$ (days)', 2, 'Auto-Correlation')
    plot_axis(P_cal, P_mea, 1, 1)
    # Add the uncertainty from the fit in days:
    if len(P_mea)>2:
        plt.text(0.13, 0.85, 'SNR=%.4f' %stats[0][0], fontsize=15)
    # Add best estimate of period in days:
    plt.text(0.10, 0.75, '$P=%.2f$ $days$' %b, fontsize=15)
    plt.title('OC-diagram for $P$')
    print 'Best      $P_{AC}={}$ days'.format(b)
    print 'Estimated $P_{AC}={}$ days'.format(P)
    plt.show()

#########################################################################################################
#                                           PRINT TO BASH                                               #
#########################################################################################################

def loading(i, i_max):
    """ This function print the loading status of. """
    sys.stdout.write(u"\u001b[1000D")
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.write(str(i + 1) + "Loding... %")
    sys.stdout.flush()
    
    
def compilation(i, i_max, text):
    """ This function print out a compilation time menu bar in the terminal. """ 
    percent = (i + 1) / (i_max * 1.0) * 100
    # print int(percent/2), 50-int(percent/2)
    # We here divide by 2 as the length of the bar is only 50 characters:
    bar = "[" + "-" * int(percent/2) + '>' + " " *(50 - int(percent/2)) + "] {}% {}".format(int(percent), text)
    sys.stdout.write(u"\u001b[1000D" +  bar)
    sys.stdout.flush()

    

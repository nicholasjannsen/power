# Numpy:
from numpy import loadtxt, savetxt, random, linspace, array, vstack
from numpy import sin, cos, pi
from numpy import sum
# Others:
import numpy as np
import math, os, sys, time, random
import matplotlib.pyplot as plt
# Functions:
from Timeseries_Tools import power, window, clean, filters, moving
from Timeseries_Tools import locate, jumps, stellar_noise, slowtrend, crosscor, autocor
# Plots:
from Plot_Tools import PLOT, TRANSIT, PHASEFOLD, SURF, plot_period_cc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
##################################################################################################

def test(name):

    #-------------------------------
    # EXERCISE 2 - STELLAR ANALYSIS:
    #-------------------------------
    
    # Test of weights:
    if name=='weights':
        t_int = [0, 1]                              # Time interval
        f_int = [0, 101]                            # Frequency interval
        amp = [1, 1, 1]                             # Frequency amplitudes                   
        ft  = [20, 50, 80]                          # Frequency positions
        t   = np.arange(t_int[0], t_int[1], 1e-3)
        S0  = amp[0]*np.sin(2*np.pi*ft[0]*t)        # Create 3 sinusoidal peaks
        S1  = amp[1]*np.cos(2*np.pi*ft[1]*t)
        S2  = amp[2]*np.sin(2*np.pi*ft[2]*t)
        S   = S0 + S1 + S2
        w   = np.random.normal(5,7,len(S))          # Create random weights
        data = np.vstack([t, S, w]).T
        # Funtions:
        Pf_power, _, _, _, = power(data, f_int, 0.1, 1)
        # Plotting:
        f = Pf_power[:,0]
        P = Pf_power[:,1]
        data0 = np.vstack([f, P]).T
        PLOT([data0], 'b-', '$f$ (c/d)', '$P$ (arb. unit)', 'Test of Random Weights', 1)

        
    # Test of window:
    if name=='window':
        # Create data:
        f_int, t_int, f_res, sampling, N = [0.1, 100], [0, 2], 0.1, 1, 1
        amp   = random.sample(xrange(5,20),N)       # Random Aplitudes   (integer)
        ft    = random.sample(xrange(1,100),N)      # Random Frequencies (integer)
        noise = np.random.normal(0,100,1000)        # Random noise signal
        print "Random Frequencies: %s" %ft
        t    = np.linspace(t_int[0], t_int[1], 1e3)
        f    = [amp[i]*np.sin(2*np.pi*ft[i]*t) for i in range(N)]  # N frequency columns
        S    = np.sum(f,0)+noise                                   # add columns and add noise
        data = np.vstack([t, S]).T
        # Functions: 
        Pf_power, _, _, _, = power(data, f_int, f_res, sampling)
        Pf_window          = window(data, f_int, f_res, sampling)
        print sum(Pf_window)
        # Plotting:
        PLOT([Pf_power], 'b-', '$f$ (c/d)', '$P$ (arb. unit)', 'Power spectrum', 1)
        PLOT([Pf_window], 'b-', '$f$ (c/d)', '$P$ (arb. unit)', 'Window function', 2)

        
    # Test of clean:
    if name=='clean':
        # This function cleans the power spectrum for N peaks having the higest signal.
        f_int = [0, 100]
        t_int = [0, 2]
        N, f_res = 3, f_int[1]/1e3
        amp   = random.sample(xrange(5,50),N)   # Random Aplitudes   (integer)
        ft    = random.sample(xrange(1,100),N) # Random Frequencies (integer)
        noise = np.random.normal(0,100,1000)/5  # Random noise signal
        print "Random Frequencies: %s" %ft
        t    = np.linspace(t_int[0], t_int[1], 1e3)
        f    = [amp[i]*np.sin(2*np.pi*ft[i]*t) for i in range(N)]  # N frequency columns
        S    = np.sum(f,0)+noise                                   # add columns and add noise
        data = np.vstack([t, S]).T
        # Functions: 
        Pf_power, _, _, _, = power(data,     f_int, f_res)
        St_clean, _, _,    = clean(data, N,  f_int, f_res)
        Pf_clean, _, _, _, = power(St_clean, f_int, f_res)
        # Plotting:
        PLOT([data, St_clean], ['b-', 'k-'], '$t$ (days)', '$S$ (arb. unit)', 'Test Clean', 1)
        PLOT([Pf_power, Pf_clean], ['b-', 'k-'], '$f$ (c/d)', '$P$ (arb. unit)', 'Test Clean', 1, \
             [1, 'Raw', 'Cleaned'])

        
    # Test of filter:
    if name=='filters':
        # This function simulates N random peaks an make a high and low pass filter.
        N = 3
        f_int = [0, 50, 100]
        t_int = [0, 2]
        f_res = 0.1
        amp   = random.sample(xrange(10,20),N)  # Random Aplitudes   (integer)
        ft    = random.sample(xrange(1,100),N)  # Random Frequencies (integer)
        noise = np.random.normal(0,101,1000)/3   # Random noise signal
        print 'Random frequencies %s' %ft
        print 'Random amplitudes  %s' %amp 
        t     = np.linspace(t_int[0], t_int[1], 1e3)
        f     = [amp[i]*np.sin(2*np.pi*ft[i]*t) for i in range(len(ft))]
        S     = np.sum(f,0)+noise
        data  = np.vstack([t, S]).T 
        # functions: 
        Pf_power, _, _, _, = power(data,    [f_int[0], f_int[2]], f_res)
        St_low, St_high    = filters(data,  [f_int[1], f_int[2]], f_res)
        Pf_low, _, _, _,   = power(St_low,  [f_int[0], f_int[2]], f_res)
        Pf_high, _, _, _,  = power(St_high, [f_int[0], f_int[2]], f_res)
        # Plotting:
        PLOT([data, St_low, St_high], ['k-', 'b-', 'r-'], '$t$ (days)', '$S$ (arb. unit)', 'Test Filter', 1)
        PLOT([Pf_power, Pf_low, Pf_high], ['k-', 'b-', 'r-'], '$f$ (c/d)', '$P$ (arb. unit)', 'Test Filter', \
             1, [1, 'Raw', 'Low-pass', 'High-pass'])

        
    # Exercise 1 - Porcyon:
    if name=='procyon':
        f_interval = [0, 160] # 20, 160
        data = np.loadtxt('/home/nicholas/Data/Kepler/procyon')
        Pf_power, _, _, _,  = power(data,  f_interval, None, 1, w_column=3)
        Pf_window           = window(data, f_interval, None, 1, w_column=3)
        PLOT([Pf_power], 'b-', '$f$ (c/d)', '$P$ (arb. unit)', 'Procyon')
        PLOT([Pf_window], 'b-', '$f$ (c/d)', '$P$ (arb. unit)')

        
    # Exercise 2 - delta Scuti:
    if name=='scuti':
        N, f_int = 7, [0, 40]
        name = "kepler-star3-delta-scuti"
        data = np.loadtxt('/home/nicholas/Data/Kepler/{}'.format(name))
        # Functions:
        Pf_power, _, _, _, = power(data,     f_int)
        St_clean, _, _,    = clean(data, N,  f_int)
        Pf_clean, _, _, _, = power(St_clean, f_int)
        # Plotting:
        t_interval = [data[0,0], data[-1,0]];  f_interval = [0, 1./(data[-1,0]-data[0,0])]
        PLOT([data, St_clean], ['k-', 'b-'], '$t$ (days)', '$S$ (arb. unit)', r'$\delta$ scuti')
        PLOT([Pf_power, Pf_clean], ['k-', 'b-'], '$f$ (c/d)', '$P$ (arb. unit)', r'$\delta$ scuti', \
             1, [1, 'Raw', 'Clean'])

       
    # Sun SoHO/GOLF:
    if name=="sun":
        name  = "solar-velocity"
        data = np.loadtxt('/home/nicholas/Data/Kepler/{}'.format(name))
        data[:,1] = data[:,1]-np.mean(np.array(data[:,1]))   # Correct for zero frequency
        t_int = [data[0,0], data[-1,0]]; f_int = [0, 555]
        peak0 = [162, 164];  N0 = 10 
        peak1 = [260, 263];  N1 = 20
        peak2 = [346, 354];  N2 = 30

        EX = 3
        
        # Exercise 3:
        if EX==3:
            Pf_power, _, _, _, = power(data,   f_int)
            St_low,   _,       = filters(data, f_int)
            PLOT([data, St_low], ['k-', 'r-'], '$t$ (days)', '$v$ (m/s)', None)
        
        # Exercise 4:
        if EX==4:
            St_band0, _, = filters(data,   peak0)
            St_band1, _, = filters(data,   peak1)
            St_band2, _, = filters(data,   peak2)
            Pf_band0, _, _, _, = power(St_band0, peak0)
            Pf_band1, _, _, _, = power(St_band1, peak1)
            Pf_band2, _, _, _, = power(St_band2, peak2)
            # Plotting: 
            PLOT([Pf_band0], ['k-'], '$P$ (arb. unit)', '$f$ (c/d)')
            PLOT([Pf_band1], ['r-'], '$P$ (arb. unit)', '$f$ (c/d)')
            PLOT([Pf_band2], ['b-'], '$P$ (arb. unit)', '$f$ (c/d)')
            PLOT([St_band2, St_band1, St_band0], ['b-', 'r-', 'k-'],  '$t$ (days)', '$v$ (m/s)')

        
        # Exercise 5:
        if EX==5:
            #----- Peak 0:
            St_band0, _,     = filters(data,     peak0, f_resolution, sampling, w_column)
            St_clean0, _, _, = clean(data, N0,   peak0, f_resolution, sampling, w_column)
            # Pf_power0, _,    = power2(data,      peak0, f_resolution, sampling, w_column)
            # Pf_clean0, _,    = power2(St_clean0, peak0, f_resolution, sampling, w_column)
            S_rest0  = data[:,1]-St_clean0[:,1]
            St_rest0 = np.vstack([data[:,0], S_rest0]).T
            S_sub0   = St_band0[:,1]-S_rest0
            St_sub0  = np.vstack([data[:,0], S_sub0]).T
            #----- Peak 1:
            St_band1, _,     = filters(data,     peak1, f_resolution, sampling, w_column)
            St_clean1, _, _, = clean(data, N1,   peak1, f_resolution, sampling, w_column)
            # Pf_power1, _,    = power2(data,      peak1, f_resolution, sampling, w_column)
            # Pf_clean1, _,    = power2(St_clean1, peak1, f_resolution, sampling, w_column)
            S_rest1  = data[:,1]-St_clean1[:,1]
            St_rest1 = np.vstack([data[:,0], S_rest1]).T
            S_sub1   = St_band1[:,1]-S_rest1
            St_sub1  = np.vstack([data[:,0], S_sub1]).T
            #----- Peak 2:
            St_band2, _,     = filters(data,     peak2, f_resolution, sampling, w_column)
            St_clean2, _, _, = clean(data, N2,   peak2, f_resolution, sampling, w_column)
            # Pf_power2, _,    = power2(data,      peak2, f_resolution, sampling, w_column)
            # Pf_clean2, _,    = power2(St_clean2, peak2, f_resolution, sampling, w_column)
            S_rest2  = data[:,1]-St_clean2[:,1]
            St_rest2 = np.vstack([data[:,0], S_rest2]).T
            S_sub2   = St_band2[:,1]-S_rest2
            St_sub2  = np.vstack([data[:,0], S_sub2]).T
            # plotting:
            plot_timeseries3(St_rest0, St_rest1, St_rest2, t_int)
            # plot_timeseries(St_sub0, t_int)
            # plot_power21(Pf_power0, Pf_clean0, peak1)
            # plot_timeseries(St_sub1, t_int)
            # plot_power21(Pf_power1, Pf_clean1, peak1)
            # plot_timeseries(St_sub2, t_int)
            # plot_power21(Pf_power2, Pf_clean2, peak2)
       
    # Exercise 6 - Solar-like stars:
    if name=="solar":
        name0  = "kepler-star1-solar-like"
        name1  = "kepler-star2-solar-like"
        data0  = np.loadtxt(os.path.join('/home/nicholas/Dropbox/Uni/Timeseries/Data', name0))
        data1  = np.loadtxt(os.path.join('/home/nicholas/Dropbox/Uni/Timeseries/Data', name1))
        t_int0 = [data0[0,0], data0[-1,0]];  f_int0 = [0, 400] # No peaks are visible above
        t_int1 = [data1[0,0], data1[-1,0]];  f_int1 = [0, 400] # No peaks are visible above
        f_int  = [0, 700]
        #-------- i) High-pass filtering: 
        Pf_power0, _, = power2(data0, f_int, f_resolution, sampling, w_column)
        Pf_power1, _, = power2(data1, f_int, f_resolution, sampling, w_column)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_Pf2.txt', Pf_power0)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_Pf2.txt', Pf_power1)
        # _, St_high0   = filters(data0,   f_int0, f_resolution, sampling, w_column)
        # _, St_high1   = filters(data1,   f_int1, f_resolution, sampling, w_column)
        # Pf_high0, _,  = power2(St_high0, [400, 700],  f_resolution, sampling, w_column)
        # Pf_high1, _,  = power2(St_high1, [400, 700],  f_resolution, sampling, w_column)
        # print("--- %s seconds ---" % (time.time() - start_time))
        # plot_timeseries21(data0, St_high0, t_int0)
        # plot_timeseries22(data1, St_high1, t_int1)
        # plot_power21(Pf_power0,  Pf_high0, f_int)
        # plot_power22(Pf_power1,  Pf_high1, f_int)
        # # Save data:
        # # print len(data0[:,0]), len(St_high0[:,1]), len(Pf_high0[:,0]), len(Pf_high0[:,1])
        # # print len(data1[:,0]), len(St_high1[:,1]), len(Pf_high1[:,0]), len(Pf_high1[:,1])
        # star1_St = np.vstack([data0[:,0],    St_high0[:,1]]).T
        # star1_Pf = np.vstack([Pf_high0[:,0], Pf_high0[:,1]]).T
        # star2_St = np.vstack([data1[:,0],    St_high1[:,1]]).T
        # star2_Pf = np.vstack([Pf_high1[:,0], Pf_high1[:,1]]).T
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_St.txt', star1_St)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_Pf.txt', star1_Pf)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_St.txt', star2_St)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_Pf.txt', star2_Pf)
        #-------- ii) Scattter:
        ## Load data:
        # St_star1 = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_St.txt')
        # St_star2 = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_St.txt')
        # std1 = bn.move_std(St_star1[:,1], window=50, min_count=1)
        # std2 = bn.move_std(St_star2[:,1], window=50, min_count=1)
        # w1   = std1**-2 
        # w2   = std2**-2
        # Data1 = np.vstack([data0[:,0], data0[:,1], w1]).T 
        # Data2 = np.vstack([data1[:,0], data1[:,1], w2]).T 
        # Pf_power1, _, = power2(Data1, f_int, f_resolution, sampling)
        # Pf_power2, _, = power2(Data2, f_int, f_resolution, sampling)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_Pfw.txt', Pf_power1)
        # np.savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_Pfw.txt', Pf_power2)
        ## Plotting:
        Pf1  = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_Pf2.txt')
        Pf2  = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_Pf2.txt')
        Pfw1 = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star1_Pfw.txt')
        Pfw2 = np.loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/solar_star2_Pfw.txt')
        dP1  = Pf1[:,1]-Pfw1[:,1]
        dP2  = Pf2[:,1]-Pfw2[:,1]
        # Plot:
        plot_power22(Pf1, Pfw1, f_int)
        plot_power22(Pf2, Pfw2, f_int)
        plot_power(np.vstack([Pf1[:,0], dP1]).T, f_int)
        plot_power(np.vstack([Pf2[:,0], dP2]).T, f_int)
        

    #----------------------------------
    # EXERCISE 3 - EXOPLANETS ANALYSIS:
    #----------------------------------
        
    #Exoplanet 1:
    if name=="exo1":
        # Load data:
        plot = 1
        name = 'kepler-exoplanet1'
        path = '/home/nicholas/Data/Kepler/'
        data = loadtxt('{}{}'.format(path, name))
        TRANSIT(data, [663, 671, 2.785e5, 2.825e5], 8, 'Time (days)', 'Siganl'); plt.show()
        #----- Correct data:
        data = locate(data, 1, 3e-4, plot)                      # Corrects for bad data
        data = jumps(data, 500, plot)                           # Corrects for jumps
        data = stellar_noise(data, [0.05, 0.2], 0.1, 8, plot)   # Cleans for stellar signals.
        data = slowtrend(data, 0.2, 20, 10, 1, plot)            # Corrects for slow trends.
        # savetxt('{}{}'.format(path, 'Exo1_corrected.txt'), data) # Save corrected timeseries
        #----- Corrected data:
        data = loadtxt('{}{}'.format(path, 'Exo1_corrected.txt')) 
        #----- CC:
        save_P   = '{}{}'.format(path, 'exo1_P.txt')    # P   = 103 h
        save_phi = '{}{}'.format(path, 'exo1_phi.txt')  # phi = 21 h
        save_cc  = '{}{}'.format(path, 'exo1_cc.txt') 
        crosscor(data, [93, 113], [15, 25], [130, 3, 2, 1])#, [save_P, save_phi, save_cc])  
        #----- Plot CC:
        P    = loadtxt('{}exo1_P.txt'.format(path))
        phi  = loadtxt('{}exo1_phi.txt'.format(path))
        cc   = loadtxt('{}exo1_cc.txt'.format(path))
        P = array(P)/(60.*24.); phi = phi/(60.*24.)
        SURF(P, phi, cc, '$P$ [days]', '$\phi$ [days]', '$A$', 'Cross-Correlation')
        plot_period_cc(P, cc, 'b-')
        #----- AC:
        autocor(data, 10, [644, 828], [4e-9, 8e-9], 1)
        #----- Phase diagram:
        PHASEFOLD(data, 4.2869, 0.857, 0.15, 0.9985, 'b.')
             
    # Exoplanet 2:
    if name=="exo2":
        name = "kepler-exoplanet2"
        data = loadtxt(os.path.join('/home/nicholas/Data/Kepler/', name))
        #----- plot:
        TRANSIT(data, [664,667,2.363e5,2.368e5], 20, '$Time$ $(days)$', '$Signal$')
        #----- Correct data:
        data = locate(data, 1, 3e-4, 0)                      # Corrects for bad data
        data = jumps(data, 300, 0)                           # Corrects for jumps
        data = slowtrend(data, 0.2, 10, 10, 1, 1)            # Corrects for slow trends.
        savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo2_corrected.txt', data)
        #----- Corrected data:
        data = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo2_corrected.txt')
        #----- CC:
        save_P   = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_P.txt'   # P   = 223 h
        save_phi = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_phi.txt' # phi = 146 h 
        save_cc  = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_cc.txt'
        crosscor(data, [213, 233], [141, 151], [230,3,2,1], [save_P, save_phi, save_cc]) 
        #----- Plot CC:
        P    = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_P.txt')
        phi  = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_phi.txt')
        cc   = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo2_cc.txt')
        P = array(P)/(60.*24.); phi = phi/(60.*24.)
        SURF(P, phi, cc, '$P$ $[days]$', '$\phi$ $[days]$', '$A$', '$Cross-Correlation$')
        plot_period_cc(P, cc, 'r-')
        #----- AC:
        autocor(data, 20, 2.5e-9, 835, 1)
        #----- Phase diagram:
        PHASEFOLD(data, 9.2875, 6.094, 0.3, 0.9990, 'r.')
       
        
    # Exoplanet 3:
    if name=="exo3":
        name  = "kepler-exoplanet3"
        data  = loadtxt(os.path.join('/home/nicholas/Data/Kepler/', name))
        #----- Plot:
        TRANSIT(data,[654, 668, 1.4557e5, 1.4578e5], 2, '$Time$ $(days)$','$Signal$')
        #----- Correct data:
        data = locate(data, 1, 3.1e-4, 0)               # Corrects for bad data
        data = jumps(data, 200, 0)                      # Corrects for jumps
        data = slowtrend(data, 0.5, 25, 25, 0, 1)       # Corrects for slow trends.
        savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo3_corrected.txt', data)
        #----- Corrected data:
        data = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo3_corrected.txt')
        #----- CC:
        save_P   = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3d_P.txt' # 89h, 260h, 1863h
        save_phi = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3d_phi.txt' # 18h, 100, 1806
        save_cc  = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3d_cc.txt'
        crosscor(data, [79, 99], [13, 23], [170, 3, 2, 1], [save_P, save_phi, save_cc])  
        crosscor(data, [250, 270], [95, 105], [250, 3, 2, 1], [save_P, save_phi, save_cc])  
        crosscor(data, [1853, 1873], [1801, 1811], [490, 3, 2, 1], [save_P, save_phi, save_cc])
        #----- Plot CC:
        P    = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3b_P.txt')
        phi  = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3b_phi.txt')
        cc   = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo3b_cc.txt')
        P = array(P)/(60.*24.); phi = phi/(60.*24.)
        SURF(P, phi, cc, '$P$ $[days]$', '$\phi$ $[days]$', '$A$', '$Cross-Correlation$')
        plot_period_cc(P, cc, 'g-')
        #----- AC:
        #3b: [666, 790], [1.66e-9, 7.8e-9]
        #3c: [600, 830], [7.8e-9, 1]
        autocor(data, 20, [666, 790], [1.66e-9, 7.8e-9], 1)  
        #----- Phase diagram:        
        # PHASEFOLD(data, 3.6962, 0.748, 0.2, 0.9990, 'g.')  # b
        # PHASEFOLD(data, 10.8541, 4.168, 0.3, 0.9985, 'g.')  # c
        PHASEFOLD(data, 77.620, 75.25, 1.0, 0.9990, 'g.')  # d
       
        
    # Exoplanet 4:
    if name=="exo4":
        name = "kepler-exoplanet4"
        data = loadtxt(os.path.join('/home/nicholas/Data/Kepler/', name))
        #----- plot:
        TRANSIT(data, [656, 662, 1.6577e5, 1.660e5], 8, '$Time$ $(days)$','$Signal$')
        #----- Correct data:
        data = locate(data, 2, 3.4e-4, 0)                      # Corrects for bad data
        data = jumps(data, 300, 0)                             # Corrects for jumps
        data = slowtrend(data, 0.2, 25, 10, 1, 1)              # Corrects for slow trends.
        savetxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo4_corrected.txt', data)
        # Corrected data:
        data = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/Exo4_corrected.txt')
        #----- CC:
        save_P   = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_P.txt'   # P=77h
        save_phi = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_phi.txt' # phi=39h
        save_cc  = '/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_cc.txt'
        crosscor(data, [67, 87], [30, 45], [256, 3, 2, 1], [save_P, save_phi, save_cc]) 
        #----- Plot CC:
        P    = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_P.txt')
        phi  = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_phi.txt')
        cc   = loadtxt('/home/nicholas/Dropbox/Uni/Timeseries/Data/exo4_cc.txt')
        P = array(P)/(60.*24.); phi = phi/(60.*24.)
        SURF(P, phi, cc, '$P$ $[days]$', '$\phi$ $[days]$', '$A$', '$Cross-Correlation$')
        plot_period_cc(P, cc, 'm-')
        #----- AC:
        autocor(data, 20, 2.8e-9, 835, 1)
        #----- Phase diagram:
        PHASEFOLD(data, 3.2136, 1.618, 0.3, 0.9988, 'm.')
       
    return 
    
if __name__ == '__main__': #---------------------------- Main function --------------------------#
    # FUNCTION CALL:
    test(name='weights')

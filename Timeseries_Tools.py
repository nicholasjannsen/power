# Numpy:
import numpy as np

# Modules:
import math, sys, time

# Plots:
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Own functions:
import Timeseries_Tools as tt
import Plot_Tools as pt

##################################################################################################

def power(data, f_interval=None, f_resolution=None, sampling=None, w_column=None):
    """
    This function calculate the power spectrum of a time varying data set.
    ----------INPUT:
    data           : Is a matrix with a column of time, signal, and possible weights. 
    f_interval     : Frequency interval containing [fmin, fmax] (optional).
    f_resolution   : Frequency resolution (optional).
    sampling       : Sampling (oversampling >1) of the data (optional).
    w_column       : Column number of weight (optional). w_column = 0 (calculates without weights)
    ---------OUTPUT:
    Pf_data        : Frequencies (f) and power (P).
    P_comp         : Components for defining P (alpha and beta).
    f_interval     : Frequency interval may be usefull later.
    f_resolution   : Frequency resolution is handy when is it not specified
    """

    # Checking data structure:
    if ( np.size(data[0,:])<2   and                  # 2 columns minimum
         np.size(data[:,0])<2   and                  # 2 rows minimum 
         isinstance(data,float)):                    # Text
         sys.exit('Error: Wrong data structure')     # Abort

    # Replacing nan by 0:
    if np.sum(np.sum(data))==np.nan:
        data[data==nan]=0
    # Replacing inf by 0:
    if np.sum(np.sum(data))==np.inf or np.sum(np.sum(data))==-np.inf:
        data[data== np.inf]=0
        data[data==-np.inf]=0
       
    # Splitting data:
    t = data[:,0]                              # [days]
    S = data[:,1]-np.mean(np.array(data[:,1]))       # [arbt. unit] Correcting signal

    # Handy definitions:
    dt_min = np.min(np.diff(t))     # Minimum time interval
    dt_max = np.max(np.diff(t))     # Maximum time interval
    dt_med = np.median(np.diff(t))  # Median time interval
    f_Nq = 1./(2*dt_min)            # Cyclic Nyquist frequency
    f_md = 1./(2*dt_med)            # Cyclic median frequency
    
    # Checking if data is ordered in time:
    if dt_min<0:
        sys.exit('Error: Data not ordered in time')

    # Checking the existance of f_interval and f_resolution:
    # If f_interval and f_resolution is not defined:
    if f_interval==None and f_resolution==None:
        if dt_max>2*dt_med or math.isnan(f_Nq):
            # print 'Using f_md         = %f c/d' %f_md
            f_interval = [0, f_md]
            f_resolution = 1./(t[-1]-t[0])
        else:
            # print 'Using f_Nq         = %f c/d' %f_Nq
            f_interval = [0, f_Nq]
            f_resolution = 1./(t[-1]-t[0])
    # If only f_resolution is defined:
    elif f_interval==None and f_resolution!=None and isinstance(f_resolution, float): 
        f_resolution = f_resolution
        if dt_max>2*dt_med or math.isnan(f_Nq):
            # print 'Using f_md         = %f c/d' %f_md
            f_interval = [0, f_md]
        else:
            # print 'Using f_Nq         = %f c/d' %f_Nq
            f_interval = [0, f_Nq]
    # If only f_interval is defined:
    elif f_resolution==None and len(f_interval)==2:
        f_interval   = f_interval
        f_resolution = 1./(t[-1]-t[0])
    # If both f_interval and f_resolution is defined:
    elif f_interval!=None and f_resolution!=None:
        f_interval   = f_interval
        f_resolution = f_resolution
    else:
        sys.exit('Error: Wrong input argument for "f_interval"')
    # print 'Using f_interval   = {} c/d'.format(f_interval)    # Use .format when having tuples
    # print 'Using f_resolution = {} c/d'.format(f_resolution)
   
    # The frequency interval is found:
    if sampling==None: 
        f = np.arange(f_interval[0], f_interval[1], f_resolution)
    elif sampling!=None:
        # print 'Using sampling     = %f' %sampling
        f = np.arange(f_interval[0], f_interval[1], f_resolution*sampling)

    # The power spectrum is calculated:
    s  = np.zeros(len(f))
    c  = np.zeros(len(f))
    ss = np.zeros(len(f)) 
    cc = np.zeros(len(f))
    sc = np.zeros(len(f))
    picon = 2*np.pi*t
    
    # Without weights
    if w_column==0 or np.size(data[0,:])==2:
        # print 'Calculating WITHOUT weights'
        for i in range(len(f)):
            f_i = f[i]
            sinsum = np.sin(f_i*picon)
            cossum = np.cos(f_i*picon)
            s[i]  = np.sum(S*sinsum)
            c[i]  = np.sum(S*cossum)
            ss[i] = np.sum(sinsum**2)
            cc[i] = np.sum(cossum**2)
            sc[i] = np.sum(sinsum*cossum)
            # Print compilation time to bash:
            pt.compilation(i, len(f), 'power: without weights')
        print # This function is needed for compilation 

    # Calculate with weights:
    if np.size(data[0,:])>2:
        # print 'Calculating WITH weights'
        if w_column==None:
            w = data[:,2]
        elif w_column!=None:
            w = data[:,w_column]
        for i in range(len(f)):
            f_i = f[i]
            sinsum = np.sin(f_i*picon)
            cossum = np.cos(f_i*picon)
            s[i]  = np.sum(w*S*sinsum)
            c[i]  = np.sum(w*S*cossum)
            ss[i] = np.sum(w*sinsum**2)
            cc[i] = np.sum(w*cossum**2)
            sc[i] = np.sum(w*sinsum*cossum)
            # Print compilation time to bash:
            pt.compilation(i, len(f), 'power: with weights')
        print 

    # Calculate alpha, beta, P, A
    alpha = (s*cc-c*sc)/(ss*cc-sc**2)
    beta  = (c*ss-s*sc)/(ss*cc-sc**2)
    P     = alpha**2+beta**2
    A     = np.sqrt(P)

    # Output:
    Pf_data = np.vstack([f, P]).T 
    P_comp  = np.vstack([alpha, beta]).T
    return Pf_data, P_comp, f_interval, f_resolution  



def window(data, f_interval=None, f_resolution=None, sampling=None, w_column=None): 
    """
    This function calculate the so-called window function. The function calls the routine 'power'. It returns the power spectrum of the window function.
    ---------INPUT:
    data         : Is a matrix with a column of time, signal, and possible weights. 
    f_interval    : Frequency interval [fmin, fmax] (optional).
    f_resolution  : Frequency resolution (optional).
    oversampling  : Oversampling of the data (optional).
    w_column      : Column the weights are placed (optional).
    --------OUTPUT:
    Pf_window     : Frequency and power of the window function."""
    print('-------------------------- window')
    
    # Avoid overwritting data:
    data0 = data.copy()

    f_range = round(f_interval[0]+(f_interval[1]-f_interval[0])/2)
    picon   = 2*np.pi*f_range*data[:,0]
    fsin    = np.sin(picon)
    fcos    = np.cos(picon)

    # Sinusoidal
    data0[:,1] = fsin
    Pf_power, _, _, _, = tt.power(data0, f_interval, f_resolution, sampling, w_column)
    f    = Pf_power[:,0]
    Psin = Pf_power[:,1]

    # Co-sinusoidal
    data0[:,1] = fcos
    Pf_power, _, _, _, = tt.power(data0, f_interval, f_resolution, sampling, w_column)
    f    = Pf_power[:,0]
    Pcos = Pf_power[:,1]

    # Output:
    P = 1./2*(Pcos+Psin)
    Pf_window = np.vstack([f, P]).T
    return Pf_window



def clean(data, N_peaks, f_interval=None, f_resolution=None, sampling=None, w_column=None):
    """
    This function indentify and select a sepcified number of highest valued peaks. The peaks are determined with a high accuracy and is then subtracted from the times series. As a output the routine returns this more "clean" time series.  
    -----------INPUT:
    data            : Is a matrix with a column of time, signal, and possible weights. 
    f_interval      : Frequency interval containing [fmin, fmax] (optional).
    f_resolution    : Frequency resolution (optional).
    sampling        : Sampling (oversampling >1) of the data (optional).
    w_column        : Column where the weights are placed (optional).
    ----------OUTPUT:
    St_clean        : Times (t) and a cleaned Signal (S).
    P_comp          : Components of P is alpha and beta.
    f_peaks         : Highest frequency peak value for subtracted peaks."""
    print('-------------------------- clean')
    
    # Avoid overwritting data:
    data0 = data.copy()

    # Standard frequency resolution:
    T = data0[-1,0]-data[0,0]
    if f_resolution==None:
        f_resolution = 1/T
        
    # Avoid 0 as input as not peaks are found:
    if f_interval[0]==0:
        f_interval = [f_resolution, f_interval[1]]
        
    # Constants:
    SAMPLING = 1
    f_RES    = 0.1*f_resolution     # Standard frequency resolution
    picon    = 2*np.pi*data0[:,0]      # Optimization constant
    f_peaks  = np.zeros(N_peaks)
    A_peaks  = np.zeros(N_peaks)
    
    for i in range(N_peaks):
        k = i+1
        print '%s. Peak' %k

        # 1. Iteration - start finding largest peak:
        Pf_power, _, _, _, = tt.power(data0, f_interval, f_resolution, sampling, w_column)
        f     = Pf_power[:,0];   P = Pf_power[:,1];  j = np.nanargmax(P)
        f_int = (f[j-1], f[j+1]) # Smaller f_int (Tuple instead of array for optimization)

        # Testing that the frequency resolution > sigma_f to continue:
        A_peak    = P[j]
        A_av      = np.mean(np.sqrt(P))
        sigma_a   = 0.8*A_av
        sigma_phi = sigma_a/A_peak
        sigma_f   = np.sqrt(3)*sigma_phi/(np.pi*T)
        if f_RES>sigma_f:   
               
            # 2. Iteration: uses now f_res and so on..
            Pf_power, _, _, _, = tt.power(data0, f_int, f_RES, SAMPLING, w_column)
            f     = Pf_power[:,0];   P = Pf_power[:,1];  j = np.nanargmax(P)
            f_int = (f[j-1], f[j+1])
        
            # 3. Iteration: last
            Pf_power, P_comp, _, _, = tt.power(data0, f_int, f_RES, SAMPLING, w_column)
            f = Pf_power[:,0];  P = Pf_power[:,1];  j = np.nanargmax(P)
            fpicon = picon*f[j]  # Optimization constant
            alpha  = P_comp[:,0];  beta = P_comp[:,1]
            alpha0 = alpha[j]*np.sin(fpicon)
            beta0  = beta[j]* np.cos(fpicon)
            data0[:,1] = data0[:,1] - alpha0 - beta0
            f_peaks[i] = f[j]
            A_peaks[i] = np.sqrt(P[j])

    # Output:
    St_clean = data0
    print f_peaks, A_peaks
    return St_clean, f_peaks, A_peaks



def filters(data, f_interval, f_resolution=None, sampling=None, w_column=None):
    """
    This function takes a data set and a frequency interval and calcuates the power spectrum within this interval. The software can be used as a low-pass, band-pass, or a high-pass filter, which i solely determined by the frequency interval.  
    ----------INPUT:
    data           : Is a matrix with a column of time, signal, and possible weights. 
    f_interval     : Frequency interval containing [fmin, fmax].
    f_resolution   : Frequency resolution (optional).
    sampling       : Sampling (oversampling >1) of the data (optional).
    w_column       : Column the weights are placed (optional).
    ---------OUTPUT:
    St_low_band    : Time series for low/band-pass filter.
    St_high        : Time series for high-pass filter."""
    print('-------------------------- filters')

    # Avoid overwritting data:
    data0 = data.copy()
    
    # Avoid 0 as input as not peaks are found:
    if f_interval[0]==0:
        f_interval = [f_resolution, f_interval[1]]
    
    # Calculates power spectrum:
    Pf_power, P_comp, _, _, = tt.power(data0, f_interval, f_resolution, sampling, w_column)
    t     = data0[:,0]
    f     = Pf_power[:,0]
    alpha = P_comp[:,0] 
    beta  = P_comp[:,1]

    # Calculates P_filter:
    P_filter = np.zeros(len(t))
    fpicon = 2*np.pi*f                    # Optimization constant
    for i in range(len(t)):
        tfpicon     = fpicon*t[i]         # Optimization constant
        alpha_sin   = alpha*np.sin(tfpicon)
        beta_cos    = beta* np.cos(tfpicon)
        P_filter[i] = np.sum(alpha_sin + beta_cos)

    # Calculates window function:
    Pf_window = tt.window(data0, f_interval, f_resolution, sampling)
    P_window  = Pf_window[:,1]
    
    # Bandpass/Lowpass and Highpass filter:
    S_low_band  = P_filter/np.sum(P_window)
    S_high      = data0[:,1]-S_low_band
    St_low_band = np.vstack([t, S_low_band]).T
    St_high     = np.vstack([t, S_high]).T
    return St_low_band, St_high 



def moving(filtertype, S0, n): 
    """
    This function can be used to correct for slow trends using a "moving mean" filter. For the median filter, instead of deleting bad data these are replaced by a median value. 
    -----------INPUT:
    filtertype      : Filter is either: median or mean.
    S0              : Signal. 
    n               : Integer used as step size of moving filter.
    ----------OUTPUT:
    S_new           : Filtered signal (S)."""
    print('-------------------------- moving')
    
    # Constants:
    S     = S0.copy()      # Avoid overwritting data:
    S_new = np.zeros(len(S))
    nzero = np.zeros(2*n+1)
    
    # Moving median filter:
    if filtertype=='median':
        print 'Moving median filter'
        # Interval: d[n, 1+n, ... , N-1, N-n]
        for i in range(len(S)-2*n):   
            S_new[n+i] = np.median(S[range((n+i)-n, (n+i)+n+1)])
        for i in range(n):
        # Interval: d[-n, -(n-1), ... , n-1, n] - Low end of data
            low = nzero
            low[range(n-i)] = S[0]*np.ones(n-i)
            low[-(n+1+i):]  = S[range(0, n+1+i)]
            S_new[i]        = np.median(low)
        # Interval: d[N-n, N-(n-1), ... , N+(n-1), N+n] - High end of data
            high = nzero
            high[range(n+1+i)] = S[range(len(S)-(n+i+1), len(S))]
            high[-(n-i):]      = S[-1]*np.ones(n-i)
            S_new[len(S)-1-i]  = np.median(high)

    # Moving mean filter:
    if filtertype=='mean':
        print 'Moving mean filter'
        # Interval: d[n, 1+n, ... , N-1, N-n]
        for i in range(len(S)-2*n):   
            S_new[n+i] = np.mean(S[range((n+i)-n, (n+i)+n+1)])
        for i in range(n):
        # Interval: d[-n, -(n-1), ... , n-1, n] - Low end of data
            low = nzero
            low[range(n-i)] = S[0]*np.ones(n-i)
            low[-(n+1+i):]  = S[range(0, n+1+i)]
            S_new[i]        = np.mean(low)
        # Interval: d[N-n, N-(n-1), ... , N+(n-1), N+n] - High end of data
            high = nzero
            high[range(n+1+i)] = S[range(len(S)-(n+1+i), len(S))]
            high[-(n-i):]      = S[-1]*np.ones(n-i)
            S_new[len(S)-1-i]  = np.mean(high)

    # Output:
    return S_new



def locate(data, n=1, cutoff=5e-4, plot=None): 
    """
    This function can be used to locate bad data points using a "moving median" filter. For the median filter, instead of deleting bad data these are replaced by a median value. 
    -----------INPUT:
    data            : Containing [time, signal].
    n               : Integer used as step size of moving filter (optional).
    cutoff          : Integer limits for the bad data (optional).
    plot            : if plot==1 is plots the outliers (optional).
    ----------OUTPUT:
    [t, S]          : Data corrected for bad data."""
    print('-------------------------- locate')
    
    # Data:
    t = data[:,0]   # Time
    S = data[:,1]   # Signal
    
    # Finding dif:
    S_med = tt.moving('median', S, n)
    dif0  = S/S_med - 1
    
    # Replace median signal if outside cutoff region:
    above    = np.where(dif0>cutoff)[:][:]
    S[above] = S_med[above]
    below    = np.where(dif0<-cutoff)[:][:]
    S[below] = S_med[below]

    # Consistency check for median replacement:
    S_med = tt.moving('median', S, n)
    dif1  = S/S_med - 1
    
    # Plot outliers:
    if plot==1:
        pt.plot_locate(t, dif0, dif1, cutoff, n, 'Time (days)', 'dif')

    # Output:
    return np.vstack([t, S]).T



def jumps(data, gapsize, plot=None):
    """
    This function corrects for jumps in the data larger than "gapsize". Normally ajacent datapoint to a jump is effected, hence, a specified number of points of each side of a jump can be removed.
    ------------INPUT:
    data             : Data containing [time, signal].
    gapsize          : Signal difference at which there will be considered as a jump.
    plot             : Plots the original and corrected data. If plot==1 a plot is made (optional)
    -----------OUTPUT:
    [t, S]           : Data corrected for jumps."""
    print('-------------------------- jumps')
    
    # Unfold the data:
    t      = data[:,0]
    S      = data[:,1].copy()

    # Find distances/difference between data points:
    S_diff = np.diff(S)
    
    # Find gaps:
    index = np.where(abs(S_diff)>gapsize)[0][:]
    
    # Move the data when a jump:
    for i in index:
        S[i + 1:] -= S_diff[i]
    
    # Plot outliers:
    if plot==1:
        pt.plot_gapsize(S_diff, 'diff', 'log(N  of  diff)')
        pt.plot_jumps(t, data[:,1], S, 'Time (days)', 'Signal')

    # Output:
    return np.vstack([t, S]).T



def stellar_noise(data, f_int, sampling, N, plot=None):
    """
    This function clean the power spectrum for long period variations from the stellar host or other long term variations. 
    ------------INPUT:
    data             : Containing [time, signal].
    f_int            : Frequency interval that should be cleaned/filtered.
    sampling         : Sampling of the data used in clean.
    N                :  Numbers of peaks to be removed by clean.
    plot             : Plot power spectrum and timeseries if plot is 1 (optional).
    -----------OUTPUT:
    St               : Corrected time series [time, Signal]. """
    print('-------------------------- stellar noise')
    
    # Run functions:
    f_int0   = [0, 1]
    Pf_power, _, _, f_res = tt.power(data, f_int0, None, sampling)
    St, _, _,             = tt.clean(data, N,  f_int, f_res)
    Pf, _, _, _,          = tt.power(St, f_int0, None, sampling)

    # Plot outliers:
    if plot==1:
        fx_int = [0, 0.5]; fy_int = [0, 2e6]
        pt.plot_stellarnoise_power(Pf_power, Pf, fx_int, fy_int,'Frequency $(c/d)$','Power')
        pt.plot_stellarnoise_time(data, St, 'Time (days)', 'Signal')

    # Output:
    return St



def slowtrend(data, gapsize, n=25, m=25, jump=None, plot=None):
    """
    This function corrects for slow trends in the time series.
    -----------INPUT:
    data            : Data containing [time, signal].
    gapsize         : Signal difference at which there will be considered as a jump.
    n and m         : Integer used in moving median and mean filter. n=m=25 if None (optional).
    jump            : If jump==1 the data is filtered with a n=2 median filter 'm' points of each                       side around a jump in time that is bigger than 'gapsize' (optional).
    ----------OUTPUT:
    data_new        : Data corrected for slow trends."""
    print('-------------------------- slowtrend')

    # Splitting data:
    t = data[:,0]
    S = data[:,1].copy()
    
    # Median and mean filters:
    S_medi  = tt.moving('median', S,      n)
    S_medi2 = tt.moving('median', S,      2)
    S_mean  = tt.moving('mean'  , S_medi, m)

    # Use median filter m points of each side around jumps:
    if jump==1:
        t_diff = np.diff(t)                              # Find difference between data points
        index  = np.where(np.abs(t_diff)>gapsize)[0][:]  # Find gap indices
        mcon   = range(-m, 1+m)                          # Optimization constant range
        for i in index:
            k = mcon+(1+i)*np.ones(len(mcon))  
            for j in k:
                S_mean[int(j)] = S_medi2[int(j)]
    
    # Correction:
    S_new  = S/S_mean
 
    # Data:
    data1 = np.vstack([t, S_medi]).T 
    data2 = np.vstack([t, S_mean]).T
    data3 = np.vstack([t, S_new]).T
    
    # Plot outliers:
    if plot==1:
        pt.plot_slowtrend(data, data1, data2, data3, n, m, 'Time (days)', 'Signal')
        pt.PLOT(data3[:,0], [data3[:,1]], 'b-', '$t$ [days]', '$S$', 'Corrected Lightcurve')

    #Output:
    return data3



def model(t, P, phi, dT, A): 
    """
    This function is used to create a transit model for the cross-correlation. 
    ----------INPUT:
    t              : Time series with time (t) in [days] and signal (S)
    P              : Period
    phi            : Phase
    dT             : Transit duration
    A              : Scale amplitude
    """
    
    # Model created:
    Model = np.zeros(len(t))
    N_t   = np.ceil(t[-1]/P)   # Max number of transits
    for m in np.arange(N_t+1):
        model        = np.abs(t - m*P - phi) < dT/2.
        Model[model] = A
    return Model



def cc_coefficient(x, y): 
    """
    This function find the cross-correlation coefficienten between two datasets. 
    ----------INPUT:
    x              : Signal from 1. data set.
    y              : Signal from 2. data set.
    """
    cor  = np.sum( (x-np.mean(x)) * (y-np.mean(y)) )
    norm = sqrt( np.sum((x-np.mean(x))**2) * np.sum((x-np.mean(x))**2) )
    r    = cor/norm
    return r



def crosscor(data, P_int, phi_int=None, model_const=None, save=None): 
    """
    This function is used to create a transit model for the Cross-Correlation (CC). To create the model the subrutine called 'model' is used. To perform the CC the CC coefficient is also needed and this is calculated in the subroutine 'cc_coefficients'.
    ----------INPUT:
    data           : Time series with time (t) in [days] and signal (S).
    P_int          : Periode interval to perform CC [hours].
    phi_int        : Phase interval to perform CC [hours] (optional).
    model_const    : Settings for model [dT, dP, dPhi] all in [min] (optional).
    save           : If 1 the correlation is saved to data file with user defined name (optional)."""
    print('-------------------------- crosscor')
    start_time = time.time()      # Take time

    # Load functions:
    from Timeseries_Tools import model, cc_coefficient
    from Plot_Tools import SURF, plot_period_cc

    # Unpack data:
    t = data[:,0]*24.*60.       # Time [min]
    S = data[:,1].copy()        # Save copy
 
    # Normalized signal MUST be used:
    x = (1-S)-mean(1-S)
 
    # Model parameters:
    if model_const==None:
        dT   = 120.                 # Transit duration  [min]
        dP   = 5.                   # Period resolution [min]
        dphi = 30.                  # Phase  resolution [min]
        A    = 1.                   # Scale amplitude
    else:
        dT   = model_const[0]       # [min]
        dP   = model_const[1]       # [min]
        dphi = model_const[2]       # [min]
        A    = model_const[3] 

    # Parameter space:
    P_int   = array(P_int)*60.             # Period interval [min]
    if phi_int==None:
        phi_int = [0, int(P_int[1]+dphi)]  # Phase interval [min] (Default: all possible)
    else:
        phi_int = array(phi_int)*60.       # Phase interval  [min]
        
    # Tested period-grid:
    P     = arange(P_int[0], P_int[1], dP)           # Period range [min]
    phi   = arange(phi_int[0], phi_int[1], dphi)     # Phase  range [min]
    
    # Dimentions:
    N = len(P) 
    M = len(phi) 
    print 'CC matrix = N(P) x M(phi) = {} x {} = {}'.format(N, M, N*M)
    
    # Preparations for cross-correlation:
    cc    = zeros((N, M))                           # N*M matrix with zeros for amplitudes
    Model = zeros(len(t))                           # 0-array for model data     
    
    # Range of highest number of transits given by P_int:
    N_max = range(1, int(t[-1]/P_int[0])+2)

    # Perform Cross-correlation:
    for i in range(N):
        for j in range(M):
            y        = model(t, P[i], phi[j], dT, A)
            r_cc     = cc_coefficient(x, y)
            cc[i, j] = r_cc
        # Keep track on time:
        if i==round(N*0.25):
            print ('Done 25  procent --- %s seconds ---' % (time.time() - start_time))       
        if i==round(N*0.50):
            print ('Done 50  procent --- %s seconds ---' % (time.time() - start_time))       
        if i==round(N*0.75):
            print ('Done 75  procent --- %s seconds ---' % (time.time() - start_time))       
        if i==round(N-1):
            print ('Done 100 procent --- %s seconds ---' % (time.time() - start_time))       
       
    # Best P and Phi by cross-correlation:
    cc_max   = max(cc)
    cc_max_i = where(cc==cc_max)
    h = 60.;     P_hour = P[cc_max_i[0][0]]/h; phi_hour = phi[cc_max_i[1][0]]/h
    d = 60.*24.; P_days = P[cc_max_i[0][0]]/d; phi_days = phi[cc_max_i[1][0]]/d
    print 'Best Period: {:.6f} hours and {:.6f} days'.format(P_hour, P_days)
    print 'Best Phase : {:.6f} hours and {:.6f} days'.format(phi_hour, phi_days)
    print 'P resolution  : {:.4f}'.format(dP/(60.*24.))
    print 'phi resolution: {:.4f}'.format(dphi/(60.*24.))
    
    # Save result?
    if save!=None: 
        savetxt(save[0], P)
        savetxt(save[1], phi)
        savetxt(save[2], cc)
        
    return

            

def autocor(data, npeaks, cutoff_x, cutoff_y, plot=None):
    """----------------------------------- FUNCTION ---------------------------------------:    
    # This function performs a auto-correlation on the data signal.
    #-----------INPUT:
    # data           : Time series time [mins] and signal. 
    # npeaks         : Number of peaks to be found in ACF by the 'argrelmax' function from Scipy.
    # cutoff_x       : Time range where peaks should be used.
    # cutoff_y       : ACF range where peaks should be used.
    # plot           : If plot==1 the data is plotted."""
    print '------------------------------------------------------------------------- autocor'
    
    # Load functions:
    from scipy.interpolate import griddata     # Linear interpolation
    from scipy.signal import argrelmax         # Peak location
    from scipy import stats                    # Uncertainty to linear fit
    from Plot_Tools import PLOT

    # Avoid overwritting data:
    t = data[:,0].copy()*24*60         # [min]
    S = 1 - data[:,1].copy()    # Normalized signal MUST be used

    # Plot corrected timeseries:
    if plot==1:
        data0 = vstack([t/(60.*24.), S]).T
        PLOT([data0], 'b-', '$t$ [days]', '$1-S$', 'Correected data')
        
    
    # Interpolate to uniform grid:
    dt = median(diff(t))
    tt = arange(t[0], t[-1]+dt, dt)
    SS = griddata(t, S, tt, method='nearest')   # Function from scipy libiary

    # Remove points below 0:
    SS[SS<0] = 0.0

    # Plot interpolated data:
    if plot==1:
        data1 = vstack([tt/(60.*24.), SS]).T
        PLOT([data0, data1], ['b-','k.'], '$t$ [days]', '$1-S_{grid}$', 'Interpolation')
    
    # Prepare auto-correlation:
    N   = len(tt)
    acf = zeros(N)

    # Perform auto-correlation:
    for i in range(1, N):
        S_stat  = SS[i:]  - mean(SS[i:])                   # Stationary grid to be correlated
        S_move  = SS[:-i] - mean(SS[:-i])                  # Moving the stationary grid by i
        acf[i]  = sum(S_stat*S_move)/(float(N)-float(i))   # Correlates the two grides    
    # Auto-correlation for i=0:
    S_stat  = SS - mean(SS)
    acf[0]  = (1./float(N))*sum(S_stat**2)                 # Correlates the two grides

    # Find peaks in ACF and find location and height:
    peaks_i = argrelmax(acf, order=npeaks)         
    peaks_x = tt[peaks_i]
    peaks_y = acf[peaks_i]
    
    # Peaks inside y-range:
    peaks_keep = peaks_y > cutoff_y[0]
    peaks_x    = peaks_x[peaks_keep]
    peaks_y    = peaks_y[peaks_keep]
    peaks_keep = peaks_y < cutoff_y[1]
    peaks_x    = peaks_x[peaks_keep]
    peaks_y    = peaks_y[peaks_keep]
    peak0      = peaks_x[0]
    
    # Peaks inside x-range:
    peaks_keep = peaks_x > cutoff_x[0]*24.*60.  # cutoff_y is given in days
    peaks_x    = peaks_x[peaks_keep]
    peaks_y    = peaks_y[peaks_keep]
    peaks_keep = peaks_x < cutoff_x[1]*24.*60. 
    peaks_x    = peaks_x[peaks_keep]
    peaks_y    = peaks_y[peaks_keep]
         
    # Plot autocorrelation:
    if plot==1:
        data0 = vstack([tt/(60.*24.), acf]).T
        data1 = vstack([peaks_x/(60.*24.), peaks_y]).T
        PLOT([data0, data1], ['b-', 'k.'], '$t_{grid}$ [days]', '$ACF$', 'Auto-correlation')

    # Measured periods:
    P_mea = peaks_x - t[0]          # Remove time-offset
    P     = mean(diff(peaks_x))     # Mean period value

    # This is an comtempolary solution: When the cutoff_x[0]>0 the linear plot do no work since
    # it needs the 1. detected transit, hence, one have to count N*P to the first peak that's
    # inside the range cutoff_x. Here 6 period is evident before the first inside cutoff_x:
    #P_mea = P_mea - 6*P
     
    # Uncertainty:
    P_std = std(diff(peaks_x))
    
    # Calculate periods:
    P_cal = zeros(len(P_mea))
    for i in range(1, len(P_mea)+1):
        P_cal[i-1] = i*P-P
        
    # Find period by linear regression:
    coef, stats = np.polynomial.polynomial.polyfit(P_cal, P_mea, 1, full=True)
    b     = coef[0]
    a     = coef[1]
    p     = arange(0., P_cal[-1], 1.)
    P_fit = a*p + b
  
    # Write out the estimated period:
    print 'Best: P = {} +/- {} hours'.format(b/60., P_std/60.)
    print 'Best: P = {} +/- {} days'.format(b/(60.*24.), P_std/(60.*24.))
    print 'Mean: P = {} +/- {} days'.format(P/(60.*24.), P_std/(60.*24.))
    
    # Plot linear fit:
    if plot==1:
        data0 = vstack([P_cal/(60.*24.), P_mea/(60.*24.)]).T
        data1 = vstack([p/(60.*24.), P_fit/(60.*24.)]).T
        PLOT([data0, data1], ['go','k-'], \
             r'Calculated: $N \times P-P$ [days]', r'Measured: $N \times P$ [days]', \
             'ACF Period: ${:.4f} \pm {:.4f}$ days'.format(b/(60.*24.), P_std/(60.*24.)))
    
    return 

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os
import time

#casename = np.array(['r06'             ,  'r06'])
#casename = np.array(['r06_M2'          , 'r06M2'])
#casename = np.array(['r08_Coarse'      , 'r08C'])
#casename = np.array(['r08_Fine'        , 'r08F'])
#casename = np.array(['r10'             , 'r10'])
casename = np.array(['r10_short_Coarse', 'r10SC'])
#casename = np.array(['r10_short_Fine'  , 'r10SF'])
# path2root = f"/home/duncan/remote/hal/Results/Plate_{casename[0]}/DMD/test_noise"
path2root = f"/home/duncan/remote/hal/Results/Plate_{casename[0]}/DMD/New_FWH_Guilherme"

dirs = sorted(np.array([d for d in os.listdir(path2root) if os.path.isdir(os.path.join(path2root, d))]), reverse=True)
# files = np.full((30,), '', dtype='U100')
# files[:] = [f for f in os.listdir(os.path.join(path2root, dirs[0])) if os.path.isfile(os.path.join(path2root,dirs[0], f))]
files = [f for f in os.listdir(os.path.join(path2root, dirs[0])) if os.path.isfile(os.path.join(path2root,dirs[0], f))]


def Calc_PSD(data, nfft = 1024 ,overlap = 0.75, D = 0.05, U = 137.2, window = 'Hanning'):
    """
    Computes the Power Spectral Density (PSD) in and returns it vs Strouhal number.

    Parameters:
        data   : np.ndarray with shape (N, 2), where data[:,0] is time and data[:,1] is pressure.
        nfft   : int, number of points per FFT window.
        overlap: float, fraction of overlap between windows.
        D      : float, characteristic length (e.g. jet diameter).
        U      : float, reference velocity (e.g. jet velocity).

    Returns:
        St     : 1D array of Strouhal numbers.
        PSD    : 1D array of PSD values in dB re 20 µPa^2/Hz.
    """
    # Start timer
    start_time = time.time()
    #
    # Cálculo de parâmetros
    nt = data.shape[0]
    print(nt)
    dt = data[1,0]-data[0,0]
    Fs = 1/dt
    nOvlp = overlap*nfft
    nBlks = int(np.floor((nt-nOvlp)/(nfft-nOvlp)))
    #print('     novelp: ',nOvlp)
    print('     nBlks : ',nBlks)
    #
    # Janela de Hanning e peso da janela
    if window == 'Hanning':
        print('     Hanning window')
        winf = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(nfft)) / (nfft - 1)))
        winWeight = 8 / 3
    else:
        # Janela retangular (n tem janela pq tp né...)
        print('     No window')
        winf      = np.ones(nfft)
        winWeight = 1 
    #
    # Incializando
    spec = np.zeros(nfft)
    #
    for iBlk in range(nBlks):
        it_end = int(min(iBlk*(nfft-nOvlp)+nfft, nt))
        it_start = int(it_end - nfft)
        #
        # Pressão já é acústica, então teoricametne não precisa tirar a flutuação?
        spec += abs(np.fft.fft(data[it_start:it_end,1]*winf)/nfft)**2
    #
    # Cálculo da média
    #print('     spec size:',spec.shape)
    spec /= nBlks
    #
    # Criando eixos das frequências
    freq = Fs * np.arange(nfft) / nfft
    St   = freq*D/U
    dSt  = St[1]- St[0]
    dFr  = freq[1] - freq[0]
    #
    #
    spec2 = 2*spec*winWeight/dSt        # Pa²/St
    #print('     spec2 size:',spec2.shape)
    PSD   = 10*np.log10(spec2/(4e-10))  # pref = 2e-5; pref² = 4e-10
    #
    # End timer
    end_time = time.time()
    delta_t  = (end_time-start_time) # [sec]
    #
    # Relatório
    # print('-------------------------------'   )
    # print('PSD Finished:'                     )
    # print('-------------------------------'   )
    # print('Relatório:'                        )
    # print(f'    run time: {(delta_t):.2f} s'  )
    # print(f'    deltaT : {dt:.2e}'            )
    # print(f'    nt     : {nt}'                )
    # print(f'    deltaFr: {dFr:.2f} Hz'        )
    # print(f'    deltaSt: {dSt:.2f}'           )
    # print(f'    fmax   : {freq[-1]:.2f} Hz'   )
    # print(f'    Stmax  : {St[-1]:.2f}'        )
    # print('-------------------------------\n' )
    #
    return St, PSD

# Start with tab10 colormap (nice 10-color cycle)
colors = list(plt.cm.tab10.colors)

# Define RGB tuples you want to block (red, blue, orange from tab10)
blocked = [
    (1.0, 0.0, 0.0),   # pure red, just in case
    colors[0],         # tab10 blue
    colors[2],         # tab10 green
    colors[3],]        # tab10 red


# Filter out blocked colors
colors = [c for c in colors if c not in blocked]

# Apply new cycle (no red, no orange, no blue)
plt.gca().set_prop_cycle(color=colors)

for iFile in files:

    PSD_original = np.load(os.path.join(f"/home/duncan/Vinicius_IC/PSD_original/Plate_{casename[0]}",f"PSD_{iFile}"))
    plt.plot(PSD_original[:int(len(PSD_original[:,0])/2-1),0],PSD_original[:int(len(PSD_original[:,0])/2-1),1], linestyle = '-', color = 'red', label = 'Num solid', zorder = 1, linewidth = '2.5')
    plt.legend()

    for i, iDir in enumerate(dirs):
        data = np.load(os.path.join(path2root,iDir, iFile))
    # for i in range (10):
    #     data = np.load(os.path.join(path2root,dirs[i], iFile))
        signal = data[:,1].T
        dt = data[1,0] - data[0,0]


        # Compute FFT
        St, psd = Calc_PSD(data, nfft = 1024, overlap=0.5, window='Hanning')
        # signal_fft = fft(signal)/len(signal)
        # n = len(signal)  # Length of signal
        # freq = np.fft.fftfreq(n, d = dt)  # Frequency bins

        # Keep only the positive frequencies and their magnitudes
        D = 0.05
        U = 137.2
        # positive_freq_indices = np.where(freq >= 0)
        # frequencies = freq[positive_freq_indices]
        # St = frequencies*D/U
        # magnitudes = signal_fft[positive_freq_indices]
        # for j in range(len(St)-1):
        #     dSt = St[j+1] - St[j]
        #     magnitudes[j] = magnitudes[j]/dSt
        # # print(magnitudes)
        # dSt = St[-1] - St[-2]
        # magnitudes[-1] = magnitudes[-1]/dSt
        # signal_rms = np.abs(magnitudes)/np.sqrt(2)
        # signal_db = 20*np.log10(signal_rms/(2e-5))  # pref = 2e-5; pref² = 4e-10

        # Plot FFT
        imode  = iDir.split('_')[-1]
        # imode  = dirs[i].split('_')[-1]
        # if imode != '5':
        # plt.plot(St, signal_db, label= f'{imode}%')
        if imode == 'forcedturbo1000m':
            plt.plot(St[:int(len(psd)/2-1)], psd[:int(len(psd)/2-1)], label= f'{imode}', linestyle = '-.', color = 'green', zorder = 20, linewidth = '2.5')
            plt.legend()
        elif imode == 'conditioned':
            plt.plot(St[:int(len(psd)/2-1)], psd[:int(len(psd)/2-1)], label= f'{imode}', linestyle = '-.', color = 'blue', zorder = 20-i) #sem a linha ficar mais clara
            plt.legend()
        else:
            # plt.plot(St[:int(len(psd)/2-1)], psd[:int(len(psd)/2-1)], label= f'{imode}%', linestyle = '--', zorder = 20-i, alpha = (0.875-0.0625*i))
            plt.plot(St[:int(len(psd)/2-1)], psd[:int(len(psd)/2-1)], label= f'{imode}', linestyle = '--', zorder = 20-i) #sem a linha ficar mais clara
            plt.legend()
    
    name = iFile.split('.')[0]
    # plt.title(f'{name}')
    plt.semilogx()
    # plt.xlim(0.1,1.6)
    # plt.ylim(55,110)
    plt.grid(axis = 'y')
    # plt.axvline(x = 0.7865461318900323, label = '0.7865461318900323', linestyle = '--')
    plt.legend()
    plt.xlabel('St', fontsize = 16)
    plt.ylabel('SPL[dB/St]', fontsize = 16)
    # fontSize = 10
    # plt.legend(fontsize = fontSize)
    plt.show()
    # plt.savefig('./Observador_DMD/Obs_theta90_phi260')
    # break
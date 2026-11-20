# Bibliotecas
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import timeit # Similar o time
from numba import jit # type: ignore # Optmize numerical computations (deixando uma linguagem de baixo nível)



# Outras bibliotecas
@jit(nopython=True)
def compute_scalar_product(array1, array2, axes):
        """Computes the scalar product of 2 arrays."""
        #temp = array1*array2
        #result = np.sum(temp, axis=axes-1)
        return np.sum(array1*array2)


#@jit(nopython=True)
def Compute_RadiationVec(analogy, surface_type, surface_geo, nDim, coord, normals, Observer_Locations, nPanel, nObserver, FreeStreamMach):

    """Compute radiation vector form source to observers."""

    Beta2 = 1.0 - FreeStreamMach**2.0

    for iPanel in range(nPanel):

        x = coord[iPanel,0]
        y = coord[iPanel,1]
        z = 0.0
        if ( nDim == 3):
            z = coord[iPanel,2]

        nx = normals[iPanel,0]
        ny = normals[iPanel,1]
        nz = 0.0
        if (nDim == 3):
            nz = normals[iPanel,2]

        #CheckNormal =  x*nx+y*ny+z*nz
        #if (CheckNormal<0):
        #	nx = -nx
        #	ny = -ny
        #	nz = -nz
        dS = normals[iPanel,3]
        #Area_Factor = 1.0 # ??? NÃO ENTENDI ??? As normais já estão escaladas pela área da célula correspondente?

        surface_geo[iPanel,0] = x
        surface_geo[iPanel,1] = y
        surface_geo[iPanel,2] = nx
        surface_geo[iPanel,3] = ny
        surface_geo[iPanel,4] = dS # dS é a área teoricamente (?)  
        #surface_geo[iPanel,4] = Area_Factor*dS
        
        if ( nDim == 3):
            surface_geo[iPanel,5] = z
            surface_geo[iPanel,6] = nz

        for iObserver in range(nObserver):
            r1 = Observer_Locations[iObserver,0]-x
            r2 = Observer_Locations[iObserver,1]-y
            r3 = Observer_Locations[iObserver,2]-z

            if (analogy == '1A' and surface_type == 'solid'):
                
                r_mag = np.sqrt(r1*r1+r2*r2+r3*r3)
                r1    = r1/r_mag
                r2    = r2/r_mag
                r3    = r3/r_mag

            elif (analogy == '1A_WT' and surface_type == 'solid'): # Conclusão, método 1A e 1A_WT são aparentemente iguais caso u0 = 0
                
                r_star = np.sqrt(r1**2 + Beta2 * (r2**2 + r3**2))               # if u0 = 0: r_star = r1**2 + r2**2 + r3** = r_mag (igual ao 1A)
                r_mag  = (-FreeStreamMach*r1 + r_star) / Beta2                  # if u0 = 0: r_mag  = r_star/1             = r_mag (igual ao 1A)
                r1     = (-FreeStreamMach*r_star + r1 ) / (Beta2 * r_mag)       # if u0 = 0: r1     = r1/r_mag             = r_1   (igual ao 1A)
                r2     = r2/r_mag                                               # if u0 = 0: r2     = r2/r_mag             = r_2   (igual ao 1A)
                r3     = r3/r_mag                                               # if u0 = 0: r3     = r3/r_mag             = r_3   (igual ao 1A)


            surface_geo[iPanel,7+iObserver*4]  = r1
            surface_geo[iPanel,8+iObserver*4]  = r2
            surface_geo[iPanel,9+iObserver*4]  = r3
            surface_geo[iPanel,10+iObserver*4] = r_mag

    return surface_geo

@jit(nopython=True)
def Extract_NoiseSources(Fr, data, nSample, nPanel, nObserver, FreeStreamPressure, surface_geo):
    # Hierarquia: TEMPO -> CÉLULA -> OBSERVADOR

    for iSample in range(nSample):
        # Feito para cada intervalo de tempo
        for iPanel in range(nPanel):
            # Feito para cada painel/célula

            # compute monopole and dipole source terms on the FWH surface

            # Recebendo as normais
            nx    = surface_geo[iPanel,2]
            ny    = surface_geo[iPanel,3]
            nz    = surface_geo[iPanel,6]

            # Calcula os versores
            n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
            nx /= n_mag
            ny /= n_mag
            nz /= n_mag

            # Recebendo os valores de pressão
            p = data[iSample,iPanel]

            # Cálculo versão 1:
            # rho=0.0; ux = 0.0; uy = 0.0; uz = 0.0
            
            #F1= rho*(ux*nx+uy*ny+uz*nz)*(ux)+(p-FreeStreamPressure)*nx
            #F2= rho*(ux*nx+uy*ny+uz*nz)*(uy)+(p-FreeStreamPressure)*ny
            #F3= rho*(ux*nx+uy*ny+uz*nz)*(uz)+(p-FreeStreamPressure)*nz
            
            # Cálculo versão 2: 
            F1 = (p-FreeStreamPressure)*nx
            F2 = (p-FreeStreamPressure)*ny
            F3 = (p-FreeStreamPressure)*nz
            

            for iObserver in range(nObserver):
                # Para cada observador
                Fr[iObserver,iPanel,iSample]= F1*surface_geo[iPanel,7+iObserver*4]+F2*surface_geo[iPanel,8+iObserver*4] + F3*surface_geo[iPanel,9+iObserver*4]
                
                
    return Fr

@jit(nopython=True)
def Extract_Mean(Fr, Fr_mean, nSample, nPanel, nObserver):

    for iPanel in range(nPanel):
        for iSample in range(nSample):
            for iObserver in range(nObserver):
                Fr[iObserver,iPanel,iSample] = Fr[iObserver,iPanel,iSample] - Fr_mean[iObserver,iPanel]
    return Fr

@jit(nopython=True)
def Compute_RetardedTime(pp_ret, Fr, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, a_inf):

    dt_sampling = dt * SamplingFreq

    for iObserver in range(nObserver):
        for iPanel in range(nPanel):

            dS = surface_geo[iPanel,4]
            for iSample in range(nSample):
                
                # Cálculo de d(Fr)/dt baseado em diferenças finitas de segunda ordem (!não tenho mt certeza sobre!)
                if (iSample==0):
                    Fr_dot = (-Fr[iObserver,iPanel,iSample+2]+4.0*Fr[iObserver,iPanel,iSample+1]-3.0*Fr[iObserver,iPanel,iSample])/2.0/dt_sampling
                elif (iSample==nSample-1):
                    Fr_dot = (3.0*Fr[iObserver,iPanel,iSample]-4.0*Fr[iObserver,iPanel,iSample-1]+Fr[iObserver,iPanel,iSample-2])/2.0/dt_sampling
                else:
                    Fr_dot = (Fr[iObserver,iPanel,iSample+1]-Fr[iObserver,iPanel,iSample-1])/2.0/dt_sampling

                first  = Fr_dot/(surface_geo[iPanel,10+iObserver*4]*a_inf)
                second = Fr[iObserver,iPanel,iSample]/(surface_geo[iPanel,10+iObserver*4]**2)

                pp_ret[iObserver,iPanel,iSample] = first + second           
                pp_ret[iObserver,iPanel,iSample] = pp_ret[iObserver,iPanel,iSample]*dS/(4.0*np.pi)
                                
    return pp_ret

@jit(nopython=True)
def Compute_RetardedTime_WT(pp_ret, data, surface_geo, nSample, nPanel, nObserver, FreeStreamPressure, FreeStreamMach, M_i, a_inf, dt, SamplingFreq):

    Lr = np.zeros((nObserver, nPanel, nSample), dtype = np.float32)
    Lm = np.zeros((nPanel,nSample), dtype = np.float32)

    for iSample in range(nSample):
        for iPanel in range(nPanel):
            # compute monopole and dipole source terms on the FWH surface

            nx = surface_geo[iPanel,2]
            ny = surface_geo[iPanel,3]
            nz = surface_geo[iPanel,6]

            p = data[iSample,iPanel]

            L1= (p-FreeStreamPressure)*nx
            L2= (p-FreeStreamPressure)*ny
            L3= (p-FreeStreamPressure)*nz
            
            L_temp = np.array([L1,L2,L3])

            for iObserver in range(nObserver):
                R_temp = np.array([surface_geo[iPanel,7+iObserver*4], surface_geo[iPanel,8+iObserver*4], surface_geo[iPanel,9+iObserver*4]])
                Lr[iObserver,iPanel,iSample] = compute_scalar_product(L_temp, R_temp, 3)

            Lm[iPanel,iSample] = compute_scalar_product(L_temp, M_i, 3)

    dt_sampling = dt * SamplingFreq

    for iObserver in range(nObserver):
        for iPanel in range(nPanel):
            dS = surface_geo[iPanel,4]
            for iSample in range(nSample):
                
                if (iSample==0):
                    Lr_dot = (-Lr[iObserver,iPanel,iSample+2]+4.0*Lr[iObserver,iPanel,iSample+1]-3.0*Lr[iObserver,iPanel,iSample])/2.0/dt_sampling
                elif (iSample==nSample-1):
                    Lr_dot = (3.0*Lr[iObserver,iPanel,iSample]-4.0*Lr[iObserver,iPanel,iSample-1]+Lr[iObserver,iPanel,iSample-2])/2.0/dt_sampling
                else:
                    Lr_dot = (Lr[iObserver,iPanel,iSample+1]-Lr[iObserver,iPanel,iSample-1])/2.0/dt_sampling

                
                R_temp = np.array([surface_geo[iPanel,7+iObserver*4], surface_geo[iPanel,8+iObserver*4], surface_geo[iPanel,9+iObserver*4]])
                Mr = compute_scalar_product(M_i,R_temp,3)

                R = surface_geo[iPanel,10+iObserver*4]
                
                pp_ret[iObserver,iPanel,iSample] = Lr_dot / (a_inf * R * (1.0 - Mr)**2) + \
                                                    (Lr[iObserver,iPanel,iSample] - Lm[iPanel,iSample])/(R**2 * (1.0 - Mr)**2) + \
                                                    (Lr[iObserver,iPanel,iSample]*(Mr - FreeStreamMach**2))/(R**2 * (1.0 - Mr)**3)

                pp_ret[iObserver,iPanel,iSample] = pp_ret[iObserver,iPanel,iSample]*dS/(4.0*np.pi)

    return pp_ret


@jit(nopython=True)
def Compute_ObserverTime( t_interp, t_Obs, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, start_iter, a_inf):

    r_minmax = np.zeros((nObserver,2))
    for iObserver in range(nObserver):
        r_min = 10.0e31
        r_max = 0.0
        for iPanel in range(nPanel):
            r = surface_geo[iPanel,10+iObserver*4]
            if (r>r_max):
                r_max = r
            if (r<r_min):
                r_min = r

            for iSample in range(nSample):
                t_src = dt*(start_iter+iSample*SamplingFreq)
                t_Obs[iObserver,iPanel,iSample]=t_src + r/a_inf

        r_minmax[iObserver,0]= r_min
        r_minmax[iObserver,1]= r_max


    for iObserver in range(nObserver):
        t_interp_start = dt*(start_iter)+r_minmax[iObserver,1]/a_inf
        t_interp_end   = dt*(start_iter+nSample*SamplingFreq-1)+r_minmax[iObserver,0]/a_inf
        dt_interp = (t_interp_end - t_interp_start)/(nSample-1)

        for iSample in range(nSample):
            t_interp[iObserver,iSample] = t_interp_start + dt_interp*iSample

    return t_interp, t_Obs

#@jit(nopython=True)
def Interp_PressureSignal(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver):
    
    for iObserver in range(nObserver):
        for iPanel in range(nPanel):
            aux_x = t_Obs[iObserver,iPanel,:]
            aux_y = pp_ret[iObserver,iPanel,:]			
            f = interp1d(aux_x, aux_y, kind = 'cubic', bounds_error=False)
            pp_interp[iObserver,iPanel,:] = f(t_interp[iObserver,:])
    return pp_interp

@jit(nopython=True)
def Integrate_Sources(pp_TimeDomain, pp_interp, nSample, nPanel, nObserver):

    for iObserver in range(nObserver):
        for iSample in range(nSample):
            for iPanel in range(nPanel):
                pp_TimeDomain[iObserver,iSample] = pp_TimeDomain[iObserver,iSample] + pp_interp[iObserver,iPanel,iSample]
    return pp_TimeDomain

# -------------------------------------------------
# ------------------ MINHA VERSÃO -----------------
# -------------------------------------------------


def main(
        pressure, 
        Observer,
        coord,
        normals,
        FreeStreamVar,
        dt,
        nDim = 3,
        SamplingFreq = 1,
        FWH_analogy = '1A',
        surface_type = 'solid',
        start_iter = 0,
        name = 'Mean_values.npy',
        name2save = 'Observer_noise.npy'):
    
    print ("\nFfowcs-Williams & Hawkings Analogy Solver")

    # Variáveis do escoamento livre
    FreeStreamPressure = FreeStreamVar[0] 
    FreeStreamVelocity = FreeStreamVar[1] 
    a_inf              = FreeStreamVar[2] # Velocidade do som 


    # DEBUG
    print('\nFreestream variables')
    print(f'    p0 : {FreeStreamPressure}')
    print(f'    u0 : {FreeStreamVelocity}')
    print(f'    c0 : {a_inf}')

        
    # Mach do escoamento livre     
    FreeStreamMach = FreeStreamVelocity/a_inf
    print(f'    m0 : {FreeStreamMach}') # DEBUG


    # Algumas outras variáveis auxiliares
    U0_i = np.array([FreeStreamVelocity,0.0,0.0]) # Vetor velocidade (U0, 0, 0) 
    M_i = np.array([-U0_i[0]/a_inf,-U0_i[1]/a_inf,-U0_i[2]/a_inf]) # Vetor mach (-U0/c, -0, -0) 

    # DEBUG
    print('\nVetores velocidade/Mach')
    print(f'    U0 : {U0_i}')
    print(f'    Mi : {M_i}')

    # Load observers
    nObserver = Observer.shape[0] # (nObs x 3)

    nSample   = pressure.shape[0]
    nPanel    = pressure.shape[1]

    # DEBUG
    print('\nNumber of things:')
    print(f'    nObs : {nObserver}')
    print(f'    nSam : {nSample}')
    print(f'    nPan : {nPanel}')


    # Now following Beckett implementation:
    # Allocating some variables # (float32)
    surface_geo        = np.zeros((nPanel,2*nDim+1+4*nObserver), dtype = np.float32)
    pp_ret             = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32) # RAM demanding here
    pp_interp          = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32) # RAM demanding here
    t_Obs              = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32) # RAM demanding here
    t_interp           = np.zeros((nObserver,nSample)                  , dtype = np.float32) 
    pp_TimeDomain      = np.zeros((nObserver,nSample)                  , dtype = np.float32) 
    
    fwh_start = timeit.default_timer()

    # Calculando os vetores de irradição/radiação
    print ('1. Creating surface metrics and radiation vector.\n')
    surface_geo = Compute_RadiationVec(FWH_analogy, surface_type, surface_geo, nDim, coord, normals, Observer, nPanel, nObserver, FreeStreamMach)


    # Prints que parecem servir para debbug (possivelmente posso tirar dps)
    print ('    Observer, Radiation vector (min), Radiation vector (max)')
    for iObserver in range(nObserver):
        print ('    ', iObserver, surface_geo[:,10+iObserver*4].min(), surface_geo[:,10+iObserver*4].max())

    # Qual analogia usar
    # Será usado as outras funções desse código 
    if FWH_analogy == '1A':
        print ('2.1 Extracting noise sources.\n')
        Lr = np.zeros((nObserver,nPanel,nSample), dtype= np.float32)
        Lr = Extract_NoiseSources(Lr, pressure, nSample, nPanel, nObserver, FreeStreamPressure, surface_geo)        

        # Fluctuation on zero mean.
        print ("2.2 Fluctuation on zero mean.\n")
        Lr_mean = np.mean(Lr, axis = 2) # axis = 2 is the sample axis
        Lr = Extract_Mean(Lr, Lr_mean, nSample, nPanel, nObserver)

        # From Compute_TimeDomainPanelSignal
        print ("2.3 Computing retarted time.\n")
        pp_ret = Compute_RetardedTime(pp_ret, Lr, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, a_inf)

        # Delete - In python it is not neccessary however this program will eat all your available memory.
        del Lr_mean
        del Lr

    elif FWH_analogy == '1A_WT':
        print ("2. Computing retarted time for WT formulation.\n")
        pp_ret = Compute_RetardedTime_WT(pp_ret, pressure, surface_geo, nSample, nPanel, nObserver, FreeStreamPressure, FreeStreamMach, M_i, a_inf, dt, SamplingFreq)
    

    # From Compute_ObserverTime
    print ("3. Computing Oberserver time.\n")
    t_interp, t_Obs = Compute_ObserverTime( t_interp, t_Obs, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, start_iter, a_inf)
    # From Interpolate_PressureSignal - Check with Beckett the order of the interpolator.
    print ("4. Interpolating the pressure signal.\n")
    pp_interp = Interp_PressureSignal(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver)

    # Free RAM
    del t_Obs
    del pp_ret

    # From Integrated_Sources
    print ("5. Integrating sources.\n")
    pp_TimeDomain = Integrate_Sources(pp_TimeDomain, pp_interp, nSample, nPanel, nObserver)

    # Free RAM
    del pp_interp

    print ("Writing the results.\n")
    header = "#"
    for iObserver in range(nObserver):
        header += " Oberserver %d"%iObserver
    
    combined_data = np.column_stack((t_interp.transpose(), pp_TimeDomain.transpose()))
    np.save(name2save, combined_data)
    #np.savetxt("Observer_Noise.dat", np.column_stack((t_interp.transpose(),pp_TimeDomain.transpose()))  , fmt='%.18e', delimiter=' ', newline='\n', header=header)


    # Calculating the mean
    t_interp_mean = np.mean(t_interp, axis = 1)
    pp_TimeDomain_mean = np.mean(pp_TimeDomain, axis = 1)    
    print('\n       pp_TimeDomain_mean: ', pp_TimeDomain_mean)

    # Calculo em Db
    P_rms2 = 1/2*(np.mean(pp_TimeDomain**2, axis=1))
    P_ref = 20e-6
    SPL = 10 * np.log10(P_rms2/P_ref**2)

    # Inicializando matriz de valores médios
    #mean_values = np.empty((nObserver,3), dtype=np.float64)

    # Botando os valores
    #mean_values[:,0] = t_interp_mean
    #mean_values[:,1] = pp_TimeDomain_mean
    #mean_values[:,2] = SPL

    #print('\nMean values: \n', mean_values)
    #print('t_interp <--> pp_TimeDomain_mean <--> SPL')
    #np.save(name, mean_values)

    print('\nOASPL: ', SPL)

    # Free RAM
    del surface_geo
    del t_interp
    del pp_TimeDomain

    # ======== end cpu time
    fwh_end = timeit.default_timer()
    fwh_time = fwh_end - fwh_start
    
    print ("Total time: %.3f min."%(fwh_time/60))

    return None

import Gui_FWH_SU2 as FWH
#import Gui_FWH_V3_dtORIGINAL as FWH
import numpy as np
import h5py
import os
import time

# Inputs
dt = 8e-6                                                           # !!! AQUI !!!
p0 = 101322.85023
u0 = 0
c0 = 343
FreeStreamVar = np.array([p0, u0, c0])

# Caminhos para os arquivoss
path2root = '/home/vinicius.rodrigues/Results/'
casename  = 'Plate_r10_short_Coarse'                                             # !!! AQUI !!!

# Carregando auxiliares
print('\nLoading auxiliaries...')
start = time.time()

path2aux  = os.path.join('/home/jet/Poitiers/hdf5/Plates/aux', casename + '_aux.h5')
h5 = h5py.File(path2aux, 'r')
area = h5['area'][:]
grid = h5['grid'][:,:,:]
h5.close()
end = time.time()
print(f'Loading auxs time: {(end-start):.2f} s')


# Achando as células corretas (com área não nula)
print('\nAchando as células corretas...')
start = time.time()

idx = np.where(area != 0 )[0]
print(f'    idx shape: {idx.shape}')

# Pegando os inputs somente nas células corretas
area     = area[idx]
grid     = grid[idx,:,:]

# Achando os planos
xmax = np.where(np.all(grid[:, :, 0] == np.max(grid[:, :, 0]), axis=1))[0]
xmin = np.where(np.all(grid[:, :, 0] == np.min(grid[:, :, 0]), axis=1))[0]

ymax = np.where(np.all(grid[:, :, 1] == np.max(grid[:, :, 1]), axis=1))[0]
ymin = np.where(np.all(grid[:, :, 1] == np.min(grid[:, :, 1]), axis=1))[0]

zmax = np.where(np.all(grid[:, :, 2] == np.max(grid[:, :, 2]), axis=1))[0]
zmin = np.where(np.all(grid[:, :, 2] == np.min(grid[:, :, 2]), axis=1))[0]

# Teste para ver se a divisão entre planos está correta
nArea = len(area)
if nArea != (len(xmax)+len(xmin)+len(ymax)+len(ymin)+len(zmax)+len(zmin)):
    raise(ValueError('Soma dos planos está diferente da grid total!)'))

end = time.time()
print(f'Tempo para achar células corretas: {(end-start):.2f} s')


# Achando os versores normais
print('\nCriando vetores normais...')
start = time.time()

nvet       = np.zeros((nArea,3))
nvet[xmax] = np.array([ 1,0,0])
nvet[xmin] = np.array([-1,0,0])

nvet[ymax] = np.array([0, 1,0])
nvet[ymin] = np.array([0,-1,0])

nvet[zmax] = np.array([0,0, 1])
nvet[zmin] = np.array([0,0,-1])

# Pegando a célula como a média dos quatro vértices que a compõe
grid = np.mean(grid[:,:,:], axis =1)

end = time.time()
print(f'Tempo para nvet e parecidos: {(end-start):.2f}')


print('\n-------------------------------------------------')
print('------------ Loop over all Files ----------------'  )
print('-------------------------------------------------\n')

# Finding files to post process
#path2rec = os.path.join(path2root, casename ,'DMD/Opt_Delay_Reconstruction')
path2rec = '.' # test
#files = [f for f in os.listdir(path2rec) if os.path.isfile(os.path.join(path2rec, f))]
#files = ['optdmd_delay_results_80.h5', 'optdmd_delay_results_85.h5', 'optdmd_delay_results_90.h5', 'optdmd_delay_results_95.h5', 'optdmd_delay_results.h5'] ### atenção
#files = ['optdmd_delay_c0.1_results.h5', 'optdmd_delay_c0.3_results.h5', 'optdmd_delay_c0.4_results.h5', 'optdmd_delay_c0.6_results.h5', 'optdmd_delay_c0.7_results.h5', 'optdmd_delay_c0.8_results.h5', 'optdmd_delay_c0.9_results.h5']
#files = ['optdmd_delay_75percent_forcedeigs_results.h5', 'optdmd_delay_results.h5', 'optdmd_delay_75percent_results.h5','Pressure_ens_HDF.h5']
#files = ['Pressure_ens_HDF.h5']
files = ['spodmd_turbo1000m_results.h5']

# Loading observers
#Observer = np.load('./Obs3.npy', allow_pickle=True).item()
Observer = np.load('/home/jet/FWH_Molina/Observers/ObsAll.npy', allow_pickle=True).item()
Observer['xyz'] = Observer['xyz'][3] #### OBS3
Observer['name'] = Observer['name'][3] #### OBS3
nObs = len(Observer['name'])
nObs = 1 ####OBS3
print('\nTotal nObs: ', nObs)

# Loop over all files
for iFile in files:
    # Carregando dados de pressão
    print('Loading pressure for ', iFile)
    start = time.time()

    # Achando o respectivo modo baseado no nome do arquivo
    imode  = iFile.split('_')[-3] ############################################################# cuidado #######################################################################
#    imode  = imode.split('.')[0]
    # InName = iFile.split('_')[0] + '_' + iFile.split('_')[1]

    #
    path2data = os.path.join(path2rec, iFile)
    #h5        = h5py.File(path2data, 'r')
    #pressure  = h5['Pressure'][:,:]
    # pressure  = h5['Time'][:,:]                                         # !!! AQUI !!!
    # h5.close()
    try:
        pressure = np.real(h5py.File(path2data, 'r')['Pressure'][:,:])
    except:
        pressure = np.real(h5py.File(path2data, 'r')['reconstructed_data'][:,:].T)
    print(pressure.shape)
    pressure = np.vstack((pressure, pressure))
    print(pressure.shape)
    end = time.time()
    print(f'    Loading pressure time: {(end-start):.2f} s')

    # Células certas para a pressão
    pressure = pressure[:,idx] # Teoricamente os dados reconstruidos ja vem sem as celulas cuja area é nula (não é o caso pro dmd)
    if pressure.shape[1] != nArea:
        print('\nWARNING!')
        print(f'    nPressure: {pressure.shape[1]}')
        print(f'    nArea    : {nArea}')
        break

    # Inputs
    print('\n     FWH\n')
    analogy = '     1A'
    normals = np.column_stack((nvet, area))
    print(f'\n     ANALOGY : {analogy} \n')

    # Loop over all observers
    for i in range(nObs):

        # Criando observador
        #observer = Observer['xyz'][i,:]
        observer = Observer['xyz'] #### OBS3
        observer = observer[np.newaxis,:]

        #name2save = os.path.join(path2root, casename, 'DMD/Full_Noise_20000', f'mode_{imode}', f'{Observer['name'][i]}.npy')
        name2save = os.path.join(path2root, casename, 'DMD/New_FWH_Guilherme', f'mode_{imode}', f'{Observer['name']}.npy') #### OBS3
        #name2save = os.path.join(path2root, casename, 'DMD/test_noise', f'{Observer['name']}.npy') #### OBS3 test


        # Main
        print('\n------------------------MAIN------------------------')
        print(f"          Observer: {Observer['name'][i]}")
        FWH.main(
                pressure        = pressure,
                Observer        = observer,
                coord           = grid,
                normals         = normals,
                FreeStreamVar   = FreeStreamVar,
                dt              = dt,
                name2save       = name2save)

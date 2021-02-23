#@title Create mouse { form-width: "300px" }
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal
import math
from scipy.stats import kstest
import copy
from mouse import Mouse, TrimTime, GetRears, RearStat

def shuffle(X,y):
    # shuffle
    N = X.shape[0]
    indexes = np.arange(N)
    np.random.shuffle(indexes)

    X_shuffled = X[indexes]
    y_shuffled = y[indexes]
    return X_shuffled, y_shuffled

def get_center_coords(session, day):
    xc = None
    yc = None

    if day == 1:
        if session == 26:
            xc = 0.58
            yc = 0.48
            
    
    return (xc, yc)

def get_sync_time(session, day):
    if session == 22:
        sync_times = [34.5, 18.1, 14.25]

    elif session == 23:
        sync_times = [54.3, 43.8, 30.8]

    elif session == 24:
        sync_times = [13.5, 8.45, 19.4]
    
    elif session == 25:
        sync_times = [16.9, 43.9, 16.9]
    
    time_shift = sync_times[day-1]

    return time_shift

def create_mouse(session, day, build_spikes, auto_spikes = 1):

    PATH = './Data_Holes'
    
    time_shift = get_sync_time(session, day)
    xc, yc = get_center_coords(session, day)
    name = 'CA1_'+str(session)+'_HT'+str(day)

    path_track = os.path.join(PATH, name + '_track.csv')
    path_neuro = os.path.join(PATH, name + '_NR_raw_neuro.csv')
    path_spatial_info = os.path.join(PATH, name + '_spatial_info.csv')
    
    if auto_spikes:
        path_spikes = os.path.join(PATH, name + '_NR_spikes.csv')
    else:
        path_spikes = os.path.join(PATH, name + '_NR_spikes_Vova.csv')


    ms = Mouse(name, session, day, time_shift, build_spikes, xc, yc,
               path_track = path_track, path_neuro = path_neuro,
               path_spikes = path_spikes, path_spatial_info = path_spatial_info,
               xy_delim = '\t', xy_sk_rows = 0)

    return ms

def create_ubermouse(mslist, day):

    ubermouse = mslist[0]
    ubermouse.name = 'CA1_mixed_over_'+str(len(mslist))+'_HT'+str(day)
    ubermouse.session = None
    ubermouse.day = day
    ubermouse.time_shift = None


    max_n_cells = max([ms.n_cells for ms in mslist])
    ubermouse.n_cells = max_n_cells

    for ms in mslist[1:]:
        if ms.pc_mask is None:
            raise Exception('empty pc mask found during merging')

        ubermouse.n_frames += ms.n_frames
        ubermouse.pc_mask += np.concatenate([ms.pc_mask, 
                                             np.zeros(max_n_cells - ms.n_cells, dtype = bool)])

        ubermouse.x = np.concatenate([ubermouse.x, ms.x])
        ubermouse.y = np.concatenate([ubermouse.y, ms.y])
        #ubermouse.scx = np.concatenate([ubermouse.scx, ms.scx])
        #ubermouse.scy = np.concatenate([ubermouse.scy, ms.scy])

        ubermouse.xc = (min(ubermouse.scx)+max(ubermouse.scx))/2
        ubermouse.yc = (min(ubermouse.scy)+max(ubermouse.scy))/2

        ubermouse.time = np.concatenate([ubermouse.time, ms.time])
        ubermouse.v = np.concatenate([ubermouse.v, ms.v])
        ubermouse.vx = np.concatenate([ubermouse.vx, ms.vx])
        ubermouse.vy = np.concatenate([ubermouse.vy, ms.vy])
        ubermouse.angle = np.concatenate([ubermouse.angle, ms.angle])

        ubermouse.spatial_info = np.concatenate([ubermouse.spatial_info, ms.spatial_info])
        ubermouse.rand_spatial_info = np.concatenate([ubermouse.rand_spatial_info, ms.rand_spatial_info])
        ubermouse.rand_spatial_info_std = np.concatenate([ubermouse.rand_spatial_info_std, ms.rand_spatial_info_std])

        spikes_to_add = np.concatenate([ms.spikes,
                                        np.zeros((ms.spikes.shape[0],
                                                  max_n_cells - ms.n_cells))], axis = 1)
        ubermouse.spikes = np.concatenate([ubermouse.spikes, spikes_to_add])

        smoothed_spikes_to_add = np.concatenate([ms.smoothed_spikes,
                                                 np.zeros((ms.smoothed_spikes.shape[0],
                                                            max_n_cells - ms.n_cells))], axis = 1)
        ubermouse.smoothed_spikes = np.concatenate([ubermouse.smoothed_spikes, smoothed_spikes_to_add])
        
        neur_to_add = np.concatenate([ms.neur,
                                       np.zeros((ms.neur.shape[0],
                                                 max_n_cells - ms.n_cells))], axis = 1)
        ubermouse.neur = np.concatenate([ubermouse.neur, neur_to_add])

    ubermouse.scx = (ubermouse.x-min(ubermouse.x))/(max(ubermouse.x)-min(ubermouse.x))-0.000001
    ubermouse.scy = (ubermouse.y-min(ubermouse.y))/(max(ubermouse.y)-min(ubermouse.y))-0.000001
    ubermouse.xc = 0.48
    ubermouse.yc = 0.49
    
    print(ubermouse.n_cells)
    print(ubermouse.spikes.shape)
    print(ubermouse.neur.shape)
    return ubermouse
    '''
    duration = min([ms.n_frames for ms in mslist])
    for ms in mslist:
        ms.start_trim = 0
        ms.end_trim = duration
        ms.trim_data()
    '''


def write_SI_info(session, day, auto_spikes = 0):
    PATH = './Data'

    ms = create_mouse(session, day, 0)
    #print(ms.spatial_info)
    for cell in tqdm.tqdm(range(ms.n_cells), leave = 1, position = 0):
        SI_sigma_thr = -1000
        check_place_cell_SI(ms, cell, SI_sigma_thr, 
                            nsim = 1000, size = 100, recalculate = 1, silent = 1)
    #print(ms.spatial_info)
    d = {'spatial_info': ms.spatial_info,
         'rand_spatial_info': ms.rand_spatial_info,
         'rand_spatial_info_std': ms.rand_spatial_info_std}

    df = pd.DataFrame(data=d)
    name = 'CA1_'+str(session)+'_HT'+str(day) + '_spatial_info.csv'
    df.to_csv(os.path.join(PATH, name))
    
#@title Data acquision { form-width: "300px" }
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import scipy.sparse as sp

def import_holes_neurodata(ms, inds = None, spmode = 0, std = 1):

    PATH = './Data'

    dname  = 'CA1_'+str(ms.session)+'_HT'+str(ms.day)

    d = ms.neur.T[ms.pc_mask, :]

    if not inds is None and not spmode:
        d = d[:, inds]
    

    scaler = MinMaxScaler()

    if std:
        #d = scaler.fit_transform(d)
        final_data = scaler.fit_transform(d.T).T
    else:
        final_data = d


    return final_data

def fill_pc_mask(ms, frcells, min_spikes, cell_selection_method):

    print('Filtering cells...')
    res = np.zeros(ms.n_cells, dtype = bool)
    cells_with_enough_spikes = np.where(ms.spikes_count > min_spikes)[0]
    ncells = int(frcells*len(cells_with_enough_spikes))

    '''
    for cell in tqdm.tqdm(range(ms.n_cells), leave = 1, position = 0):
        is_pc = check_place_cell_SI(ms, cell, SI_sigma_thr, 
                                    nsim = 1000, size = 100, silent = 1)
        #stat = check_place_cell2(ms, cell, silent = 1)
        if is_pc:
            res[cell] = True
    '''

    if cell_selection_method == 'si':
        #non_empty_cells = ms.n_cells - sum((np.isnan(ms.cell_z_score).astype(int)))
        candidate_z_scores = ms.cell_z_score[cells_with_enough_spikes]
        good_cell_inds = np.argsort(candidate_z_scores)[-ncells:]
        print('cells with enough spikes:', len(cells_with_enough_spikes))

    elif cell_selection_method == 'random':
        good_cell_inds = np.random.choice(len(cells_with_enough_spikes), ncells)
    
    elif cell_selection_method == 'nspikes':
        spcount = ms.spikes_count[cells_with_enough_spikes]
        good_cell_inds = np.argsort(spcount)[-ncells:]
    
    res[cells_with_enough_spikes[good_cell_inds]] = 1
    ms.pc_mask = res[:]

def param_checkpoint(rebuild_place_cells, rebuild_mouse, frcells, min_spikes, sessions):

    if 'prev_frcells' in globals() and 'prev_min_spikes' in globals():
        rebuild_place_cells = bool(rebuild_place_cells + \
                                   int(prev_frcells != frcells) + int(prev_min_spikes != min_spikes))
        
    if 'prev_sessions' in globals():
        rebuild_mouse = bool(rebuild_mouse + (prev_sessions != sessions))

    return rebuild_mouse, rebuild_place_cells



def load_mouse_and_data(sessions, day, build_spikes, cell_selection_method,
                        frcells, min_spikes, spikes_only, std,
                        rebuild_mouse, rebuild_place_cells, rebuild_data,
                        data_compression, **kwargs):

    rebuild_mouse, rebuild_place_cells = param_checkpoint(rebuild_place_cells, rebuild_mouse,
                                                          frcells, min_spikes, sessions)

    if 'ms' in globals() and not rebuild_mouse:
        ms = globals()['ms']
        if ms.pc_mask is None or rebuild_place_cells or rebuild_mouse:
            fill_pc_mask(ms, frcells, min_spikes, cell_selection_method)

    else:
        print('Creating mouse object...')

        if len(sessions) == 1:
            ms = create_mouse(sessions[0], day, build_spikes, auto_spikes = 0)
            if ms.pc_mask is None or rebuild_place_cells or rebuild_mouse:
                fill_pc_mask(ms, frcells, min_spikes, cell_selection_method)

        else:
            mslist = []
            for session in sessions:
                ms = create_mouse(session, day, build_spikes, auto_spikes = 0)
                if ms.pc_mask is None or rebuild_place_cells or rebuild_mouse:
                    fill_pc_mask(ms, frcells, min_spikes, cell_selection_method)
                mslist.append(ms)

            ms = create_ubermouse(mslist, day)
        
        
        #ms.pc_mask = np.ones(ms.n_cells, dtype = bool)


    if 'D' in globals() and not (rebuild_data or rebuild_place_cells or rebuild_mouse):
        D = globals()['D']

    else:
        print('Creating data...')

        indices_to_take = np.where(ms.v > 15)[0]

        data = import_holes_neurodata(ms, indices_to_take, spmode=spikes_only, std = std)
        print('Done')
        
    return indices_to_take, ms, data


def calc_gradient_magnitude(named_parameters, silence=False):
    total_amplitude = []
    for name, p in named_parameters:
        # print(name)
        if p.grad is not None:
            param_amplitude = p.grad.data.abs().max()
            total_amplitude += [param_amplitude.item()]
        elif not silence:
            print (name, 'grad is None')    

    total_amplitude = max(total_amplitude)

    return total_amplitude
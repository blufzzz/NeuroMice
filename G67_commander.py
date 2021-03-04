# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:44:04 2021

@author: vsots
"""
import numpy as np
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import Heat_field_Vitya as hf
import PlaceFields as pcf

names = ['G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_1_2D', 'G7F_2_1D', 'G7F_2_2D']
path = 'D:\Work\G67\\'
time_shift = [13.1, 11.55, 11.8, 11.1, 12.35, 9.1]

for i,name in enumerate(names):
    ms = mcl.Mouse(name, time_shift[i], path_track = path + name + '_track_fl.csv', path_neuro = path + name + '_NR_data_processed_traces.csv', xy_delim = ',')
    ms.get_spike_data(path + 'spikes_' + name + '_NR_data_processed_traces.csv')
    ms.get_min1pipe_spikes(path + name + '_NR_data_processed_spikes.csv')
    ms.get_cell_centers(path + name + '_NR_data_processed_filters\\')
    
    if i>1:
        ms.get_markers_data(path + name + '_track_fl.csv', delimiter = ',')

    ms.get_angle()
    ms.get_ks_score(min_n_spikes = 3)
    ms.get_place_cells_in_circle(mode = 'spikes')
    np.savez(path + name + '_sp.npz', [ms])

    
    # ms = pio.LoadMouse(path + name + '_min_sp.npz')
    


    ms = mcl.FieldsToList(ms)
    ms.get_binned_neuro(10)    
    drcl.DrawAllCells(ms, k_word = path + name + 'sp_circle_map')

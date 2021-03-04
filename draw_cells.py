# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:34:36 2020

@author: VOVA
"""


import numpy as np
import matplotlib.pyplot as plt
import pc_io as pio
import matplotlib.patches as patches
import matplotlib.cm as cm
# import matplotlib.rc as rc
import structure as strct
import glob
from PIL import Image
from skimage import io
from scipy.ndimage.morphology import distance_transform_edt as dtrfm

#import behavior_plot as bplt

def DrawField(mu, std, t_spec, t_end, **kwargs):
    #Draw sector for putative place field on circle trajectory diagram
   if 'ax' in kwargs:
       ax = kwargs['ax']
   else:
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar')
       
   if 'color' in kwargs:
        facecol = kwargs['color']
        edgecol = 'None'
        if isinstance(kwargs['color'], str):
            facecol = 'None'
            edgecol = kwargs['color']           
 
   else:
        facecol = 'None'
        edgecol = 'm'
      
   plt.setp(ax.get_yticklabels(), visible=False)
   plt.setp(ax.get_xticklabels(), visible=False)
   bins = np.arange(mu-std/2, mu+std/2, 5)*np.pi/180 
   for b in bins: 
       ax.add_patch(patches.Rectangle((b,t_spec), 5*np.pi/180, t_end-t_spec, facecolor = facecol, edgecolor = edgecol, linewidth = 2, alpha=0.5))
   return ax
	

def colornum_Metro(num):
    #Returns color for each number as in Moscow Metro
    return {
    1:[1,0,0],        #red
    2:[0,0.7,0],      #green
    3:[0,0,1],        #blue
    4:[0,1,1],        #cyan
    5:[0.5,0.25,0.2], #brown
    6:[1,0.5,0.1],    #orange
    7:[0.5,0,1],      #violet
    8:[0.8,0.7,0],    #yellow
    9:[0.5,0.5,0.5],  #grey
    0:[0.5,1, 0.1]}.get(num%10)   #lime



def Traceviewer(Mouse, out_fname):
    #plots neural activity traces of all cells on single linear plot
    fig = plt.figure(figsize=(50,50), dpi=300)
    ax = fig.add_subplot(111)
    
    n_cells = len(Mouse.spikes[0])
    absmax = np.amax(Mouse.neur)
    if hasattr(Mouse, 'behav'):
        for i in range(len(Mouse.time)):
            if Mouse.behav[i] > 0:
                ax.add_patch(patches.Rectangle((Mouse.time[i], 0), 0.05, n_cells+1, facecolor = colornum_Metro(int(Mouse.behav[i])), edgecolor = 'none', alpha=0.3))
    for cell in range(n_cells):
        ax.plot(Mouse.time, Mouse.neur[:,cell]/absmax + cell , c = colornum_Metro(cell), linewidth=1)
    plt.savefig(out_fname, dpi = 300)
    plt.close()



def DrawXYTrack(Mouse, cell, f_out, **kwargs):
    #Draws trajectory of mouse in arbitrary track coloured with respect to neural activity of the given cell
    if 'line_width' in kwargs:
        lnwd = kwargs['line_width']
    else:
        lnwd = 2 
        
#    markers_amplitude = Mouse.spikes[Mouse.spikes[:,cell]!=0, cell] * 100 / np.max(Mouse.spikes[:,cell])
#    markers_amplitude[markers_amplitude <= 25] = 25
#    markers_amplitude = (np.around(markers_amplitude))

    plt.clf()
    fig = plt.figure(figsize=(10,10), dpi=300)
    if hasattr(Mouse, 'mi_score'):
        fig.suptitle(Mouse.name + ' cell_' + str(cell) + ' MI =' + str(Mouse.mi_score[cell]), fontsize=16)
    else:
        fig.suptitle(Mouse.name + ' cell_' + str(cell), fontsize=16)
        
    Nbins = np.max(Mouse.neuro_bin[:,cell])
    for n_bin in range(Nbins):
        color = cm.jet(n_bin/Nbins)                
        plt.plot(Mouse.x[Mouse.neuro_bin[:,cell] == n_bin], Mouse.y[Mouse.neuro_bin[:,cell] == n_bin], c = color, ls = ' ', ms=lnwd, marker='.', zorder = 0.0) 
             
#    plt.scatter(Mouse.x[Mouse.neuro_bin[:,cell] == n_bin], Mouse.y[Mouse.neuro_bin[:,cell] == n_bin], marker='^', s=markers_amplitude*lnwd,  edgecolors = 'black', c = (1., 0., 0.), zorder = 1.0)
    plt.savefig(f_out) 
    plt.close(fig)
    
    

def plot(Mouse, cell, mode, k_word, polar, save, ax, **kwargs):
    #Main function for circle plots
    if 'line_width' in kwargs:
        lnwd = kwargs['line_width']
    else:
        lnwd = 1.5  
        
    markers_amplitude = Mouse.spikes[Mouse.spikes[:,cell]!=0, cell] * 100 / np.max(Mouse.spikes[:,cell])
    markers_amplitude[markers_amplitude <= 25] = 25
    markers_amplitude = (np.around(markers_amplitude))

    if polar:  
        #Mouse.time = np.array(list(range(1,len(Mouse.angle)+1)))/20
        
        if mode == "None":
            pass
        
        elif mode == "Plain":
            ax.plot(Mouse.angle*np.pi/180, Mouse.time, color = 'black', linewidth=lnwd, zorder = 0.0)
        
        elif mode == "Speed":
            for x in range(len(Mouse.angle)-1):
                if Mouse.speed[x] < 0.1:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0.7, 1., 0.7), linewidth=lnwd, zorder = 0.0)
                elif 0.1 <= Mouse.speed[x] < 0.5:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0., 1., 0.), linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0., 0.5, 0.), linewidth=lnwd, zorder = 0.0)
                                
        elif mode == "Direction":
            for x in range(1, len(Mouse.angle)):
                if Mouse.direction[x] >=0: 
                    ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], c = 'b', linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], c = 'c', linewidth=lnwd, zorder = 0.0)
                    
        elif mode == "Neuro":
            maxn = max(Mouse.neur[:, cell])
            minn = min(Mouse.neur[:, cell])
            for x in range(1, len(Mouse.angle)):
                color = cm.jet((Mouse.neur[x, cell]-minn)/(maxn-minn))
                ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], color, linewidth=lnwd, zorder = 0.0) 
                
        elif mode == "Neuro_bin":
            Nbins = np.max(Mouse.neuro_bin[:,cell])
            for n_bin in range(Nbins):
                color = cm.jet(n_bin/Nbins)                
                ax.plot(Mouse.angle[Mouse.neuro_bin[:,cell] == n_bin]*np.pi/180, Mouse.time[Mouse.neuro_bin[:,cell] == n_bin], color = color, ls = ' ', ms=2, marker='.', zorder = 0.0) 
                 
        ax.scatter(Mouse.angle[Mouse.spikes[:,cell]!=0]*np.pi/180, Mouse.time[Mouse.spikes[:,cell]!=0], marker='^', s=markers_amplitude*lnwd,  edgecolors = 'black', color = (1., 0., 0.), zorder = 1.0)
    

    else:
        if mode == "None":
            ax.plot(list(Mouse.x), list(Mouse.y), color = 'black', linewidth=lnwd, zorder = 0.0)
        elif mode == "Speed":
            ms = max(Mouse.speed)
            for i in range(len(Mouse.x)-1):
                if Mouse.speed[i] < ms/3:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0., 0.5, 0.), linewidth=lnwd, zorder = 0.0)
                elif Mouse.speed[i] < 2*ms/3:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0., 1., 0.), linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0.7, 1., 0.7), linewidth=lnwd, zorder = 0.0)
                                
        elif mode == "Acceleration":
            for i in range(1, len(Mouse.x)-1):
                meda = np.median(Mouse.acceleration)
                stda = np.std(Mouse.acceleration)
                if Mouse.acceleration[i] >= meda + 2*stda: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = 'm', linewidth=lnwd, zorder = 0.0)
                elif Mouse.acceleration[i] <= meda - 2*stda: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = 'c', linewidth=lnwd, zorder = 0.0)
                else: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0.7, 0.7, 0.7), linewidth=lnwd, zorder = 0.0)  
                    
        ax.scatter(Mouse.x[Mouse.spikes[:,cell]!=0], Mouse.y[Mouse.spikes[:,cell]!=0], marker='^', s=markers_amplitude*1.5,  edgecolors = 'black', c = (1., 0., 0.), zorder = 1.0)


    if save:
        path = Mouse.path_track
        plt.savefig(path + '_' + str(cell) + '_plot_' + k_word + '.png', dpi = 300)
        plt.close()



       
def DrawAllCells(Mouse, k_word):
   #Draws circle cells for all cells, place cells with fields and none-place cells without fields
   supcl = 0
   for i in range(len(Mouse.spikes[0])): #first draw all usual cells
       if i in Mouse.true_cells:
           continue
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=100)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(Mouse.name + ' cell_' + str(i+1) + ' KS_stat =' + str(Mouse.ks_stat[i]) + ' KS_p_score =' + str(Mouse.ks_p_score[i]), fontsize=16)
       plot(Mouse, i, mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 1.5)
       plt.savefig(k_word + '_' + str(i+1) + '_circ.jpg')
       plt.close()  
    
   for i, cell in enumerate(Mouse.true_cells):
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=100)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(Mouse.name + ' cell_' + str(cell+1) + ' KS_stat =' + str(Mouse.ks_stat[cell]) + ' KS_p_score =' + str(Mouse.ks_p_score[cell]), fontsize=16)
       
       ax = DrawField(Mouse.mu_list[i], Mouse.std_list[i], Mouse.time[0], Mouse.time[-1], ax = ax, color = 'c')
        
       plot(Mouse, cell, mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 1.5)
       plt.savefig(k_word + '_' + str(cell+1) + '_true_' + str(i+1) + '_circ.jpg')
       plt.close()         
       
       
       
def DrawOneCell(Mouse, cell, pf_n, k_word): 
    
       for i, pc in enumerate(Mouse.pc):
           if pc.cell_num == cell:
               break
       
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(Mouse.name + ' cell_' + str(cell+1) + ' KS_stat =' + str(Mouse.ks_stat[cell]) + ' KS_p_score =' + str(Mouse.ks_p_score[cell]), fontsize=16)
       
       ax = DrawField(Mouse.pc[i].pf[pf_n].mu, Mouse.pc[i].pf[pf_n].std, Mouse.time[0], Mouse.time[-1], ax = ax, color = [0.7,0.7,0.7])
        
       plot(Mouse, cell, mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 1.5)
       plt.savefig(k_word + '_' + str(cell+1) + '_true_' + str(i+1) + '_circ.png')
       plt.close() 
  
       

def DrawMostInformative(Mouse, k_word, Ncells):
   sorted_mi = np.argsort(Mouse.mi_score)[::-1][:Ncells]
   for i in range(Ncells):
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(Mouse.name + ' cell_' + str(sorted_mi[i]) + ' MI =' + str(Mouse.mi_score[sorted_mi[i]]), fontsize=16)
       
       plot(Mouse, sorted_mi[i], mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 0.5)
       plt.savefig(k_word + '_' + str(i) + '_' + str(sorted_mi[i]) + '_circ.png')  
       plt.close()    




     
def Draw_PC_ContourMap(Mouse, filterpath, f_out, sort = True):
    #coloring cell contours in the NVista FOV with respect to their place fields
    fnames = glob.glob(filterpath+'\\*.tif')
    im = io.imread(fnames[0])
    res_im = np.zeros([*im.shape, 3], dtype = float)
    pcs = [pc.cell_num for pc in Mouse.pc]
    if sort:
        order = np.argsort(Mouse.ks_p_score)[::-1].tolist()  #sorting by ks score in descending order
    else:
        order = range(Mouse.n_cells)
        
    for cell in order:
        im = io.imread(fnames[cell])
        cell_inside = im > 0
        col_im = np.array([im.T, im.T ,im.T], dtype = float).T   #adding 3rd dim for color

        x_min = min(im.nonzero()[1]) #minimal nonzero column
        x_len = max(im.nonzero()[1]) - x_min #count of nonzero columns
        x = [int(x_min)]                        #array of columns:  x0  color0 x1 color1 x2

        if cell in pcs: #place cells, including bimodal
            color = []                         
            npf = len(Mouse.pc[pcs.index(cell)].pf)  #number of place fields
            for ipf, pf in enumerate(Mouse.pc[pcs.index(cell)].pf):  
                color.append(np.array(cm.jet(pf.mu/180)[0:-1]))    #round colormap with respect to PF position
#                color[-1] *= 1 - np.square(pf.std/90)         #the thiner PF is, the darker its color is set
                x.append(int(x_min + x_len*(ipf+1)/npf + 1))  #next vertical strip for the next PF of this cell

        else:  #non-place cells
            color = [[0.5,0.5,0.5]]
            x.append(int(x_min + x_len + 1))
        
        #Coloring
        for nc,cin in enumerate(color):
            for c in [0,1,2]:
                col_im[:,x[nc]:x[nc+1], c] *= cin[c]

        res_im[cell_inside,:] = col_im[cell_inside,:]
        
    res_im /= np.max(res_im)
    io.imshow(res_im)        
    io.imsave(f_out, res_im)
    
def DrawContourMap(Mouse, filterpath, f_out):
    #coloring cell contours in the NVista FOV with respect to their place fields
    fnames = glob.glob(filterpath+'\\*.tif')
    im = io.imread(fnames[0])
    res_im = np.zeros([*im.shape, 3])
    pcs = [pc.cell_num for pc in Mouse.pc]
    for cell,f in enumerate(fnames):
        im = io.imread(f)
        dt = dtrfm(im)                  #distance transform
        im = np.zeros([*im.shape, 3])   #adding 3rd dim for color
        bord = (dt > 0)*(dt < 2)
        cell_inside = dt >=2

        x_min = min(cell_inside.nonzero()[1]) #minimal nonzero column
        x_len = max(cell_inside.nonzero()[1
                    ]) - x_min #count of nonzero columns
        x = [int(x_min)]                        #array of columns:  x0  color0 x1 color1 x2

        if cell in pcs: #place cells, including bimodal
            #color_bord = colornum_Metro(cell+1)    #border color
            color_bord = [[0.5,0.5,0.5]]
            color_in = []                         #inner color
            npf = len(Mouse.pc[pcs.index(cell)].pf)  #number of place fields
            for ipf, pf in enumerate(Mouse.pc[pcs.index(cell)].pf):
                color_in.append(cm.jet(pf.std/180)[0:-1])
                x.append(int(x_min + x_len*(ipf+1)/npf))

        else:  #non-place cells
            color_bord = [[0.5,0.5,0.5]]
            color_in = [[0,0,0]]
            x.append(int(x_min + x_len))
        
        #inner coloring
        for nc,cin in enumerate(color_in):
            im[:,x[nc]:x[nc+1],:] = cin
        im[~cell_inside,:] = 0
        
        res_im[bord,:] = color_bord
        res_im[cell_inside,:] = im[cell_inside,:]
    io.imshow(res_im)        
    io.imsave(f_out, res_im)
    


def DrawKScoreMap(Mouse, filterpath, f_out):
    #coloring cell contours in the NVista FOV with respect to their KS-score
    fnames = glob.glob(filterpath+'\\*.tif')
    im = io.imread(fnames[0])
    res_im = np.zeros(im.shape)
    
    for cell,f in enumerate(fnames):
        im = io.imread(f)
        if np.isinf(Mouse.ks_p_score[cell]):
            Mouse.ks_p_score[cell] = 0
        res_im[im>0] += Mouse.ks_p_score[cell]
    res_im = (res_im - np.min(res_im))/(np.max(res_im) - np.min(res_im))        
    io.imshow(res_im)        
    io.imsave(f_out, res_im)       
    
    
def DrawTSpecVsMuDistrib(Mouse, f_out):
    
    
    mu_list = []
    t_spec_list = []
    std_list = []
    for pc in Mouse.pc:
        for pf in pc.pf:
            mu_list.append(pf.mu)
            t_spec_list.append(pf.t_spec/20)
            std_list.append(pf.std)
            
    plt.clf()
    plt.rcParams.update({'font.size': 36})
    fig = plt.figure(figsize=(10,10), dpi=300)
    fig.suptitle(Mouse.name + 'mu vs t_spec', fontsize=16)
           
    plt.scatter(Mouse.angle, Mouse.time-Mouse.time[0], c = [0.7,0.7,0.7], s = 1)
    plt.scatter(mu_list,t_spec_list, s = list(np.array(std_list)/1.5), c = [193/255,0,0])
    
    plt.savefig(f_out) 
    plt.close(fig)    
    



    
def DrawTSpecHistogram(spec_list, n_bins, f_out):  
    plt.clf()
    fig = plt.figure(figsize=(10,10), dpi=300)
    fig.suptitle('t_spec distribution', fontsize=28)
           
    plt.hist(spec_list, bins = n_bins)
    plt.tick_params(axis = 'both', labelsize = 28)
    
    plt.savefig(f_out) 
    plt.close(fig) 


def DrawSelectivityScore(Mouse, f_out):

    sum_ratio = np.zeros(Mouse.n_frames)
    
    plt.clf()
    fig = plt.figure(figsize=(10,10), dpi=300)
    fig.suptitle('selectivity score', fontsize=16)
    
    for pc in Mouse.pc:
        for pf in pc.pf:
            spikes_in_zone = np.zeros(Mouse.n_frames)
            tot_spikes = np.cumsum((Mouse.spikes[:, pc.cell_num] > 0).astype(float))
            tot_spikes[tot_spikes == 0] = 1 #in order to avoid division by 0
            for i, tin in enumerate(pf.times_in):
                spikes_in_zone[tin:pf.times_out[i]] = Mouse.spikes[tin:pf.times_out[i], pc.cell_num]
            spikes_in_zone = np.cumsum((spikes_in_zone > 0).astype(float))
                
            ratio = spikes_in_zone/tot_spikes
            sum_ratio += ratio
            
            plt.plot(Mouse.time, ratio, color = [0.7, 0.7, 0.7])
            
    plt.plot(Mouse.time, sum_ratio/len(Mouse.mu_list), color = 'blue', linewidth = 4)       
    plt.savefig(f_out) 
    plt.close(fig)  
    
def Plot_All_Coord(Mouse, f_out):
    #Plots real trajectory of mouse and markers
    plt.clf()
    fig = plt.figure(figsize=(10,10), dpi=300)
    plt.plot(Mouse.x, Mouse.y, color = 'b')
    plt.plot(Mouse.markers[0], Mouse.markers[1], color = 'g')
    plt.plot(Mouse.markers[2], Mouse.markers[3], color = 'r')
    plt.savefig(f_out) 
    plt.close(fig) 
    
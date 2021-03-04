# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:39:36 2020

@author: VOVA
"""

import numpy as np
from scipy import interpolate, signal
import math
from scipy.stats import kstest
from scipy.spatial.distance import euclidean
#import inspect as insp
import SpikingPasses as  spps
import PlaceFields as pcf
import glob
from skimage import io
#from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score


class PlaceCell():
    def __init__(self, cell_num):
        self.cell_num = cell_num
        
class PlaceField():
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

class Mouse():
    def __init__(self, name1, time_shift, path_track, path_neuro, xy_delim = ' ', xy_sk_rows = 1, xy_sk_cols = 0, cen_data= np.array([0]),**kwargs):
        self.name = name1
        self.dtypef = np.float32
        self.time_shift = time_shift
        
          
        self.get_xy_data(path_track, path_neuro, cen_data, xy_delim, xy_sk_rows, xy_sk_cols)
        
        if 'path_spikes' in kwargs:
            self.get_spike_data(kwargs['path_spikes'])       
        
        
    def get_xy_data(self, path_track, neuro_path, cen_data= np.array([0]), delimiter = ' ', skip_rows = 1, skip_cols = 0):
        w = 21
        time_massive = []
        x_massive = []
        y_massive = []
        self.x = []
        self.y = []
        self.speed = []
       
        tr = np.genfromtxt(path_track, delimiter=delimiter, skip_header=skip_rows)
        time_massive = tr[:,0]
        x_massive = tr[:,1+skip_cols]
        y_massive = tr[:,2+skip_cols]
        
        #pre-filtering
        x_massive = signal.medfilt(x_massive, 21)
        y_massive = signal.medfilt(y_massive, 21)

        if cen_data.any():
            x_massive = Drop_Wrong_Points(x_massive, y_massive, cen_data, d=20)            
        
        #removing NaN values
        valid_indices = ~np.isnan(x_massive) * ~np.isnan(y_massive)
        time_massive = time_massive[valid_indices]
        x_massive = x_massive[valid_indices]
        y_massive = y_massive[valid_indices] 

        #interp to 20 fps        
        fx = interpolate.interp1d(time_massive, x_massive)
        fy = interpolate.interp1d(time_massive, y_massive)
        self.time = list(range(int(time_massive[0]*20)+1, int(time_massive[-1]*20)))
        for i in range(len(self.time)):
            self.time[i] *= 0.05
            self.x.append(fx(self.time[i]))
            self.y.append(fy(self.time[i]))
        
        #smoothing with 1-s window average filter
        # self.x = signal.medfilt(self.x, 21)
        # self.y = signal.medfilt(self.y, 21)        
        self.x = np.convolve(self.x, np.ones(w)/w, mode='same')
        self.y = np.convolve(self.y, np.ones(w)/w, mode='same')
        
        for i in range(1, len(self.x)):
            self.speed.append(round(math.sqrt((self.x[i] - self.x[i-1])**2 + (self.y[i] - self.y[i-1])**2)/(self.time[i]-self.time[i-1]), 3))

        self.min_time_trace = self.time[0]
        shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
        t_size = len(self.time)
        neur = np.genfromtxt(neuro_path, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
        self.neur = neur[:, 1:]
        
        self.n_cells = len(self.neur[0])
        self.n_frames = len(self.neur)
        
        self.x = self.x[0:self.n_frames]
        self.y = self.y[0:self.n_frames] 
        self.speed = np.array(self.speed[0:self.n_frames])
        self.time = np.array(self.time[0:self.n_frames])
        return
    
    def get_spike_data(self, spike_path):
        shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
        t_size = len(self.time)
        sp = np.genfromtxt(spike_path, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
        self.spikes = sp[:, 1:]
        return
    
    def get_min1pipe_spikes(self, spike_path):
        shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
        t_size = len(self.time)
        sp = np.genfromtxt(spike_path, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
        self.min_spikes = sp[:, 1:]
        return
    
    def get_cell_centers(self, filter_path):
        self.centers = []
        for f in glob.glob(filter_path + '*.tif'):
            xc, yc =  np.nonzero(io.imread(f))
            self.centers.append([xc.mean(), yc.mean()])
        self.centers = np.array(self.centers)
        return
        
        
    def get_markers_data(self, path_track, cen_data = [0,0,0,0], delimiter = ' ', skip_rows = 1, skip_cols = 3, w = 21):
        #It is considered here that self.time is already calculated and properly cropped with respect to neurodata
        #w is window parameter for filtering
        self.markers = []     
        f = []    
        
        tr = np.genfromtxt(path_track, delimiter=delimiter, skip_header=skip_rows)
        for i in range(4):              #pre-filtering
            self.markers.append(signal.medfilt(tr[:,i + skip_cols],11))
          
        for j in [0,2]:  
            #0 for green, 2 for red
            self.timestamp = tr[:,0]
            self.valid_indices = np.ones(np.shape(tr)[0], dtype = int)
 
            #dropping "outstanding" points 
            if max(cen_data):
                self.markers[j] = Drop_Wrong_Points(self.markers[j], self.markers[j+1], cen_data, d=80)

            #removing NaN values                   
            for i in [0,1]: 
                self.valid_indices *= ~np.isnan(self.markers[i+j])
            self.timestamp = self.timestamp[self.valid_indices.astype('bool')]
            
            for i in [0,1]:              #calculating of linear interpolating function       
                self.markers[i+j] = self.markers[i+j][self.valid_indices.astype('bool')]
                f.append(interpolate.interp1d(self.timestamp, self.markers[i+j]))
            
        #filling self.markers with interpolated values
        self.markers = [[],[],[],[]]    #x_green, y_green, x_red, y_red
        for t in self.time:
            for i in range(4): 
                self.markers[i].append(f[i](t))  
                
        #final smoothing
        # for i in range(4): 
        #     self.markers[i] = np.convolve(self.markers[i], np.ones(w)/w, mode='same')           
            
        return
        
    def get_angle(self):
        self.angle = []
        xc = (min(self.x)+max(self.x))/2
        yc = (min(self.y)+max(self.y))/2
        for i in range(len(self.x)):
            self.angle.append(180 + round((math.degrees(math.atan2(self.y[i] - yc, self.x[i] - xc))),3))
        self.angle = np.array(self.angle)  
        return

    def get_direction(self):
         der = np.ediff1d(self.angle, 0)
         der[der <= -90] += 360
         der[der >= 90] -= 360
         self.direction = np.sign(der)
         return

    def get_acceleration(self):
        w = 35
        sp = np.convolve(self.speed, np.ones(w)/w, mode='same')
        self.acceleration = np.gradient(sp, 0.05, edge_order=2)
        return
        
    def get_binned_neuro(self, Nbins):
        self.neuro_bin = np.zeros(self.neur.shape, dtype = int)
        for i in range(self.neur.shape[1]):
            ma = np.max(self.neur[:,i])
            mi = np.min(self.neur[:,i])
            self.neuro_bin[:,i] = (self.neur[:,i] - mi)*Nbins/(ma-mi)
        self.neuro_bin[self.neuro_bin == Nbins] -= 1
        return

    def get_binned_angle(self, Nbins):
        self.spatial_bin = np.zeros(len(self.angle), dtype = int)
        self.spatial_bin = self.angle*Nbins/360
        return
    

    def get_place_cells_in_circle(self, mode):
        #searching for candidate cellsin circle track (former true_cells) by activity statistics
        #mode = 'spikes' | 'raw_neuro' <==> count spike cell activity or raw neuro traces 
        self.pc = []
        for cell, sp in enumerate(self.spikes[0]):
            
            if mode == 'spikes':
                sp_angles = self.angle[self.spikes[:,cell]!=0]
                if len(sp_angles) < 3:
                    continue
                mu, std = pcf.CellFields(sp_angles.tolist(), sigma=2.5,angle_len=90, realbin=40) 
            elif mode == 'raw_neuro':
                bin_neuro = pcf.BinRawNeuro(angles = self.angle, neurodata = self.neur[:,cell], Nbin = 40)
                mu, std = pcf.CellFieldsNeuro(bin_neuro, sigma=2.5,bin_len=10, realbin=40)
            else:
                break
            
            if not len(mu):
                continue            
            self.pc.append(PlaceCell(cell))
            self.pc[-1].pf = []
            for i,m in enumerate(mu):
                #each place cell can have multiple place fields              
                pf = PlaceField(mu = m, std = std[i])
                pf.times_in, pf.times_out = spps.getPasses(self.angle, m, std[i]) 
                
                #if there are less than 50% non-spiking passes, discard this field
                siz = []
                for i, tin in enumerate(pf.times_in):
                    siz.append(sum(self.spikes[tin:pf.times_out[i], cell]))
                if np.count_nonzero(siz)/len(siz) < 0.5 and len(siz) < 10:
                    continue
                
                #t_spec is the first time the cells fires in its zone
                #n_spec is cooresponding number of in-zone visit (when the first in_zone spike occures)
                for i, tin in enumerate(pf.times_in):
                    spikes_in_zone = self.spikes[tin:pf.times_out[i], cell]
                    if np.count_nonzero(spikes_in_zone):
                         rel_t = np.nonzero(spikes_in_zone)
                         pf.t_spec = tin + rel_t[0][0]
                         pf.n_spec = i + 1
                         break
                     
                #JUST IN CASE
                if not hasattr(pf, 't_spec'):
                    pf.t_spec = self.n_frames
                    pf.n_spec = len(pf.times_in)
                pf.is_super = False
                self.pc[-1].pf.append(pf)
                
            if not len(self.pc[-1].pf):
                del self.pc[-1]
        return
        
         
    def get_ks_score(self, min_n_spikes = 3):
        self.ks_stat = []
        self.ks_p_score = []
        for cell in range(self.n_cells):
            sp_angles = self.angle[self.spikes[:,cell]!=0]
            #sp_distrib, edges = np.histogram(sp_angles.tolist(), bins = 40, range = (0,360))
            if len(sp_angles) >= min_n_spikes:
                stat, p = kstest(sp_angles, 'uniform', N = len(sp_angles), args = (0.0, 360.0))
                self.ks_stat.append(stat)
                self.ks_p_score.append(-np.log(p))
            else:
                self.ks_stat.append(0)
                self.ks_p_score.append(0)
        return 

    def trim_time(self, t_end, fps = 20):
        if  self.n_frames >= int((t_end - self.time[0])*fps):   
            self.n_frames = int((t_end - self.time[0])*fps)
        else:
            raise ValueError   
 
        fr_end = self.n_frames - 1
        
        #mandatory attribs
        self.time = self.time[0:fr_end]
        self.x = self.x[0:fr_end]
        self.y = self.y[0:fr_end]
        self.neur = self.neur[0:fr_end,:]
        
        if hasattr(self, 'spikes'):  
            self.spikes = self.spikes[0:fr_end,:]  
            
        if hasattr(self, 'markers'):
            self.markers = self.markers[:,0:fr_end]
            
        # for attr in insp.getmembers(self):      
        #     # to remove private and protected 
        #     # functions 
        #     if not attr[0].startswith('_') and not insp.ismethod(attr[1]):  
        #         print(i) 
            
        
            
#UNDER CONSTRUCTION
#    def get_holed_bins(self, step, vid_h, vid_w):
#        #step - size of bin in px; #vid_h/w - height/width of video in px
#        Nx = np.ceil(vid_w/step)
#        Ny = np.ceil(vid_h/step)
#        self.spatial_bin = np.zeros(len(self.time), dtype = int)     #initialization
#        self.spatial_bin = np.fix(self.x/step) + np.fix(self.y/step)*Nx     #binarization
#        occup = np.histogram2d(np.fix(self.x/step), np.fix(self.y/step), bins = [Nx, Ny], range=[[0,Nx], [0,Ny]])
#        true_ind = np.flatnonzero(occup[0]).tolist()           
#        self.spatial_bin = [true_ind.index(spb) for spb in self.spatial_bin.tolist()]


    
def TrimTime(Mouse, t_end):
    if  Mouse.n_frames >= int((t_end - Mouse.time[0])*20):   
        Mouse.n_frames = int((t_end - Mouse.time[0])*20)
    else:
        raise ValueError   
 
    fr_end = Mouse.n_frames - 1
    Mouse.time = Mouse.time[0:fr_end]
    Mouse.angle = Mouse.angle[0:fr_end]
    Mouse.spikes = Mouse.spikes[0:fr_end,:]
    Mouse.neur = Mouse.neur[0:fr_end,:]
    Mouse.neuro_bin = Mouse.neuro_bin[0:fr_end,:]

    return Mouse



def GetRears(Mouse, rear_fname):
    rr = np.genfromtxt(rear_fname, delimiter=',', skip_header=0)
    fr = interpolate.interp1d(rr[:,0], rr[:,1])
    Mouse.rear = [np.sign(fr(t)) for t in Mouse.time]
    return Mouse


def RearStat(Mouse, n_sup, cell):
    rear_t_inz = []
    n_spec = 0
    for i, ti in enumerate(Mouse.times_in[cell]):
        if (i==0 or i>=1 and Mouse.t_spec[n_sup][0] >= Mouse.times_out[cell][i-1])  and Mouse.t_spec[n_sup][0] <= Mouse.times_out[cell][i]:
            n_spec = i
        rear_t_inz.append(np.sum(np.abs(Mouse.rear[ti:Mouse.times_out[cell][i]]))/(Mouse.times_out[cell][i]-ti))
    if n_spec:
        rear_t_bef = sum(rear_t_inz[:n_spec])/n_spec
    else:
        rear_t_bef = 0   
    rear_t_aft = sum(rear_t_inz[n_spec:])/(len(rear_t_inz) - n_spec)
    return rear_t_inz, rear_t_bef, rear_t_aft
    

def ReformFields(Mouse): #from old version to new
    #lists of true cells, super_cells -> Mouse.pc.pf
    Mouse.pc = []
    for cell in range(len(Mouse.spikes[0])):
        pf_pos = [i for i, c in enumerate(Mouse.true_cells) if c==cell]
        if not len(pf_pos):
            continue
        Mouse.pc.append(PlaceCell(cell))
        Mouse.pc[-1].pf = []
                
        for n_pos, pos in enumerate(pf_pos):
            pf = PlaceField(mu = Mouse.mu_list[pos], std = Mouse.std_list[pos])
            pf.is_super = pos in Mouse.super_in_true
            pf.times_in = Mouse.times_in[pos]
            pf.times_out = Mouse.times_out[pos]
            if pf.is_super:
                pf.t_spec = Mouse.t_spec[Mouse.super_in_true.index(pos)]
            Mouse.pc[-1].pf.append(pf)
    return Mouse

def FieldsToList(Mouse):   #  for backward compatibility purposes
    Mouse.true_cells = []
    Mouse.mu_list = []
    Mouse.std_list = []
    Mouse.times_in = []
    Mouse.times_out = []
    
    for pc in Mouse.pc:
        for pf in pc.pf:
            Mouse.true_cells.append(pc.cell_num)
            Mouse.mu_list.append(pf.mu)
            Mouse.std_list.append(pf.std)
            Mouse.times_in.append(pf.times_in)
            Mouse.times_out.append(pf.times_out)
            
    return Mouse        


def PoolMice(mice):
    t_list = []
    n_list = []
    for ms in mice:
        for pc in ms.pc:
            for pf in pc.pf:
                t_list.append(pf.t_spec/20)
                n_list.append(pf.n_spec)
    return [t_list, n_list]

            
def CheckObj(strg):
    return strg.find("obj") > 0 or strg.find("1") > 0 or strg.find("Obj") > 0



def Drop_Wrong_Points(x, y, cen_data, d):
     #xc, yc - coordinates of center (px), r - horisontal raduis of circle(rx), k = ry/rx, d - allowed displacement
     xc = (cen_data[0] + cen_data[1])*0.5
     yc = (cen_data[2] + cen_data[3])*0.5
     r = (cen_data[1] - cen_data[0])*0.5
     k = (cen_data[3] - cen_data[2])/(cen_data[1] - cen_data[0])
     
     for i in range(len(x) - 1):
         if ~np.isnan(x[i+1]) and ~np.isnan(y[i+1]) and not i == len(x)-2:
             ri = euclidean([x[i+1],y[i+1]/k], [xc, yc/k])
             if ri > r + d or ri < r - d/2:
                 x[i+1] = np.nan
                 continue
             if ~np.isnan(x[i]) and ~np.isnan(y[i]):
                 ds = euclidean([x[i+1],y[i+1]/k], [x[i], y[i]/k])
                 if ds > d*10:
                     x[i+1] = np.nan
                     continue
                 
     return x
                     
                     
         





#    def get_obj_behavior(self, b_path, b_skip_rows, b_col):
#        #scores acts as they are in behav file, even if they are overlapping; recognizes native START and STOP marks
#        self.behav = []
#        data =  np.genfromtxt(b_path, delimiter=',', skip_header=b_skip_rows)
#        btype = 0 #archive of behavior types
#        
#        if float(data[0][0]) <= self.time[0]: #initial behavior act
#            btype = int(CheckObj(data[0][b_col]))
#        else:
#            btype = 0
#        
#        #loop for strings in behavior file
#        for j in range(len(data)):
#            while len(self.behav) < len(self.time) and self.time[len(self.behav)] < float(data[j][0]):
#                self.behav.append(btype)
#            if ~CheckObj(data[j][b_col]):
#                continue
#            if float(data[j][0]) > self.time[-1]:
#                break
#            if data[j][b_col+3] == 'START':
#                btype = 1
#            if data[j][b_col+3] == 'STOP':
#                btype = 0
#        #the last interval    
#        while len(self.behav) < len(self.time):
#            self.behav.append(btype)
#        
#    def get_mult_behavior(self, b_path, b_skip_rows, b_col, default_behav_code):
#        #scores acts as they are in behav file, even if they are overlapping; recognizes native START and STOP marks
#        self.behav = []
#        book = xlrd.open_workbook(b_path, on_demand = True)
#        sheet = book.sheet_by_index(0)
#        data = [[str(c.value) for c in sheet.row(j)] for j in range(b_skip_rows,sheet.nrows)]
#        book.release_resources()
#        del book
#    
#        btype = [] #archive of behavior types
#        
#        if float(data[0][0]) <= self.time[0]: #initial behavior act
#            btype.append(float(data[0][b_col]))
#        else:
#            btype.append(default_behav_code)
#        
#        #loop for strings in behavior file
#        for j in range(len(data)):
#            while len(self.behav) < len(self.time) and self.time[len(self.behav)] < float(data[j][0]):
#                self.behav.append(btype[:])
#            if float(data[j][b_col]) ==  default_behav_code:
#                continue
#            if float(data[j][0]) > self.time[-1]:
#                break
#            if data[j][b_col+3] == 'START' and float(data[j][b_col]) not in btype:
#                btype.append(float(data[j][b_col]))
#                while default_behav_code in btype:
#                    btype.remove(default_behav_code)
#            if data[j][b_col+3] == 'STOP' and float(data[j][b_col]) in btype:
#                self.behav.append(btype[:])
#                btype.remove(float(data[j][b_col]))
#                if not len(btype):
#                    btype.append(default_behav_code)
#        #the last interval    
#        while len(self.behav) < len(self.time):
#            self.behav.append(btype[:])
#            
#    def get_behavior(self, b_path, b_skip_rows, b_col):
#        #scores consequent acts, in case of overlapping with respect to priority; positive figures = start, negative = stop
#        self.behav = np.zeros(len(self.time))
#        book = xlrd.open_workbook(b_path, on_demand = True)
#        sheet = book.sheet_by_index(0)
#        data = [[str(c.value) for c in sheet.row(j)] for j in range(b_skip_rows,sheet.nrows)]
#        book.release_resources()
#        del book
#    
#        btype = [0] #archive of behavior types
#        for j in range(len(data)):
#            if float(data[j][b_col]) > 0:
#                btype.append(float(data[j][b_col]))
#            else:
#                btype.remove(-float(data[j][b_col]))
#            self.behav[np.array(self.time) >= float(data[j][0])] = max(btype)
##        return
#    
#       
#    def get_mi_score(self):
#        self.mi_score = np.zeros(self.neur.shape[1], dtype = np.float32)
#        for i in range(self.neur.shape[1]):
#            nbi = self.neuro_bin[:,i]
##            self.mi_score[i] = adjusted_mutual_info_score(self.neuro_bin[:,i], self.spatial_bin)
#            self.mi_score[i] = mutual_info_score(nbi[nbi>0], self.spatial_bin[nbi > 0])
#        return
#            
#    def verify_cell(self, cell, N_shuff):
#        count = 0
#        ang_shuffled = self.spatial_bin
#        for i in range(N_shuff):
#            np.random.shuffle(ang_shuffled)
#            mi = adjusted_mutual_info_score(self.neuro_bin[:,cell], ang_shuffled)
#            if mi > self.mi_score[cell]:
#                count += 1
#        return count <= 0.05*N_shuff
#    
#    def verify_cells(self, N_shuff):  
#        self.super_cells = []
#        for i in range(self.neur.shape[1]):
#            if self.verify_cell(cell = i, N_shuff = N_shuff):
#                self.super_cells.append(i)
#        return


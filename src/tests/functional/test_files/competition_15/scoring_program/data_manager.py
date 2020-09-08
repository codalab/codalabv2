# Functions performing various input/output operations saptiontemporal data
# Main contributors: Stephane Ayache, Isabelle Guyon and Lisheng Sun

# Date: January 2017 - May 2019


# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, SEE.4C, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 



import numpy as np # We will use numoy arrays
import yaml
try:
	import cPickle as pickle
except:
	import pickle
import os # use join to concatenate strings in directory names
import time
from data_io import vprint
from glob import glob as ls
from os import remove as rm
from os.path import join

class DataManager:
    
    ''' This class aims at loading, saving, and displaying data.
    
    Data members:
    datatype = one of "input" or "output"
    X = data matrix, samples in lines (time index increasing with line), features in columns.
    t = time index. The time index may show ruptures e.g. 0, 1, 2, 3, 0, 1, 2, 3; indicating cuts.
    col_names = names of the variables of the data matrix.
    ycol0 = index where variables to be predicted start.
    t0 = index where predictions start (after end of trainign data).
    now = index of the present time.
    stride = number of steps to skip until next prediction.
    horizon = number of steps to be predicted.
     
    Methods defined:
    __init__ (...)
        D.__init__([(feature, value)]) -> void		
        Initialize the data members with the tuples (feature, value) given as argument. An unlimited number of tuples can be passed as argument.
        If input_dir is given, calls loadTrainData.
        
    loadData(data_dir)
        D.loadData (input_dir, max_samples=float('inf'), verbose="True") -> success		
        Load the training and evaluation samples found in directories data_dir/training
        and data_dir/evaluation. Each subdirectory may contain multiple files,
        which will be loaded in alphabetical order.
        Returns success="True/False".       
        
    getInfo()
        D.getInfo () -> string	
        Pretty prints information about the object.
        
    getHistoricalData()
        D.getHistoricalData () -> X, t	
        Available training data up to self.now (x and y columns).
        
    getFutureOutcome()
        D.getFutureOutcome () -> Y, t	
        Ground truth for future data to be predicted (y columns only).

    saveData(filename) 
        save read data (array X, T) to pickle, for faster reload	.
        
    reloadData(filename)
        reload data from pickle.
        
    appendData(X, t)
        append samples (useful for predictions)
        
    resetTime()
        reset the time index "now" to the begining of the evaluation data.
        
    '''
	
    def __init__(self, datatype="input", data_file="", verbose=False, cache_file=""):
        '''Constructor'''
        self.version = "1"
        self.datatype = datatype 
        self.verbose = verbose
        self.cache_file=cache_file # To save/reload data in binary format (only if not empty)
        if not cache_file: 
            self.use_pickle = False
        else:
            self.use_pickle = True
        self.X = np.array([])
        self.t = np.array([])
        self.col_names = []
        self.ycol0 = 0
        self.t0 = 0
        self.now = 0
        self.stride = 0
        self.horizon = 0
        vprint(self.verbose, "DataManager :: Version = " + self.version)
        if data_file:
            self.loadData(data_file)
           
    def __repr__(self):
        return "DataManager :\n\t" + str(self.X.__repr__) + "\n\t" + str(self.t.__repr__)

    def __str__(self):
        val = "DataManager :: ========= Info =========\n" + self.getInfo()
        return val
  
    def getInfo(self):
        '''A nice string with information about the data.'''
        val = "Version = {}\n".format(self.version)
        val = val + "Data type = {}\n".format(self.datatype)
        val = val + "Stride = {}\n".format(self.stride)
        val = val + "Horizon = {}\n".format(self.horizon)
        val = val + "Num training samples = {}\n".format(self.t0)
        val = val + "Num evaluation samples = {}\n".format(self.t.shape[0]-self.t0)
        try:
            val = val + "Num input variables = {}\n".format(self.X.shape[1])
            val = val + "Input variables = {}\n".format(self.col_names)
            val = val + "Num output variables = {}\n".format(self.X.shape[1]-self.ycol0)
            val = val + "Output variables = {}\n".format(self.col_names[self.ycol0:])
        except:
            val = val + "Num input variables = NA\n"
            val = val + "Num output variables = NA"                
        return val
        
    def loadData(self, data_dir=""): 
        ''' Get the data from csv files.'''
        success = True
        data_reloaded = False
        vprint(self.verbose, "DataManager :: ========= Reading training data from " + data_dir)
        start = time.time()
        if self.use_pickle and self.reloadData(self.cache_file):
            # Try to reload the file from a pickle
            data_reloaded = True # Turn "success" to false if there is a problem.
        else:
            # Load metadata
            metadata = yaml.load(open(join(data_dir, 'metadata'), 'r'))
            self.stride = metadata['stride']
            self.horizon = metadata['horizon']
            self.ycol0 = metadata['ycol0']
            # Load the training data data into X and t.
            data_file_list = sorted(ls(join(data_dir, "training", "*.csv")))
            vprint(self.verbose, "DataManager :: ========= Load data from files:")
            vprint(self.verbose, data_file_list)
            header = np.genfromtxt(data_file_list[0], delimiter=',', max_rows=1, names=True)
            self.col_names = header.dtype.names[1:]
            for data_file in data_file_list:
                data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
                self.t = np.append(self.t, data[:,0])
                if self.X.shape[0]==0:  
                    self.X = data[:,1:]
                else:
                    self.X = np.append(self.X, data[:,1:], axis=0) 
            self.t0 = self.t.shape[0]
            # Append the evaluation data data to X and t.
            data_file_list = sorted(ls(join(data_dir, "evaluation", "*.csv")))
            vprint(self.verbose, data_file_list)
            for data_file in data_file_list:
                data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
                self.t = np.append(self.t, data[:,0])
                self.X = np.append(self.X, data[:,1:], axis=0) 
                   
        if self.use_pickle and not data_reloaded:
            # Save data as a pickle for "faster" later reload
            self.saveData(self.cache_file)
            
        end = time.time()
        if len(self.X)==0:
            success = False 
            vprint(self.verbose, "[-] Loading failed")
        else:
            vprint(self.verbose, "[+] Success, loaded %d samples in %5.2f sec" % (self.t.shape[0], end - start))
        self.resetTime()
        return success
    
    def appendData(self, X, t): 
        ''' Append a data sample (useful for predictions). '''
        vprint(self.verbose, "DataManager :: ========= Appending {} frame(s)".format(X.shape[0]))
        if X.shape[0] != t.shape[0]:
            vprint(self.verbose, "[-] Inconsistent dimensions X.len={} t.len={}".format(X.shape[0], t.shape[0]))
        self.t = np.append(self.t, t)
        if self.datatype=='output':
            rng=range(self.ycol0,X.shape[1])
        else:
            rng=range(X.shape[1])
        if self.X.shape[0]==0:
            self.X = X[:,rng]
        else:
            self.X = np.append(self.X[:,rng], X[:,rng], axis=0) 
        return 
    
    def resetTime(self):
        ''' Reset time as the begining of the evaluation samples.'''
        self.now = self.t0
        
    def getHistoricalData(self):
        ''' Provide the all samples revealed until now, usable for training. 
        Move by self.stride samples at a time. Returns [] if data are exhausted.'''
        end = self.now-1
        if end > self.t.shape[0]:
            return np.array([]), np.array([])
        else:
            self.now = self.now + self.stride
            return self.X[0:end,:], self.t[0:end]
        
    def getFutureOutcome(self):
        ''' Get the next self.horizon time stamps and output only Y variables to be predicted. 
        Move by self.stride samples at a time. Returns [] if data are exhausted.'''
        start = self.now
        end = self.now + self.horizon
        self.now = self.now + self.stride
        return self.X[start:end,self.ycol0:], self.t[start:end]
    
    def getPredictions(self):
        ''' Get a chunck of self.horizon time stamps and of predicted values. 
        Move by self.horizon samples at a time. Returns [] if data are exhausted.'''
        start = self.now
        t = self.t[start:start + self.horizon]
        end_idx = np.where(t[1:]-t[:-1]<=0)[0]
        if end_idx.shape[0]>0: 
            horizon = end_idx[0] +1
        else:
            horizon = self.horizon
        end = self.now + horizon
        self.now = self.now + horizon
        return self.X[start:end,self.ycol0:], self.t[start:end]
        
    def saveData(self, filename, format="pickle"):
        ''' Save data in pickle format or csv formal. 
        '''
        if not filename.endswith(format):
            filename=filename + '.' + format
            vprint(self.verbose, "[-] filename must end with " + format )
        vprint(self.verbose, "DataManager :: ========= Saving data to " + filename)
        start = time.time()
        try:
            if format=='pickle':
                with open(filename, 'wb') as f:
                    vprint(self.verbose, "DataManager :: Saving as pickle")
                    dict_to_save = {key:self.__dict__[key] for key in self.__dict__.keys() if not key in ['X', 't']}
                    dict_to_save['X'] = self.X 
                    dict_to_save['t'] = self.t 
                    pickle.dump(dict_to_save, f, 2)
            else:
                with open(filename, 'w') as f:
                    vprint(self.verbose, "DataManager :: Saving as csv")
                    f.write("Date")
                    for nm in self.col_names: f.write("," + nm)
                    f.write("\n")
                    for i in range(self.t.shape[0]):
                        f.write("{:g}".format(self.t[i]))
                        for j in range(self.X.shape[1]):
                            f.write(",{:g}".format(self.X[i,j]))
                        f.write("\n")
            success = True
        except Exception as e: 
            vprint (self.verbose, e)
            success = False
        end = time.time()
        vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
        return success

    def reloadData(self, filename, format="pickle"):
        ''' Reload data in pickle or csv format.
            Warning: csv format will not reload medatada, suitable only for predictions.
        '''
        if not filename.endswith(format):
            filename = filename + '.' + format
        vprint(self.verbose, "DataManager :: ========= Attempting to reload data from " + filename)
        start = time.time()
        success = True
        temp =[]
        try:
            if format=='pickle':
                with open(filename, 'rb') as f:
                    temp = pickle.load(f)
                    for key in self.__dict__.keys():
                        self.__dict__[key] = temp[key]
            elif format=='csv' and self.datatype=='output':
                data = np.genfromtxt(filename, delimiter=',', skip_header=1)
                self.t = data[:,0]
                self.X = data[:,1:]
            else: 
                vprint(self.verbose, "[-] Wrong file format " + format + " for " + self.datatype)
                success = False
        except Exception as e: 
            vprint (self.verbose, e)
            success = False     
        if success:
            end = time.time()
            vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
        self.resetTime()
        return success    


    def show(self, start=0, end=0, step=1, transpose=False):
        ''' Display frames graphically in a nice way.
            start and end and the first and last frame. step is the stride.'''
        import seaborn as sns; sns.set()
        import matplotlib.pyplot as plt
        plt.subplots(figsize=(20,15))
        import pandas as pd
        nmax = len(self.X)-1
        if start==0 and end==0:
            end=nmax
        if end>nmax: end=nmax
        if end<start: end=start
        frame_index_to_display = range(start, end+1, step)
        df = pd.DataFrame(self.X[frame_index_to_display,:], index=self.t[frame_index_to_display], columns=self.col_names)
        if transpose:
            sns.heatmap(df.transpose(), xticklabels=True, yticklabels=True, square=True)
        else:
            sns.heatmap(df, xticklabels=True, yticklabels=True, square=True)
        
# =========================== TEST PROGRAM ================================

if __name__=="__main__":	
    
    print("** CHECK WE HAVE DATA **")
    data_dir = "../sample_data"
    print(ls(join(data_dir, '*/*')))
    
    print("** REMOVE THE CACHE **")
    cfile= "titi"
    try: 
        rm(cfile + ".pickle")
    except:
        print("No cache found")
       
    print("** CALL THE CONTRUCTOR **")
    D = DataManager(verbose=True, cache_file=cfile)
    print("** OBJECT SHOULD BE EMPTY **")
    print(D)
    
    print("** LOAD THE DATA **")
    D.loadData(data_dir)
    print("** OBJECT SHOULD NO LONGER BE EMPTY **")
    print(D)

    print("** RE-LOAD THE DATA FROM PICKLE **")
    D.loadData(data_dir)
    print("** OBJECT SHOULD BE THE SAME **")
    print(D)
    
    print("** VISUALIZE DATA **")
    D.show(transpose=False)

    print("** SUCCESSIVE HISTORICAL DATA **")    
    print("** NOW = {} **".format(D.now))   
    X, t = D.getHistoricalData()
    while not t.shape[0]==0:
        print(t)
        #print(X)
        print("----------------")
        X, t = D.getHistoricalData()
    
    print("** SUCCESSIVE FUTURE OUTCOMES **")  
    D.resetTime()
    print("** NOW = {} **".format(D.now))   
    X, t = D.getFutureOutcome()
    while not t.shape[0]==0:
        print(t)
        print(X)
        print("----------------")
        X, t = D.getFutureOutcome()
        
    print("** SAVE AS CSV **")  
    D.resetTime()
    X, t = D.getFutureOutcome()
    Dout = DataManager(datatype='output', verbose=True)
    Dout.col_names=D.col_names[D.ycol0:]
    Dout.t=t
    Dout.X=X
    Dout.saveData("toto", format="csv")
    
    print("** RELOAD FROM CSV **")  
    Dout2 = DataManager(datatype='output', verbose=True)
    Dout2.reloadData("toto", format="csv")
    print(Dout2)
    
       
# Functions performing various input/output operations for the ChaLearn AutoML challenge

# Main contributors: Arthur Pesah and Isabelle Guyon, August-October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np
import os
import shutil
from scipy.sparse import * # used in data_binary_sparse 
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
from sys import version
from glob import glob as ls
from os import getcwd as pwd
#from pip import get_installed_distributions as lib
import csv

# ================ Small auxiliary functions =================

swrite = stderr.write

if (os.name == "nt"):
       filesep = '\\'
else:
       filesep = '/'
       
def write_list(lst):
    ''' Write a list of items to stderr (for debug purposes)'''
    for item in lst:
        swrite(item + "\n") 
        
def print_dict(verbose, dct):
    ''' Write a dict to stderr (for debug purposes)'''
    if verbose:
        for item in dct:
            print(item + " = " + str(dct[item]))

def mkdir(d):
    ''' Create a new directory'''
    if not os.path.exists(d):
   		os.system("mkdir " + d)
   		print("done")
        #os.makedirs(d)
        
def mvdir(source, dest):
    ''' Move a directory'''
    if os.path.exists(source):
        os.rename(source, dest)
        
def cp(source, dest):
    ''' Copy a file'''
    os.system('cp ' + source + ' ' + dest)

def rmdir(d):
    ''' Remove an existingdirectory'''
    if os.path.exists(d):
        shutil.rmtree(d)
        
def vprint(mode, t):
    ''' Print to stdout, only if in verbose mode'''
    if(mode):
            print(t) 
        
# ================ Output prediction results and prepare code submission =================
        
def write(filename, predictions):
    ''' Write prediction scores in prescribed format'''
    with open(filename, "w") as output_file:
		for row in predictions:
			if type(row) is not np.ndarray and type(row) is not list:
				row = [row]
			for val in row:
				output_file.write('{0:g} '.format(float(val)))
			output_file.write('\n')

def zipdir(archivename, basedir):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-4:]!='.zip':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
                    z.write(absfn, zfn)
                    
def zip_submission(archivename, basedir=""):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    if not basedir: basedir = pwd()
    assert os.path.isdir(basedir)
    util_dir = os.path.join(basedir, 'utilities')
    code_dir = os.path.join(basedir, 'sample_code')
    cache_dir = os.path.join(basedir, 'cache')
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
    	files = ['predict.sh', 'predictSpatioTemporal.py']
    	for fn in files:
            absfn = os.path.join(basedir, fn)
            zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
            z.write(absfn, zfn)

        for root, dirs, files in os.walk(code_dir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-3:]=='.py':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(code_dir)+len(os.sep):] #XXX: relative path
                    z.write(absfn, zfn)

        for root, dirs, files in os.walk(util_dir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-3:]=='.py':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(util_dir)+len(os.sep):] #XXX: relative path
                    z.write(absfn, zfn)

        for root, dirs, files in os.walk(cache_dir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-5:]=='.json' or fn[-5:]=='.hdf5':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(cache_dir)+len(os.sep):] #XXX: relative path
                    z.write(absfn, zfn)
                    
# ================ Display directory structure and code version (for debug purposes) =================
      
def show_dir(run_dir):
	print('\n=== Listing run dir ===')
	write_list(ls(run_dir))
	write_list(ls(run_dir + '/*'))
	write_list(ls(run_dir + '/*/*'))
	write_list(ls(run_dir + '/*/*/*'))
	write_list(ls(run_dir + '/*/*/*/*'))
      
def show_io(input_dir, output_dir):    
	import yaml
	swrite('\n=== DIRECTORIES ===\n\n')
	# Show this directory
	swrite("-- Current directory " + pwd() + ":\n")
	write_list(ls('.'))
	write_list(ls('./*'))
	write_list(ls('./*/*'))
	swrite("\n")
	
	# List input and output directories
	swrite("-- Input directory " + input_dir + ":\n")
	write_list(ls(input_dir))
	write_list(ls(input_dir + '/*'))
	write_list(ls(input_dir + '/*/*'))
	write_list(ls(input_dir + '/*/*/*'))
	swrite("\n")
	swrite("-- Output directory  " + output_dir + ":\n")
	write_list(ls(output_dir))
	write_list(ls(output_dir + '/*'))
	swrite("\n")
        
    # write meta data to sdterr
	swrite('\n=== METADATA ===\n\n')
	swrite("-- Current directory " + pwd() + ":\n")
	try:
		metadata = yaml.load(open('metadata', 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
	except:
		swrite("none\n");
	swrite("-- Input directory " + input_dir + ":\n")
	try:
		metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
		swrite("\n")
	except:
		swrite("none\n");
	
def show_version():
	# Python version and library versions
	swrite('\n=== VERSIONS ===\n\n')
	# Python version
	swrite("Python version: " + version + "\n\n")
	# Give information on the version installed
	swrite("Versions of libraries installed:\n")
	map(swrite, sorted(["%s==%s\n" % (i.key, i.version) for i in lib()]))
 
 # Compute the total memory size of an object in bytes

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

    # write the results in a csv file
def platform_score ( basename , mem_used ,n_estimators , time_spent , time_budget ):
# write the results and platform information in a csv file (performance.csv)
    import psutil
    import platform
    with open('performance.csv', 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        #['Data name','Nb estimators','System', 'Machine' , 'Platform' ,'memory used (Mb)' , 'number of CPU' ,' time spent (sec)' , 'time budget (sec)'],
        data = [
        [basename,n_estimators,platform.system(), platform.machine(),platform.platform() , float("{0:.2f}".format(mem_used/1048576.0)) , str(psutil.cpu_count()) , float("{0:.2f}".format(time_spent)) ,    time_budget ]
        ]
        a.writerows(data)


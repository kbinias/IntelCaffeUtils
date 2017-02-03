# Version 1.2
# krzysztof.binias@intel.com

# Usage:
# Use -l option with mpirun
#  iterinfo.py intelcaffe_output.log
#  iterinfo.py *.log

import sys
import os
import operator
import math
import re
import glob
from datetime import datetime, date, time

######################################### Functions #############################################

#I0201 01:36:31.795657 121166 sgd_solver.cpp:145] Iteration 0, lr = 0.08
#[5] I0203 01:54:58.087530 94321 solver.cpp:316] Iteration 480, loss = 7.02164
def prepare_line(line_str):
  line_str = line_str.replace(",", "")
  line_str = line_str.replace("=", "")
  line_str = line_str.replace("W", str(datetime.now().year), 1)

  # Exchange first I
  if line_str.count('I') == 2:
    line_str = line_str.replace("I", str(datetime.now().year), 1)

  return line_str

def get_time(vals):
  d = None
  if len(vals) < 3: return None
  try:
    if vals[0].startswith('[') == True: #[0] I0719 07:09:31.850666 18289 solver.cpp:239] Iteration 6680, loss = 6.40777
      d = datetime.strptime(vals[1] + " " + vals[2], "%Y%m%d %H:%M:%S.%f")
    else:
      d = datetime.strptime(vals[0] + " " + vals[1], "%Y%m%d %H:%M:%S.%f")
    d = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
  except ValueError:
    return None

  return d

def find_between(s, first, last, beg):
  try:
    start = s.index(first, beg) + len( first )
    end = s.index(last, start)
    return s[start:end]
  except ValueError:
    return ""

def get_params(line, vals):

  time = None
  rank = -1
  iter = -1
  loss = -1

  if len(vals) < 8: return time, rank, iter, loss
  if line.find('Iteration') < 0: return time, rank, iter, loss
  if line.find('loss') < 0: return time, rank, iter, loss

  time = get_time(vals)

  start_idx = line.find('[')

  if start_idx == 0: #[0] I0719 07:09:31.850666 18289 solver.cpp:239] Iteration 6680 loss 6.40777
    rank = find_between(line, '[', ']', start_idx)
    iter = vals[6]
    loss = vals[8]
  else: #I0722 01:20:20.228528 165631 solver.cpp:241] Iteration 25760 loss 11.0499
    rank = vals[2]
    iter = vals[5]
    loss = vals[7]

  return time, int(rank), int(iter), float(loss)

def get_args(argv):
  file_in_name = None
  timedelta_format = "m"

  file_in_name = argv[1]

  return file_in_name, timedelta_format

# Get indexes for changed ranks
def get_changed_rank_idxs(arr):
  changed_rank_idxs = []
  last_rank = -1
  for idx, row in enumerate(arr):
    if last_rank != arr[idx][0]:
      changed_rank_idxs.append(idx)
      last_rank = arr[idx][0]

  return changed_rank_idxs

def compute_timediff(arr):
  # Rank Iteration Time Loss +TimeDiff

  last_rank = -1
  last_time = datetime.now()
  avg_timedelta_arr = []
  timedelta_sum = 0
  timedelta_count = 0

  for idx, row in enumerate(arr):
    if last_rank != row[0]:
      if timedelta_count > 0: avg_timedelta_arr.append( timedelta_sum/timedelta_count )
      row.append(row[0])
      last_rank = row[0]
      last_time = row[2]
      timedelta_sum = 0
      timedelta_count = 0
    else:
      timedelta = (row[2] - last_time).seconds

      if timedelta > 0:
        timedelta_sum += timedelta
        timedelta_count = timedelta_count + 1

      row.append( timedelta )

    last_time = row[2]

  if timedelta_count > 0: avg_timedelta_arr.append( timedelta_sum/timedelta_count )

  return avg_timedelta_arr # Avarage time delta for each rank

def explode_time(seconds):
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)

  return h, m, s

def format_time(seconds, format='hms'):
  h, m, s = explode_time(seconds)

  if format == 'hms':
    return "%d:%02d:%02d" % (h, m, s)
  elif format == 'ms':
    return "%02d:%02d" % (m, s)
  elif format == 's':
    return "%02d" % (seconds)

  return "%d:%02d:%02d" % (h, m, s)

# Update net params dictionary
def update_train_params_dict(multinode, train_params_dict, vals):

  if len(vals) <= 1: return 1;

  base_idx = 0
  if multinode == True: base_idx = 1

  if vals[base_idx] == 'batch_size:' and train_params_dict.has_key('batch_size') == False:
    train_params_dict['batch_size'] = vals[base_idx+1]
  elif vals[base_idx] == 'name:' and train_params_dict.has_key('name') == False: # Network name
    train_params_dict['name'] = vals[base_idx+1]
  elif vals[base_idx] == 'max_iter:' and train_params_dict.has_key('max_iter') == False:
    train_params_dict['max_iter'] = vals[base_idx+1]
  elif vals[base_idx] == 'momentum:' and train_params_dict.has_key('momentum') == False:
    train_params_dict['momentum'] = vals[base_idx+1]
  elif vals[base_idx] == 'base_lr:' and train_params_dict.has_key('base_lr') == False:
    train_params_dict['base_lr'] = vals[base_idx+1]
  elif vals[base_idx] == 'image_data_param' and train_params_dict.has_key('data_source') == False:
    train_params_dict['data_source'] = 'image_data_param'
  elif vals[base_idx] == 'data_param' and train_params_dict.has_key('data_source') == False:
    train_params_dict['data_source'] = 'data_param'
  elif vals[base_idx] == 'dummy_data_param' and train_params_dict.has_key('data_source') == False:
    train_params_dict['data_source'] = 'dummy_data_param'
  elif vals[base_idx] == 'shuffle:' and train_params_dict.has_key('shuffle') == False:
    train_params_dict['shuffle'] = vals[base_idx+1]
  elif vals[base_idx] == 'engine:' and train_params_dict.has_key('engine') == False:
    train_params_dict['engine'] = vals[base_idx+1]

  return 0

def get_val_from_dict(dict, key):
  
  if dict.has_key(key) == True:
    return dict[key];

  return "none"

######################################################################################

file_in_name, timedelta_format = get_args(sys.argv)

file_list = glob.glob(file_in_name)

for file_in_name in file_list:

  multinode = False
  arr = []

  # Start reading data

  start_time = None
  last_time = None
  train_params_dict = {}

  # Fill data array
  with open(file_in_name, "r") as file_in:
    for line in file_in:

      #I0201 01:36:31.795657 121166 sgd_solver.cpp:145] Iteration 0, lr = 0.08
      #[5] I0203 01:54:58.087530 94321 solver.cpp:316] Iteration 480, loss = 7.02164
      line = prepare_line(line)
      if(line.startswith('[', 0, 1)):
        multinode = True

      vals = line.split()

      update_train_params_dict(multinode, train_params_dict, vals)

      if len(vals) < 8:
        continue

      last_time, rank, iter, loss = get_params(line, vals)

      if start_time == None:
        start_time = last_time

      if rank < 0 or iter < 0 or loss < 0:
        continue

      #Rank Iteration Time Loss
      arr_row = [ rank, iter, last_time, loss ]
      arr.append(arr_row)

  if len(arr) == 0:
    print("Array is empty. No enough data to compute the statistics")
    continue

  # Sort: Rank Time
  arr = sorted(arr, key=operator.itemgetter(0, 2))

  last_time = arr[len(arr)-1][2]

  # Iter step
  iter_step = arr[len(arr)-1][1] - arr[len(arr)-2][1]

  # Last loss
  last_loss = arr[len(arr)-1][3]

  # Compute iters time
  avg_timedelta_arr = compute_timediff(arr)

  seconds = (last_time - start_time).total_seconds()
  h, m, s = explode_time(seconds)

  formated_arr = []
  for idx, row in enumerate(avg_timedelta_arr):
    formated_arr.append( format_time(row,'s') )

  print("------------------------------------------------------------------")
  print("File name: %s" % (file_in_name))
  print("Model name: %s" % train_params_dict['name'])
  print("Train params: engine: %s, batch_size: %s, max_iter: %s, base_lr: %s, shuffle: %s, momentum: %s, data_source: %s" % (
    get_val_from_dict(train_params_dict,'engine'), 
    get_val_from_dict(train_params_dict,'batch_size'), 
    get_val_from_dict(train_params_dict,'max_iter'), 
    get_val_from_dict(train_params_dict,'base_lr'), 
    get_val_from_dict(train_params_dict,'shuffle'), 
    get_val_from_dict(train_params_dict,'momentum'), 
    get_val_from_dict(train_params_dict,'data_source')) )
  print("Start time: %s" % (start_time))
  print("End time: %s" % (last_time))
  print("Total time: %d:%02d:%02d" % (h, m, s))

  # Average iter time
  if len(avg_timedelta_arr) == 0:
    print("Average iter time is empty. No enough data to compute the statistics")
  else:
    if(multinode): print("Ranks: %d, iter step: %d" % (len(formated_arr), iter_step))
    if(multinode): print("Average iters time by ranks: %s" % (formated_arr))
    aver_iter_time = sum(avg_timedelta_arr)/len(avg_timedelta_arr)
    print("Average iters time: %s" % (aver_iter_time))
    estimate_learning_time_sec = aver_iter_time * (int(get_val_from_dict(train_params_dict,'max_iter'))/iter_step)
    print("Estimate learning time: %s" % ( format_time( estimate_learning_time_sec ,'hms') ))

  print("Last loss: %s" % ( last_loss ))
  print("Last iter: %s" % ( arr[len(arr)-1][1] ))

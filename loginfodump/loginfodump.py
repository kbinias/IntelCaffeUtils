#Usage:
# loginfodump.py single input_log_file process_id not_smaller_then_iters time_unit(s,m,h)
# loginfodump.py loss input_log_file not_smaller_then_iters
# loginfodump.py timediff input_log_file not_smaller_then_iters m
# program_mode: single - single process, loss - all procesess with loss, timediff - all procesess with time diff

import sys
import os
import operator
import math
import re
from datetime import datetime, date, time

######################################### Functions #############################################

#I0702 16:44:35.539449 111918 solver.cpp:239] Iteration 90680, loss = 1.90332
def prepare_line(line_str):
  line_str = line_str.replace(",", "")
  line_str = line_str.replace("=", "")
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

  time = get_time(vals)

  start_idx = line.find('[')

  if start_idx == 0: #[0] I0719 07:09:31.850666 18289 solver.cpp:239] Iteration 6680 loss 6.40777
    rank = find_between(line, '[', ']', start_idx)
    iter = vals[6]
    loss = vals[8]
  elif start_idx > 0: #I0722 01:20:20.228528 165631 solver.cpp:241] [4] Iteration 25760 loss 11.0499
    rank = find_between(line, '[', ']', start_idx)
    iter = vals[6]
    loss = vals[8]
  else: #I0722 01:20:20.228528 165631 solver.cpp:241] Iteration 25760 loss 11.0499
    rank = vals[2]
    iter = vals[5]
    loss = vals[7]

  return time, rank, iter, loss

def get_args(argv):
  program_mode = None
  file_in_name = None
  process_ids = None
  not_smaller_then_iters = None
  timedelta_format = None

  program_mode = argv[1]
  
  if program_mode == 'single': # loginfodump.py single input_log_file process_id not_smaller_then_iters time_unit(s,m,h)
    file_in_name = argv[2]
    process_ids = argv[3].split(",")
    not_smaller_then_iters = int(argv[4])
    timedelta_format = argv[5]
  elif program_mode == 'loss': # loginfodump.py loss input_log_file not_smaller_then_iters
    file_in_name = argv[2]
    not_smaller_then_iters = int(argv[3])
  elif program_mode == 'timediff': # loginfodump.py timediff input_log_file not_smaller_then_iters time_unit(s,m,h)
    file_in_name = argv[2]
    not_smaller_then_iters = int(argv[3])
    timedelta_format = argv[4]

  return program_mode, file_in_name, process_ids, not_smaller_then_iters, timedelta_format

# Get indexes for changed ranks
def get_changed_rank_idxs(arr):
  changed_rank_idxs = []
  last_rank = -1
  for idx, row in enumerate(arr):
    if last_rank != arr[idx][0]:
      changed_rank_idxs.append(idx)
      last_rank = arr[idx][0]

  return changed_rank_idxs

def get_timedelta(timedelta):
  if timedelta_format == "m":
    return float(format(timedelta/60.0, '.2f'))
  elif timedelta_format == "h":
    return float(format(timedelta/3600.0, '.2f'))

  return timedelta

def compute_timediff(arr):
  # Rank Iteration Time Loss +TimeDiff

  last_rank = -1
  last_time = datetime.now()
  avg_timedelta_arr = []
  timedelta_sum = 0
  timedelta_count = 0

  for idx, row in enumerate(arr):
    if last_rank != row[0]:
      row.append(0)
      last_rank = row[0]
      if timedelta_count > 0: avg_timedelta_arr.append( get_timedelta(timedelta_sum/timedelta_count) )
      timedelta_sum = 0
      timedelta_count = 0
    else:
      timedelta = (row[2] - last_time).seconds

      if timedelta > 0:
        timedelta_sum += timedelta
        timedelta_count = timedelta_count + 1

      row.append( get_timedelta(timedelta) )

    last_time = row[2]

  if timedelta_count > 0: avg_timedelta_arr.append( get_timedelta(timedelta_sum/timedelta_count) )

  return avg_timedelta_arr # Avarage time delta for each rank

def file_creator_csv(program_mode, file_out_name, arr):

  print("Dump data to output file")

  file_out = open(file_out_name, "w")

  # Single process with loss and time diff
  if program_mode == 'single':
    # File header
    if timedelta_format == "m":
      file_out.write("#Rank;Iteration;Time;Loss;TimeDiff(min)\n")
    elif timedelta_format == "h":
      file_out.write("#Rank;Iteration;Time;Loss;TimeDiff(hour)\n")
    else:
      file_out.write("#Rank;Iteration;Time;Loss;TimeDiff(sec)\n")

    for idx, row in enumerate(arr):
      if idx == 0: continue
      str_out = "%s;%s;%s;%s;%s" % (row[0], row[1], row[2], row[3], row[4])
      file_out.write(str_out + "\n")

  elif program_mode == 'loss': # All procesess with loss

    changed_rank_idxs = get_changed_rank_idxs(arr)

    file_out.write("#Iteration")
    for idx in changed_rank_idxs:
      file_out.write(";R-" + arr[idx][0])
    file_out.write("\n")

    # Fill row
    for idx in range(0,changed_rank_idxs[1]):
      if idx == 0: continue # Skip first iterations
      file_out.write(arr[idx][1])

      for val in changed_rank_idxs:
        file_out.write(";" + arr[val + idx][3])

      file_out.write("\n")

  elif program_mode == 'timediff': # All procesess with time diff

    changed_rank_idxs = get_changed_rank_idxs(arr)

    file_out.write("#Iteration")
    for idx in changed_rank_idxs:
      file_out.write(";R-" + arr[idx][0])
    file_out.write("\n")

    # Fill row
    for idx in range(0,changed_rank_idxs[1]):
      if idx == 0: continue # Skip first iterations
      file_out.write(arr[idx][1])

      for val in changed_rank_idxs:
        file_out.write(";" + str(arr[val + idx][4]))

      file_out.write("\n")

  file_out.close()

######################################################################################

program_mode, file_in_name, process_ids, not_smaller_then_iters, timedelta_format = get_args(sys.argv)

arr = []

print("Start reading data")

start_time = None
last_time = None

# Fill data array
with open(file_in_name, "r") as file_in:
  for line in file_in:

    #[0] I0719 07:09:31.850666 18289 solver.cpp:239] Iteration 6680, loss = 6.40777
    #I0722 01:18:31.942049 21188 solver.cpp:241] [3] Iteration 25720, loss = 11.0501
    #I0722 01:18:31.942049 21188 solver.cpp:241] Iteration 25720, loss = 11.0501
    line = prepare_line(line)

    vals = line.split()

    if len(vals) < 8:
      continue

    last_time, rank, iter, loss = get_params(line, vals)

    if start_time == None:
      start_time = last_time

    if rank < 0 or iter < 0 or loss < 0:
      continue

    if process_ids != None and process_ids != "*" and len(process_ids) > 0 and rank not in process_ids:
      continue

    if (int(iter) % not_smaller_then_iters) != 0:
      continue

    #Rank Iteration Time Loss
    arr_row = [ rank, iter, last_time, loss ]
    arr.append(arr_row)

print("Sort data")

# Sort: Rank Time
arr = sorted(arr, key=operator.itemgetter(0, 2))

# Single process with loss and time diff
if program_mode == 'single':

  file_name, file_ext = os.path.splitext(file_in_name)
  file_out_name = file_name + "_single_" + arr[0][0] + ".csv"

  print("Compute time diff")
  avg_timedelta_arr = compute_timediff(arr)

  file_creator_csv(program_mode, file_out_name, arr)

  print("Summary:")
  print("   Start time: %s" % (start_time))
  print("   End time: %s" % (last_time))
  print("   Total time: %s (hour)" % ( format( ((arr[len(arr)-1][2] - start_time).total_seconds() / 3600.0) , '.2f' ) ))
  print("   Average iter group time: %s" % (avg_timedelta_arr))
  print("   Number of records: %s" % (len(arr)))

elif program_mode == 'loss': # All procesess with loss

  file_name, file_ext = os.path.splitext(file_in_name)
  file_out_name = file_name + "_loss.csv"

  file_creator_csv(program_mode, file_out_name, arr)

  print("Summary:")
  print("   Start time: %s" % (start_time))
  print("   End time: %s" % (arr[len(arr)-1][2]))
  print("   Total time: %s (hour)" % ( format( ((arr[len(arr)-1][2] - start_time).total_seconds() / 3600.0) , '.2f' ) ))

elif program_mode == 'timediff': # All procesess with time diff

  file_name, file_ext = os.path.splitext(file_in_name)
  file_out_name = file_name + "_timediff.csv"

  print("Compute time diff")
  avg_timedelta_arr = compute_timediff(arr)

  file_creator_csv(program_mode, file_out_name, arr)

  print("Summary:")
  print("   Start time: %s" % (start_time))
  print("   End time: %s" % (arr[len(arr)-1][2]))
  print("   Total time: %s (hour)" % ( format( ((arr[len(arr)-1][2] - start_time).total_seconds() / 3600.0) , '.2f' ) ))
  print("   Average iter group time for each rank: %s" % (avg_timedelta_arr))

print("Finish")

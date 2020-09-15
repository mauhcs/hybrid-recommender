"""
This file is meant to parse the item_history data to a format that the evaluate function understand. 

A common way to use this file goes as follows:

import pandas as pd
from utils.make_Y import make_Y

user_master  = pd.read_csv("data/user_master.tsv", sep="\t")
item_hist    = pd.read_csv("data/item_history.tsv", sep="\t")
make_Y(user_master.user_id.unique(), ihist_train, "FILENAME_WITHOUT_EXTENSION")

"""
import csv
from collections import defaultdict
from multiprocessing import Pool, Queue, Manager

# A global object for conveniently share variables 
# between processes.
class Var:
  def __init__(self, value, data):
    self.value = value
    self.data  = data

GlobalVar = Var(0, None)

def make_Y_for_user(user_id, Q):
  """
  This is a worker function, each process will build a row with
  user_id, recommended_list.  Something like:
  9999 1,2,3,4,5,6,7,8,3,7,8,... 
  """
  data = GlobalVar.data
  _user_truth = []
  for _, row in data[data.user_id == user_id].iterrows():
      _user_truth.extend([row.item_id.replace("I","")] * row.frequency)
  Ui = len(_user_truth)
  _user_truth = _user_truth + ["0"]*(72-Ui)
  #print("Put", user_id, Len.value)
  Q.put([user_id, _user_truth])
  
def writer(Q, filepath):
  """
  Each worker builds its own row and sends it to the writer to write to a tsv file.
  It is more efficient to have only one thread righting, also it avoids any 
  race condition or deadlocks, where two or more processes could be waiting for each other to 
  open/close the target file.
  """
  
  print("Starting Writer")
  with open(filepath, "wt") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["user_id", "items_actioned_on"])
    
  with open(filepath, "a") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    i = 0
    while 1:
      m = Q.get()
      if m == "kill":
        print("\nStoping Writer")
        break
      if len(m) == 2:
        user_id     = m[0]
        #print("GEt",user_id)
        _user_truth = m[1]
        tsv_writer.writerow([user_id, f",".join(_user_truth) ])
        if i % 1000 == 0:
          print(f"{i:,} / {GlobalVar.value:,} ({i/GlobalVar.value * 100:.2f})%", end="\r", flush=True)
        i+=1

def make_Y(user_ids, data, file_name):
  GlobalVar.value = len(user_ids)
  GlobalVar.data = data
  filepath = f"data/{file_name}.tsv"
  Q = Manager().Queue()
  with Pool(processes=32) as pool:
    #put listener to work first
    watcher = pool.apply_async(writer, (Q,filepath))
    #fire off workers
    jobs = []
    for user_id in user_ids[:]:
        job = pool.apply_async(make_Y_for_user, (user_id, Q))
        jobs.append(job)
    
    for job in jobs: 
        job.get()
        
    Q.put('kill')
    watcher.get()

  return 
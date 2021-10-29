import glob, json
import numpy as np
import matplotlib.pyplot as plt

class jsonPlot():
 """
 PLOTTING FROM JSON FILES
 todo: ADD METRICS PLOT (RIGHT NOW ONLY DOING EVAL)
 """
  def __init__(self, folderpath):
    self.folder = folderpath

  def read_json(self,filepath):
    with open(filepath) as json_file:
      data = json.load(json_file)
    acc = data['eval/label_top_1_accuracy']
    loss = data['eval/regularization_loss']
    step = data['global_step']
    json_file.close()
    return (acc,loss,step)
  
  def get_result(self):
    acc = []
    loss = []
    step = []
    best = []
    for filepath in glob.glob(self.folder+'/result_*.json'):
        iacc,iloss,istep = self.read_json(filepath)
        if 'best' in filepath:
          best = [iacc, iloss, istep]
        else:  
          acc.append(iacc)
          loss.append(iloss)
          step.append(istep)
    # sort by step
    acc = np.array(acc)
    loss = np.array(loss)
    step = np.array(step)
    sortidx = step.argsort()
    return (best, acc[sortidx], loss[sortidx], step[sortidx])

    
  def plot_accuracy(self):
    best, acc, loss, step = self.get_result()

    plt.figure(figsize=(10,10))
    plt.plot(step, acc)
    plt.plot(best[-1],best[0] ,'ro')
    plt.xlabel('Iteration step')
    plt.ylabel('Eval Accuracy (top 1)')

 
 
 class simclrCommand():
    '''
    simclrCommand(model_dir, model_folder, params={})
    class to run simclr command,
    input requires a model_dir (where all models will be)
    input requires a model_folder (for this particular model
    either will be created or point to training folder)
    params are already set, but if you want to change or append
    you can put in your own params dictionary.
    Before runing, please note which default params are set.

    Example usage:
    model_dir = '/content/drive/MyDrive/FourthBrain/TeamSemiSuperCV/pretraining/'
    model_folder = 'test'
    params = {"train_batch_size":16, "width_multiplier":2 , "sk_ratio":0.0625}
    mycommand = simclrCommand(model_dir, model_folder,params)
    mycommand.run_command()

    note that you can check the command before running simply by printing:
    print(mycommand.compile_command())
    '''
  def __init__(self, model_dir, model_folder, params={}):
    #default params
    self.params = {'train_mode':"pretrain",
          "mode":"train_then_eval",
          "train_batch_size":64,
          "train_epochs":100 ,
          "learning_rate":1.0,
          'weight_decay':1e-4,
          "temperature":0.5,
          "dataset":"xray",
          "image_size":244,
          "eval_split":"test",
          "resnet_depth":50,
          "use_blur":"False",
          "color_jitter_strength":0.5,
          'model_dir': model_dir+model_folder ,
          "use_tpu":"False"}
    # add in or change params
    if len(params) > 0:
      for k,v in params.items():
        self.params[k]=v

  def compile_command(self):
      # compiles the command as a string
    simclr_command = ['python run.py']
    for k,v in self.params.items():
      simclr_command.append(f'--{k}={v}')
    return (" ").join(simclr_command)

  def run_command(self):
      # runs the command
    !{self.compile_command()}

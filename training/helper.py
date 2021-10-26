
 class simclrCommand():
    # class to run simclr command,
    # input requires a model_dir (where all models will be)
    # input requires a model_folder (will be created)
    # params are already set, but if you want to change or append
    # you can put in your own params dictionary.
    # Before runing, please note which default params are set.
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

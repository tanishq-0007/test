# Requirements and steps 

py -3.10 -m venv tfenv
tfenv\Scripts\activate
python -m pip install --upgrade pip
pip install tensorflow==2.16.1 matplotlib ipython notebook pydot graphviz
python
  import tensorflow as tf
  print(tf.__version__)
  print(tf.config.list_physical_devices('GPU'))
  //  Output :  2.16.1
                [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]     //
  exit()
  python newfile.py

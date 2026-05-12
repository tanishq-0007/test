# Requirements and steps 

py -3.10 -m venv tfenv
<br>
tfenv\Scripts\activate
<br>
python -m pip install --upgrade pip
<br>
pip install tensorflow==2.16.1 matplotlib ipython notebook pydot graphviz
<br>
python
<br>
  <\t>import tensorflow as tf
  <br>
  <\t>print(tf.__version__)
  <br>
  <\t>print(tf.config.list_physical_devices('GPU'))
  <br>
  //  Output :  2.16.1
  <br>
                [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]     //
  <br>
  <\t>exit()
  <br>
  python newfile.py

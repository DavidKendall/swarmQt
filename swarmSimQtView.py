"""
swarmSimQtView.py

PyQt5-based animated display for swarm simulator.
(1) The swarm moves, perimeter agents are shown red, inner agents black.
It's interactive -
(2) The display pane is scrollable and zoomable (scale-factor can be varied).
(3) Animation can be paused/resumed and sped up/slowed down.
(4) Moving the mouse within the display area prints model (ie, in swarm model
coordinate system) position on the console (y-axis points up).
(3) A mouse click anywhere in the display area causes swarm coordinates
and perimeter status the be displayed on console. If the position pointed at
is within cohesion range of 1 or more agent, the data for these are displayed;
otherwise data for ALL agents displayed.
(4) Axes are drawn crossing at model (0,0), with grid lines at 100-pixel intervals.
With designed pane dimensions of of 2000x2000 px and scale factor of 50 this
gives grid lines at intervals of 5 in swarm coords. Zoom is by factors of 2, 5
alternately giving grid interval = 5x10^n or 10^n in logical (swarm) coordinates
depending on zoom factor.
"""
from PyQt5.QtWidgets import QWidget, QApplication, QScrollArea, QPushButton,\
                    QLabel, QHBoxLayout, QVBoxLayout, QInputDialog, QLineEdit
from PyQt5.QtGui import QPainter, QColor, QCursor
from PyQt5.QtCore import Qt, QTimer
import swarmSimModel as mdl
import numpy as np
import sys
import argparse

pallette = [Qt.black, Qt.red, Qt.green, Qt.blue]

## Widget providing animated display - will go inside a scroll pane ##
class Display(QWidget):
   ## Initialise UI, model, timer ##  
  def __init__(self, data, kwargs, scf=100.0, intvl=64):
    super().__init__()
    self.initUI(2000, 2000)
    self.dta = data
    self.scaleFact = scf
    self.scaleMultplr = 2 # alt 2, 5
    self.kwargs = kwargs
    if not 'speed' in kwargs:
      kwargs['speed'] = 0.05
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update) # QWidget.update() fires a paintEvent
    self.timer.setInterval(intvl)
    
  ## Initialise UI with preferred geometry, mouse tracking
  def initUI(self, width, height):    
    self.setGeometry(50, 50, width, height)
    self.setWindowTitle('Swarm Display')
    self.setMouseTracking(True)
    self.setCursor(QCursor(Qt.CrossCursor))
    self.show()

  # Display position pointed at my mouse, in swarm coordinates
  def mouseMoveEvent(self, event):
    ax = (event.x()/self.size().width() - 0.5) * self.scaleFact    
    ay = -(event.y()/self. size().height() - 0.5) * self.scaleFact    
    print("({0:.2f},{1:.2f})  ".format(ax,ay), end="\r")

  # If mouse clicked while animation paused, display data of agent(s) in range,
  # or all agents, if none in range
  def mousePressEvent(self, evt):
    if self.timer.isActive():
      return
    # swarm coords pointed at: 
    ax = (evt.x()/self.size().width() - 0.5) * self.scaleFact    
    ay = -(evt.y()/self. size().height() - 0.5) * self.scaleFact 
    # agents within range:
    inrange = np.hypot(self.dta[mdl.POS_X]-ax, self.dta[mdl.POS_Y]-ay) < self.dta[mdl.CF]
    if np.count_nonzero(inrange) == 0:
      print("All agents:    ")
      print("x:", self.dta[mdl.POS_X].round(2))
      print("y:", self.dta[mdl.POS_Y].round(2))
      print("p:", self.dta[mdl.PRM])
    else:
      print("Agents near ({0:.2f},{1:.2f}):".format(ax,ay))
      print("x:", np.extract(inrange, self.dta[mdl.POS_X]).round(2))
      print("y:", np.extract(inrange, self.dta[mdl.POS_Y]).round(2))
      print("p:", np.extract(inrange, self.dta[mdl.PRM]))

  '''
  Model stepping options can be varied here by changing code in mdl.d_step(...).
  By default, scaling='linear', exp_rate=0.2, with_perimeter=False,
    perimeter_directed=False, stability_factor=0.0
  Note that the perimeter is shown in a distinctive colour if EITHER of  
    with_perimeter or perimeter_directed is TRUE.
  '''
  def paintEvent(self, event):
    width = int(self.size().width());     height = int(self.size().height())
    lh = self.dta.shape[1]
    mdl.d_step(self.dta, **self.kwargs)
    clrs = np.where(self.dta[mdl.PRM],1,0) 
    qp = QPainter()
    qp.begin(self)
    qp.setPen(Qt.cyan)
    hw = width//2;    iw = width//20
    hh = height//2;   ih = height//20
    for i in range(0, hw, iw):
      qp.drawLine(hw+i,0,hw+i,height)
      qp.drawLine(hw-i,0,hw-i,height)
    for i in range(0, hh, ih):
      qp.drawLine(0,hh+i,width,hh+i)
      qp.drawLine(0,hh-i,width,hh-i)
    qp.setPen(Qt.blue)
    qp.drawLine(0,hh,width,hh)
    qp.drawLine(hw,0,hw,height)
    for i in range(lh):
      gx = int((self.dta[mdl.POS_X,i]/self.scaleFact + 0.5)*width)
      gy = int((-self.dta[mdl.POS_Y,i]/self.scaleFact + 0.5)*height)
      qp.setPen(pallette[clrs[i]])
      qp.drawEllipse(gx-2, gy-2, 4, 4)   
    qp.end()    
## 
## end of Display class

## 'Main window': contains timer control buttons in a panel sitting
##   to left of a scroll pane containing a Display instance.  ###########
class Window(QWidget):
  def __init__(self, data, kwargs):
    super().__init__()
    self.kwargs = kwargs
    self.initUI(data, kwargs)

  def initUI(self, data, kwargs):
    self.dsp = Display(data, kwargs)  ## make a Display instance
    self.stpBtn = QPushButton("Step")
    self.rnpBtn = QPushButton("Run")
    self.fstBtn = QPushButton("Faster")
    self.slwBtn = QPushButton("Slower")
    self.lngBtn = QPushButton("Longer step")
    self.shtBtn = QPushButton("Shorter step")
    self.outBtn = QPushButton("Zoom out")
    self.zInBtn = QPushButton("Zoom in")  
    self.dmpBtn = QPushButton("Dump")  
    self.loadBtn = QPushButton("Load")  

    self.tmrLbl = QLabel("{:d}".format(self.dsp.timer.interval()))
    self.tmrLbl.setAlignment(Qt.AlignCenter)
    self.scfLbl = QLabel("{:.1f}".format(self.dsp.scaleFact))
    self.scfLbl.setAlignment(Qt.AlignCenter)
    self.sszLbl = QLabel("{:.3f}".format(self.dsp.kwargs['speed']))
    self.sszLbl.setAlignment(Qt.AlignCenter)

    vbox = QVBoxLayout()           ## these buttons laid out vertically,
    vbox.addStretch(1)             ## centred by means of a stretch at each end
    vbox.addWidget(self.stpBtn)
    vbox.addWidget(self.rnpBtn)
    vbox.addWidget(self.fstBtn)
    vbox.addWidget(self.tmrLbl)
    vbox.addWidget(self.slwBtn)
    vbox.addWidget(self.lngBtn)
    vbox.addWidget(self.sszLbl)
    vbox.addWidget(self.shtBtn)
    vbox.addWidget(self.outBtn)
    vbox.addWidget(self.scfLbl)
    vbox.addWidget(self.zInBtn)
    vbox.addWidget(self.dmpBtn)
    vbox.addWidget(self.loadBtn)
    vbox.addStretch(1)
    
    scrollArea = QScrollArea()        ## and a scroll pane;
    scrollArea.setWidget(self.dsp)    ## put the former in the latter 
    scrollArea.horizontalScrollBar().setValue(500);
    scrollArea.verticalScrollBar().setValue(600);

    hbox = QHBoxLayout()    ## Put button panel to left of the scrollArea
    hbox.addLayout(vbox)
    hbox.addWidget(scrollArea)
    self.setLayout(hbox)

    self.setGeometry(50, 50, 1200, 700) # Initally 1100x700 px with 50 px offsets

    self.rnpBtn.clicked.connect(self.stopStart) # Register handlers: 
    self.fstBtn.clicked.connect(self.faster)    #   - methods in Display instance
    self.slwBtn.clicked.connect(self.slower)
    self.stpBtn.clicked.connect(self.step)
    self.lngBtn.clicked.connect(self.longerStep)
    self.shtBtn.clicked.connect(self.shorterStep)
    self.outBtn.clicked.connect(self.zoomOut)
    self.zInBtn.clicked.connect(self.zoomIn)
    self.dmpBtn.clicked.connect(self.saveState)
    self.loadBtn.clicked.connect(self.loadState)

  ## Timer control methods
  def step(self):
    self.dsp.update()

  def faster(self):
    i = self.dsp.timer.interval()
    if i > 16:
      i //= 2
      self.dsp.timer.setInterval(i)
      self.tmrLbl.setText("{:d}".format(self.dsp.timer.interval()))

  def slower(self):
    i = self.dsp.timer.interval()
    if i < 1024:
      i *= 2
      self.dsp.timer.setInterval(i)
      self.tmrLbl.setText("{:d}".format(self.dsp.timer.interval()))

  def stopStart(self):
    if self.dsp.timer.isActive():
      self.dsp.timer.stop()
      self.rnpBtn.setText("Run")
    else:
      self.dsp.timer.start()
      self.rnpBtn.setText("Pause")

  def longerStep(self):
    s = self.dsp.kwargs['speed'] * 2.0
    self.sszLbl.setText("{:.3f}".format(s))
    self.dsp.kwargs['speed'] = s

  def shorterStep(self):
    s = self.dsp.kwargs['speed'] / 2.0
    self.sszLbl.setText("{:.3f}".format(s))
    self.dsp.kwargs['speed'] = s

  ## Zoom control methods
  def zoomOut(self):
    self.dsp.scaleFact *= self.dsp.scaleMultplr # initially x2
    if self.dsp.scaleMultplr == 2:              # alternately x5, x2
      self.dsp.scaleMultplr = 5
    else:
      self.dsp.scaleMultplr = 2
    self.dsp.update()
    self.scfLbl.setText("{:.1f}".format(self.dsp.scaleFact))

  def zoomIn(self):
    if self.dsp.scaleMultplr == 2:  # alternate multiplier BEFORE using
      self.dsp.scaleMultplr = 5
    else:
      self.dsp.scaleMultplr = 2
    self.dsp.scaleFact /= self.dsp.scaleMultplr
    self.dsp.update()
    self.scfLbl.setText("{:.1f}".format(self.dsp.scaleFact))

  def saveState(self):
    path, ok = QInputDialog.getText(self, "Save","Path:", QLineEdit.Normal, "")
    if ok and path != '':
      mdl.saveState(self.dsp.dta, path)

  def loadState(self):
    path, ok = QInputDialog.getText(self, "Load","Path:", QLineEdit.Normal, "")
    if ok and path != '':
      self.dsp.dta = mdl.loadState(path)
      self.dsp.update() #repaint
##
## End Window class

'''
Run the QtView
:args: a list of command-line arguments created by argparse
'''
def runQtView(args):
  swarm_args = {k:v for k,v in args.items() if k in ['random', 'load_state', 'read_coords', 'cf', 'rf', 'kc', 'kr', 'kd', 'goal', 'loc', 'grid', 'seed'] and v is not None}
  step_args = {k:v for k,v in args.items() if k in ['scaling', 'exp_rate', 'speed', 'perimeter_directed', 'stability_factor', 'perimeter_packing_factor'] and v is not None} 
  if 'random' in swarm_args.keys():
    n = swarm_args['random']
    del swarm_args['random']
    b = mdl.mk_rand_swarm(n, **swarm_args)
  elif 'read_coords' in swarm_args.keys():
    xs, ys = mdl.readCoords(swarm_args['read_coords'])
    del swarm_args['read_coords']
    b = mdl.mk_swarm(xs, ys, **swarm_args)
  elif 'load_state' in swarm_args.keys():
    b = mdl.loadState(swarm_args['load_state'])
  else:
    print("Error in swarm creation")
    return

  app = QApplication([]) # constructor requires no runtime args
  win = Window(b, step_args)
  win.show()
  sys.exit(app.exec_())

############################## Mainline 
parser = argparse.ArgumentParser()
swarm = parser.add_mutually_exclusive_group(required=True)
swarm.add_argument('-r', '--random', type=int, help='create random swarm of size RANDOM')
swarm.add_argument('-s', '--load_state', help='load initial swarm state from LOAD_STATE')
swarm.add_argument('-c', '--read_coords', help='read initial agent positions from READ_COORDS')
parser.add_argument('--cf', type=float, help='radius of the cohesion field')
parser.add_argument('--rf', type=float, help='radius of the repulsion field')
parser.add_argument('--kc', type=float, help='weight of the cohesion vector')
parser.add_argument('--kr', type=float, help='weight of the repulsion vector')
parser.add_argument('--kd', type=float, help='weight of the direction vector')
parser.add_argument('--goal', type=float, help='the swarm has a goal with coordinates (GOAL, GOAL)')
parser.add_argument('--loc', type=float, help='initially centre of the swarm is at coordinates (LOC, LOC)')
parser.add_argument('--grid', type=float, help='initially swarms is distributed in an area of 2.GRID x 2.GRID')
parser.add_argument('--seed', type=int, help='seed for random number generator for random swarm')
parser.add_argument('--scaling', choices=['linear', 'quadratic', 'exponential'], help='scaling method for computation of repulsion vector')
parser.add_argument('--exp_rate', type=float, help='exponential rate if scaling="exponential"')
parser.add_argument('--speed', type=float, help='distance moved per unit time')
parser.add_argument('--perimeter_directed', action='store_true', help='use only perimeter agents in goal seeking')
parser.add_argument('--stability_factor', type=float, help='constrain agent movement if magnitude of resultant vector is less than STABILITY_FACTOR * speed')
parser.add_argument('--perimeter_packing_factor', type=float, help='reduce repulsion field by PERIMETER_PACKING_FACTOR for perimeter agents')
args = vars(parser.parse_args())
runQtView(args)



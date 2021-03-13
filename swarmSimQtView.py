"""
swarmSimQtView.py

PyQt5-based animated display for swarm simulator.
The swarm moves, perimeter agents are shown red, inner agents black.

It's interactive -
  - The display pane is scrollable and zoomable (scale-factor can be varied).
  - Animation can be paused/resumed and sped up/slowed down.
  - A step count is displayed on the GUI.
      This is incremented whenever the model steps, and is reset when a model is loaded.
  - Positioning the mouse within the display area displays position in the model coordinate
      system (y-axis points up).
  - A mouse click in the display area within 5 px of an agent while animation is paused displays
      swarm coords and perimeter status (of all agents within range) in a scrollable information window.
      If no agents are within range, a message to this effect is displayed.
      Also displays COH, REP circles of agents on main display; any previous displayed circles are reteined
      if SHIFT-clicked.
      
  - The information window may be hidden/shown/cleared. If it is visible, it must be hidden or closed before
    the app will exit. The 'Quit' button closes both windows and exits.
    
  - The information window is editable: text may be copied/cut/pasted/deleted/edited.

Axes are drawn crossing at model (0,0), with grid lines at 100-pixel intervals.
With designed pane dimensions of of 2000x2000 px and scale factor of 50 this
gives grid lines at intervals of 5 in swarm coords. Zoom is by factors of 2, 5
alternately giving grid interval = 5x10^n or 10^n in logical (swarm) coordinates
depending on zoom factor.

The animation mechanism is that a timer tick OR a click on the Step button causes the model's
d_step function to be invoked (with whatever kwargs have been chosen), then a repaint is scheduled.

"""
from PyQt5.QtWidgets import QWidget, QApplication, QScrollArea, QPushButton, QMainWindow,\
           QLabel, QHBoxLayout, QVBoxLayout, QFormLayout, QInputDialog, QLineEdit, QPlainTextEdit
from PyQt5.QtGui import QPainter, QColor, QCursor
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtCore
import swarmSimModel as mdl
import numpy as np
import sys
import argparse
import json

pallette = [Qt.black, Qt.red, Qt.darkGreen, Qt.blue, Qt.magenta, Qt.cyan, Qt.yellow]

"""
Widget providing animated display - will go inside a scroll pane
Also contains working code and data for animating the display.
"""
class Display(QWidget):
   ## Initialise UI, model, timer ##  

  def __init__(self, data, kwargs, scf=100.0):
    super().__init__()
    #self.initUI(2000, 2000)
    self.dta = data
    self.scaleFact = scf
    self.scaleMultplr = 2 # alt 2, 5
    self.stepCt = 0
    self.running = False
    self.kwargs = kwargs
    self.xv = None;  self.yv = None; self.mag = None; self.ang = None;
    self.ecf = None; self.erf = None
    if not 'speed' in kwargs:
      kwargs['speed'] = 0.05
 
    # Initialise model on basis of initial data      
    self.xv, self.yv, self.mag, self.ang, self.ecf, self.erf, _, _ = mdl.compute_step(self.dta, **self.kwargs)
    self.agtClrs = np.where(self.dta[mdl.PRM],1,0) 
    self.infoDsp = InfoDisplay()
    
    self.setGeometry(50, 50, 2000, 2000)
    self.setWindowTitle('Swarm Display')
    self.setMouseTracking(True)
    self.setCursor(QCursor(Qt.CrossCursor))
    self.showCircles = []
    self.show()

  # Step the model
  def step(self):
    mdl.apply_step(self.dta) # update positions from prevous step computation and do next  one ...
    self.xv, self.yv, self.mag, self.ang, self.ecf, self.erf, _, _ = mdl.compute_step(self.dta, **self.kwargs)
    self.agtClrs = np.where(self.dta[mdl.PRM],1,0) 
    self.showCircles = []
    self.update()
    self.stepCt += 1

  # Display position pointed at my mouse, in swarm coordinates
  def mouseMoveEvent(self, event):
    ax = (event.x()/self.size().width() - 0.5) * self.scaleFact    
    ay = -(event.y()/self. size().height() - 0.5) * self.scaleFact    
    print("({0:.2f},{1:.2f})  ".format(ax,ay), end="\r")

  #Helper to following function: Assemble information string about an agent;
  #update display colour of neighbouring agents
  def infoMsg(self, agt):
    msg = "\nAgent {:d} at ({:.2f},{:.2f})".format(agt, self.dta[mdl.POS_X,agt], self.dta[mdl.POS_Y,agt])
    if self.dta[mdl.PRM, agt]:
      msg += " (on perim)"
    if self.mag is not None: # once d_step has run, xv, yv, mag are all non-null
      msg += "\n{:d} neighbours".format(int(self.dta[mdl.COH_N,agt]))
      if self.dta[mdl.COH_N,agt] > 0:
        msg += ":"
        for j in range(self.dta.shape[1]):
          if self.mag[agt,j] <= self.ecf[agt,j] and agt != j:
            msg += " {:d}:{:.3f}\u221f{:.1f};".format(j, self.mag[agt,j], self.ang[j,agt]*180/np.pi)
      msg += "\n{:d} repellors".format(int(self.dta[mdl.REP_N,agt]))     
      if self.dta[mdl.REP_N,agt] > 0:
        msg += ":"
        for j in range(self.dta.shape[1]):
          if self.mag[agt,j] <= self.erf[agt,j] and agt != j:
            msg += " {:d}:{:.3f}\u221f{:.1f};".format(j, self.mag[agt,j], self.ang[j,agt]*180/np.pi)
      msg += "\n"
    return msg

  # If mouse clicked while animation paused, display data of agent(s) within 5 px of click; display COH,
  # REP circle(s) of selected agent(s). If SHIFT-click, circles of previously sel'd agents are retained.
  def mousePressEvent(self, evt):
    if self.running:
      return
    # find agents pointed at (within 5 px):
    inrange = np.hypot(((self.dta[mdl.POS_X]/self.scaleFact + 0.5)*self.size().width() - evt.x()), 
                      (-self.dta[mdl.POS_Y]/self.scaleFact + 0.5)*self.size().height() - evt.y()) < 5 

    mods = QApplication.keyboardModifiers()
    if not mods == Qt.ShiftModifier:
      self.showCircles = []
    self.showCircles.extend(np.where(inrange)[0])
    if len(self.showCircles) == 0:
      message = "\n{:d} steps:\nNo agents in range".format(self.stepCt)
    else:
      message = "\n{:d} steps:".format(self.stepCt)
      for i in self.showCircles:
        message += self.infoMsg(i)
      self.update()

    self.update()
    self.infoDsp.tArea.appendPlainText(message)
    self.infoDsp.show()
 
  # Note that the perimeter is shown in a distinctive colour if perimeter_directed is TRUE.
  def paintEvent(self, event):
    width = int(self.size().width());     height = int(self.size().height())
    lh = self.dta.shape[1]
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
      qp.setPen(pallette[self.agtClrs[i]])
      qp.drawEllipse(gx-2, gy-2, 4, 4)
      if i in self.showCircles:      
        rx = int(self.dta[mdl.CF,i]/self.scaleFact*width); ry = int(self.dta[mdl.CF,i]/self.scaleFact*height) 
        qp.setPen(Qt.darkGreen)
        qp.drawEllipse(gx-rx, gy-ry, 2*rx, 2*ry)  
        rx = int(self.dta[mdl.RF,i]/self.scaleFact*width); ry = int(self.dta[mdl.RF,i]/self.scaleFact*height) 
        qp.setPen(Qt.magenta)
        qp.drawEllipse(gx-rx, gy-ry, 2*rx, 2*ry)  
    qp.end()
## 
## end of Display class

"""
Window to display information about agent(s) clicked upon.
Some pretty bizarre code to improvise a resize event:  Qt5 does not seem to provide a natural way of doing this.
"""
class InfoDisplay(QMainWindow):
  resized = QtCore.pyqtSignal()

  def __init__(self, parent=None):
    super(InfoDisplay, self).__init__(parent=parent)
    
    self.setWindowTitle("Information Display")
    self.setGeometry(900, 0, 600, 700)
    self.centralwidget = QWidget(self)
    self.centralwidget.setObjectName("centralwidget")
    self.setCentralWidget(self.centralwidget)
    QtCore.QMetaObject.connectSlotsByName(self)
    self.resized.connect(self.rszHndlr)

    layout = QFormLayout()
    self.tArea = QPlainTextEdit(self)
    #self.tArea.setReadOnly(True)
    self.tArea.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    layout.addWidget(self.tArea)
    
    self.tArea.resize(self.width()-1, self.height()-1)
    
  def resizeEvent(self, event):
    self.resized.emit()
    return super(InfoDisplay, self).resizeEvent(event)

  def rszHndlr(self):
    #print("resize ({:d} x {:d})".format(event.size().width(), event.size().height()))
    self.tArea.resize(self.width()-1, self.height()-1)
## end of InfoDisplay class


"""
'Main window': contains animation timer and control buttons in a panel sitting  to left of
 a scroll pane containing a Display instance.
"""
class Window(QWidget):
  def __init__(self, data, kwargs):
    super().__init__()
    self.initUI(data, kwargs)
   
  def initUI(self, data, kwargs, intvl=64):
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.tick) # QWidget.update() fires a paintEvent
    self.timer.setInterval(intvl)
 
    self.dsp = Display(data, kwargs)  ## make a Display instance
    
    self.stpLbl = QLabel("{:d}".format(self.dsp.stepCt))
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
    self.clrBtn =  QPushButton("Clear Info") 
    self.hideBtn =  QPushButton("Hide Info") 
    self.showBtn =  QPushButton("Show Info") 
    self.quitBtn =  QPushButton("Quit") 

    self.tmrLbl = QLabel("{:d}".format(self.timer.interval()))
    self.tmrLbl.setAlignment(Qt.AlignCenter)
    self.scfLbl = QLabel("{:.1f}".format(self.dsp.scaleFact))
    self.scfLbl.setAlignment(Qt.AlignCenter)
    self.sszLbl = QLabel("{:.3f}".format(self.dsp.kwargs['speed']))
    self.sszLbl.setAlignment(Qt.AlignCenter)

    vbox = QVBoxLayout()           ## these buttons laid out vertically,
    vbox.addStretch(1)             ## centred by means of a stretch at each end
    vbox.addWidget(self.stpLbl)
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
    vbox.addWidget(self.clrBtn)
    vbox.addWidget(self.hideBtn)
    vbox.addWidget(self.showBtn)
    vbox.addWidget(self.quitBtn)
    vbox.addStretch(1)
    
    scrollArea = QScrollArea()        ## a scroll pane for the graphical display;
    scrollArea.setWidget(self.dsp)    ## put the display in the scroll pane 
    scrollArea.horizontalScrollBar().setValue(500);
    scrollArea.verticalScrollBar().setValue(600);

    hbox = QHBoxLayout()    ## Put button panel to left of the scrollArea
    hbox.addLayout(vbox)
    hbox.addWidget(scrollArea)
    self.setLayout(hbox)

    self.setGeometry(50, 50, 1200, 700) # Initally 1200x700 px with 50 px offsets

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
    self.clrBtn.clicked.connect(self.clearInfo)
    self.hideBtn.clicked.connect(self.hideInfo)
    self.showBtn.clicked.connect(self.showInfo)
    self.quitBtn.clicked.connect(self.quit)

  # Handle a timer tick by stepping the model and firing a repaint 
  def tick(self):
    self.dsp.step()
    self.stpLbl.setText("{:d}".format(self.dsp.stepCt))
    
  ## Timer control methods
  def step(self):
    self.dsp.step()
    self.stpLbl.setText("{:d}".format(self.dsp.stepCt))

  def faster(self):
    i = self.timer.interval()
    if i > 16:
      i //= 2
      self.timer.setInterval(i)
      self.tmrLbl.setText("{:d}".format(self.timer.interval()))

  def slower(self):
    i = self.timer.interval()
    if i < 1024:
      i *= 2
      self.timer.setInterval(i)
      self.tmrLbl.setText("{:d}".format(self.timer.interval()))

  def stopStart(self):
    if self.timer.isActive():
      self.timer.stop()
      self.dsp.running = False
      self.rnpBtn.setText("Run")
    else:
      self.timer.start()
      self.dsp.running = True
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
      self.dsp.update()    #repaint
      self.dsp.stepCt = 0  #and reset step count

  def clearInfo(self):
    self.dsp.infoDsp.tArea.setPlainText("");

  def hideInfo(self):
    self.dsp.infoDsp.hide()

  def showInfo(self):
    self.dsp.infoDsp.show()

  def quit(self):
   self.dsp.infoDsp.close()
   self.close()

## End Window class

'''
Run the QtView
:args: a list of command-line arguments created by argparse
'''
def runQtView(args):
  swarm_args = {k:v for k,v in args.items() if k in ['random', 'load_state', 'read_coords', 'cf', 'rf', 'kc', 'kr', 'kd', 'goal', 'loc', 'grid', 'seed'] and v is not None}
  step_args = {k:v for k,v in args.items() if k in ['scaling', 'exp_rate', 'speed', 'perim_coord', 'stability_factor', 'pc', 'pr'] and v is not None} 
  if 'random' in swarm_args.keys():
    n = swarm_args['random']
    goal = json.loads(swarm_args['goal'])
    swarm_args['goal'] = [[goal[0]], [goal[1]]]
    del swarm_args['random']
    b = mdl.mk_rand_swarm(n, **swarm_args)
  elif 'read_coords' in swarm_args.keys():
    b, swarm_args, step_args = mdl.load_swarm()
  elif 'load_state' in swarm_args.keys():
    b = mdl.loadState(swarm_args['load_state'])
  else:
    print("Error in swarm creation")
    return

  mdl.dump_swarm(b, swarm_args, step_args) 
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
parser.add_argument('--cb', type=float, help='radius of the cohesion field')
parser.add_argument('--rb', type=float, help='radius of the repulsion field')
parser.add_argument('--kc', type=float, help='weight of the cohesion vector')
parser.add_argument('--kr', type=float, help='weight of the repulsion vector')
parser.add_argument('--kd', type=float, help='weight of the direction vector')
parser.add_argument('--goal', default='[0.0, 0.0]', help='GOAL should be a string like "[10.0, 15.5]"')
parser.add_argument('--loc', type=float, help='initially centre of the swarm is at coordinates (LOC, LOC)')
parser.add_argument('--grid', type=float, help='initially swarms is distributed in an area of 2.GRID x 2.GRID')
parser.add_argument('--seed', type=int, help='seed for random number generator for random swarm')
parser.add_argument('--scaling', choices=['linear', 'quadratic', 'exponential'], help='scaling method for computation of repulsion vector')
parser.add_argument('--exp_rate', type=float, help='exponential rate if scaling="exponential"')
parser.add_argument('--speed', type=float, help='distance moved per unit time')
parser.add_argument('--perim_coord', action='store_true', help='use only perimeter agents in goal seeking')
parser.add_argument('--stability_factor', type=float, help='constrain agent movement if magnitude of resultant vector is less than STABILITY_FACTOR * speed')
parser.add_argument('--pc', type=float, help='multiply cohesion weight by PC (>= 1.0) for perimeter agents')
parser.add_argument('--pr', type=float, help='multiply repulsion field radius by PR (0 < PR <= 1.0) for perimeter agents')
args = vars(parser.parse_args())
runQtView(args)



import sys
import time
import math
import pickle
import random
import collections
import scipy.spatial
import numpy   as np
import tkinter as tk

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPainter,QPen
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication,QPushButton
from PyQt5.QtWidgets import QWidget, QApplication,QLabel, QPlainTextEdit

root = tk.Tk()
screen_width  = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

open_set   = set()
closed_set = set()

n_node     = 1000
t_coord_in = {}
t_in_coord = {}

MIN = float("-inf")
MAX = float("inf")
maxDepth = 100


# node_colour = [0 for i in range(n_node)] # 0  - not visited, 1 - open list, 2 - closed set, 3- Start node, 4 - Goal node

def generate_points_edges_tree(xl,yl,number=4):
	# Generate coordinates for last level
	nodes=[]
	# last level start at x = 20 to xl-20.
	startx = 20
	endx = xl
	starty= 5 
	endy =  yl
	gap_between_two_node = endx //number # For last level. 
	# print("GAP",gap_between_two_node)
	x=startx
	y=endy
	for i in range(number):
		nodes.append((x,y))
		x=x+gap_between_two_node
	#######################
	levels =  math.ceil(math.log2(number))

	heightlevel = endy // (levels + 5)
	prev_level =[]
	for k in nodes:
		prev_level.append(k) # Remember always, cant directly give reference to list with other variable.
	temp_edge = []
	for i in range(levels):
		temp_node=[]
		j=0
		y -= heightlevel
		while j <len(prev_level):
			#print("j", j,len(prev_level))
			if j< len(prev_level)-1:
				parent = ( (prev_level[j][0]+prev_level[j+1][0] )//2, y )
				temp_edge.append([parent,(prev_level[j][0],prev_level[j][1])])
				temp_edge.append([parent,(prev_level[j+1][0],prev_level[j+1][1])])	
				nodes.append(parent)
				temp_node.append(parent)
			else:
				# print("AA",prev_level[j])
				parent = ( prev_level[j][0], y )
				temp_edge.append([parent,(prev_level[j][0],prev_level[j][1])])
				nodes.append(parent)
				temp_node.append(parent)

			j+=2
		#print(".................")
		del prev_level
		prev_level = temp_node	
	#print("YEs..")
	with open('edge_list.pkl', 'wb') as f:
		pickle.dump(temp_edge, f)	

	return nodes,temp_edge

def generate_points(xl,yl,number=50):
	x_coordinates = np.random.randint(xl, size=number)
	y_coordinates = np.random.randint(yl, size=number)
	t=[]
	for i,j in zip(list(x_coordinates),list(y_coordinates)):
		t.append([i,j+70])

	
	return t

def make_edge_list_tsp(node,n_nodes):
	edg = []
	for i in range(n_nodes):
		for j in range(n_nodes):
			if (i<j):
				edg.append([(node[i][0],node[i][1]),(node[j][0],node[j][1])])

	with open('edge_list.pkl', 'wb') as f:
		pickle.dump(edg, f)	

	return edg 			


def find_neighbors(pindex, triang):
	return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]

# return list of list of tuples Ex.[[(x1,y1),(x2,y2)]]
def make_edge_list(node,n_node,tri,bf):
	temp=[]
	for k in range(n_node):
		pindex = k
		neighbor_indices = find_neighbors(pindex,tri)
		for i in range(len(neighbor_indices)):
			# if i%2!=0:
			if i%bf!=0:
				temp.append([(node[pindex][0],node[pindex][1]),(node[neighbor_indices[i]][0],node[neighbor_indices[i]][1])])	
	with open('edge_list.pkl', 'wb') as f:
		pickle.dump(temp, f)
	return temp

# Make adjacency list 
def make_adj_list(a,d,n):
	t=[[]for i in range(n+1)]
	tt=[]
	for i in a:
		t[d[i[0]]].append(d[i[1]])
		t[d[i[1]]].append(d[i[0]])
	for i in t:
		tt.append(list(set(i)))
	return tt


def make_dict_index_coord(points):
	# t = {}
	t_in_coord={}
	#print("points in in - co")
	#print(points)
	cnt=1
	for i in points:
		t_in_coord[cnt] = (i[0],i[1])
		cnt+=1
	return t_in_coord


# Mapping co-ordinates to node index.
def make_dict_coord_index(points):
	# t={}
	t_coord_in={}
	#print("points in co - in")
	#print(points)
	cnt=1
	for i in points:
		t_coord_in[(i[0],i[1])] = cnt
		cnt+=1
	return t_coord_in
				
# Make dictionary for indexing nodes.
def make_dict_node(points):
	dict_node = {}
	cnt=1
	for i in points:
		dict_node[1] = i
		cnt+=1
	return make_dict_node

def ret_edg():
	with open('edge_list.pkl', 'rb') as f:
			edg = pickle.load(f)
	return edg

# def make_dict_index_coord(points):
# 	t = {}
# 	cnt=1
# 	for i in points:
# 		t[cnt] = (i[0],i[1])
# 		cnt+=1
# 	return t

def movegen(adj,n_ind):
	return adj[n_ind]

def backtrace(parent, start, end):
	path = [end]
	while path[-1] != start:
		path.append(parent[path[-1]])
	path.reverse()
	return path

##################### START ###############################

class Node(object):
	def __init__(self,idx, value=float('-Inf')):
		self.id = idx
		self.value = value
		self.children = [None]*2 #b.f max=2

class Tree(object):
	def __init__(self,adjacency_list,lr):
		self.adjacency_list = adjacency_list
		self.lr = lr
		self.num_nodes = len(adjacency_list)-1
		self.root = Node(idx=self.num_nodes)
		self.current = self.root
		self.globalDict = {}
		self.globalDict[self.num_nodes] = self.current
		if self.num_nodes > 3:
			self.buildTree()
	
	def getParentId(self):
		temp = self.adjacency_list[self.current.id]
		for x in temp:
			if x > self.current.id:
				return x
	
	def getParent(self, idx):
		temp = self.adjacency_list[idx]
		for x in temp:
			if x > idx:
				return x
	
	def getChildrenIds(self):
		temp = self.adjacency_list[self.current.id]
		tlist = []
		for i,x in enumerate(temp):
			if x < self.current.id:
				tlist.append(x)
		
		tlist = sorted(tlist)
		if self.lr==0:
			tlist.reverse()
		return tlist
	
	def getChildren(self, idx):
		temp = self.adjacency_list[idx]
		tlist = []
		for i,x in enumerate(temp):
			if x < idx:
				tlist.append(x)
		
		tlist = sorted(tlist)
		if self.lr==0:
			tlist.reverse()
		return tlist

	def buildTree(self):
		self.currChildren = self.getChildrenIds()
		if len(self.currChildren) > 0:
			for j,childId in enumerate(self.getChildrenIds()):
				self.currChildren = self.getChildrenIds()
				child = Node(idx=childId,value=random.sample(range(-100, 100),1)[0])
				self.currChildren[j] = child
				self.current = self.currChildren[j]
				self.globalDict[self.current.id] = self.current
				self.buildTree()
		else:
			self.current.value = random.sample(range(-100, 100),1)[0]
			self.globalDict[self.current.id] = self.current
			self.current = self.globalDict[self.getParentId()]
			return

def save_ToPickle(name,data):
	print ('{}: {}\n'.format(name,data))
	with open('../data/{}.pkl'.format(name), 'wb') as f:
		pickle.dump(data, f)

def list_ToPickle(toSave_data):
	names = ['nodes','edges','adjacency']
	for data,name in zip(toSave_data,names):
		save_ToPickle(name,data)

def alphaBeta(tree,node_id,depth,alpha,beta,maxPlay,path):
	curr_node = tree.globalDict[node_id]
	if depth == 0 or len(tree.getChildren(node_id))==0:
		return [curr_node.value,path]
	
	if maxPlay:
		v = MIN
		for i in tree.getChildren(node_id):
			path.append(i)
			v     = max(v,alphaBeta(tree,i,depth-1,alpha,beta,False,path)[0])
			alpha = max(alpha, v)
			if beta <= alpha:
				break
		curr_node.value = v
		return [curr_node.value,path]
	
	else:
		v = MAX
		for i in tree.getChildren(node_id):
			path.append(i)
			v = min(v, alphaBeta(tree,i,depth-1,alpha,beta,True,path)[0])
			beta = min(beta, v)
			if beta <= alpha:
				break
		curr_node.value = v
		return [curr_node.value, path]

def Algorithm(adjacency_list,lr):
	tree      = Tree(adjacency_list,lr)
	n_nodes_1 = len(adjacency_list)-1
	path = [n_nodes_1]
	best = [n_nodes_1]
	for i in range(0,n_nodes_1):
		curr_node = tree.globalDict[i+1]
	
	path = alphaBeta(tree,n_nodes_1,maxDepth,MIN,MAX,True,path)[1]
	for i in range(0,n_nodes_1):
		curr_node = tree.globalDict[i+1]
	
	x =	0
	while x < len(path)-1:
		if path[x] != tree.getParent(path[x+1]):
			path = path[:x+1] + [tree.getParent(path[x+1])] + path[x+1:]
			x += 2
		else:
			x += 1
	
	tempNode = tree.root
	while len(tree.getChildren(tempNode.id)) !=  0:
		for x in tree.getChildren(tempNode.id):
			if tree.globalDict[x].value == tree.root.value:
				best.append(tree.globalDict[x].id)
				tempNode = tree.globalDict[x]
				break
	
	idx2val = {}
	for i in range(1,n_nodes_1):
		idx2val[tree.globalDict[i].id] = tree.globalDict[i].value
	
	save_ToPickle('values',idx2val)
	save_ToPickle('path',path)
	save_ToPickle('best',path)
	print ('\n\n')
	
	return [path,best]

def my_main(n_node = 10, bf = 2,gg=1):
	x_dim = screen_width-100
	y_dim = screen_height-150
	edge_list = []
	
	global dict_index_coord
	global dict_coord_index
	global adjacency_list 
	
	if (gg==1):
		leaf_n_node = n_node
		tree_nodes,tree_edge_list = generate_points_edges_tree(x_dim,y_dim,leaf_n_node)
		dict_index_coord = make_dict_index_coord(tree_nodes)
		dict_coord_index = make_dict_coord_index(tree_nodes)
		adjacency_list   = make_adj_list(tree_edge_list,dict_coord_index,len(tree_nodes))
		list_ToPickle([tree_nodes,tree_edge_list,adjacency_list])
		
		return

##################### END ###############################

class Example(QMainWindow):
	
	def __init__(self):
		super().__init__()
		self.nodes = 100
		self.bf = 2
		self.dict_index_coord = {}
		self.open_list = []
		self.closed_list = []	
		self.init_phase = -1
		self.start_x=0
		self.start_y=0
		self.goal_x=0
		self.goal_y=0
		
		self.initUI()
		self.setMinimumSize(QSize(screen_width,screen_height))	
		self.setWindowTitle("Alviz v0.2") 
	
	def initUI(self):			   
		
		self.exitAct = QAction( '&Exit', self)		
		self.exitAct.setShortcut('Ctrl+Q')
		self.exitAct.setStatusTip('Exit application')
		self.exitAct.triggered.connect(qApp.quit)

		self.genAct = QAction( '&Generate Graph', self)
		self.genAct.triggered.connect(self.clickMethod)
		self.genTreeAct = QAction( '&Generate Tree', self)
		self.genTreeAct.triggered.connect(self.clickMethod1)
		self.genTSPAct = QAction( '&Generate TSP', self)
		self.genTSPAct.triggered.connect(self.clickMethod2)
		self.startAct = QAction( '&Start Node', self)
		self.goalAct = QAction( '&Goal Node', self)
		self.genRevertAct = QAction( '&Revert', self)
		self.genRevertAct.triggered.connect(self.clickMethodRevert)
		
		self.nodeLabel = QLabel('Number of leaf_nodes:')
		self.nodeText = QPlainTextEdit('100')
		self.nodeText.setFixedSize(50,28)
		self.bfLabel = QLabel('Branching Factor:')
		self.bfText = QPlainTextEdit('2')
		self.bfText.setFixedSize(50,28)
		self.lrLabel = QLabel('lr:')
		self.lrText = QPlainTextEdit('1')
		self.lrText.setFixedSize(50,28)

		self.resetAct =  QAction( '&Reset Screen', self)
		self.resetAct.triggered.connect(self.reset_screen)


		self.menubar = self.menuBar()
		self.fileMenu = self.menubar.addMenu('File')
		self.fileMenu.addAction(self.exitAct)
		self.toolbar = self.addToolBar('')

		self.toolbar.addWidget(self.nodeLabel)
		self.toolbar.addWidget(self.nodeText)
		self.toolbar.addWidget(self.bfLabel)
		self.toolbar.addWidget(self.bfText)

		self.toolbar.addWidget(self.lrLabel)
		self.toolbar.addWidget(self.lrText)

		self.toolbar.addAction(self.genAct)
		self.toolbar.addAction(self.genTreeAct)
		self.toolbar.addAction(self.genTSPAct)
		self.toolbar.addAction(self.startAct)
		self.toolbar.addAction(self.goalAct)
		self.toolbar.addAction(self.genRevertAct)
		self.toolbar.addAction(self.resetAct)
		
		self.setMouseTracking(True)
		self.startAct.setEnabled(False)
		self.goalAct.setEnabled(False)
		self.path_t=[]
		self.best_t=[]

		# my_main()

	def reset_screen(self):
		self.init_phase = -1
		self.startAct.setEnabled(False)
		self.goalAct.setEnabled(False)
		self.update()
		

	def clickMethod(self):
		self.init_phase = 0
		# print('Clicked Pyqt button.')
		self.nodes = int(self.nodeText.toPlainText())
		self.bf = int(self.bfText.toPlainText())
		global node_colour
		node_colour = [0 for i in range(self.nodes+1)] 
		my_main(self.nodes,self.bf,0)
		self.update()

	def clickMethod1(self):
		self.init_phase = 0
		# print('Clicked Pyqt button. 1')
		self.nodes = int(self.nodeText.toPlainText())
		self.bf = int(self.bfText.toPlainText())
		global node_colour
		node_colour = [0 for i in range(self.nodes+1)] 
		my_main(self.nodes,self.bf,1)
		self.update()
		
	def clickMethod2(self):
		self.init_phase = 6
		# print('Clicked Pyqt button. 2')
		self.nodes = int(self.nodeText.toPlainText())
		self.bf = int(self.bfText.toPlainText())
		global node_colour
		node_colour = [0 for i in range(self.nodes+1)] 
		my_main(self.nodes,self.bf,2)
		self.update()
	def clickMethodRevert(self):
		self.init_phase = 0
		# print('Clicked Pyqt button. Revert')
		self.nodes = int(self.nodeText.toPlainText())
		self.bf = int(self.bfText.toPlainText())
		global node_colour
		node_colour = [0 for i in range(self.nodes+1)] 
		# my_main(self.nodes,self.bf,2)
		self.update()			
		

	# def mouseClickEvent(self,e):
	def mousePressEvent(self, e):
		x=e.x()
		y=e.y()
		text = "x: {0},  y: {1}".format(x, y)
		min_x=99999
		max_x=-99999
		min_y=99999
		max_y=-99999
		self.findClosestCoordinate(min_x,min_y,x,y)
		# self.label.setText(text)
		# print(text)
	
	def findClosestCoordinate(self,min_x,min_y,x,y):
		edg=ret_edg()
		myset = set()
		min_dist=999999999
		for e in edg: 
			# myset.add(e[0])
			# myset.add(e[1])
			dist=(x-e[0][0])**2+(y-e[0][1])**2
			if(dist<min_dist) :
				min_dist=dist
				min_x=e[0][0]
				min_y=e[0][1]
		# print("minimum x :"+str(min_x))
		# print("minimum y :"+str(min_y))
		if(self.init_phase==0):#initial phase for planar graph generation
			self.start_x=min_x
			self.start_y=min_y
			self.init_phase=1
			# self.update()
		elif(self.init_phase==1):#After selecting the start node	
			self.goal_x=min_x
			self.goal_y=min_y
			self.init_phase=5
			# self.update()
		elif(self.init_phase==5):#After selecting the goal node
			# print("start_x , start_y"+str(self.start_x)+","+str(self.start_y))
			start_node = dict_coord_index[(self.start_x,self.start_y)]#1
			# print(start_node)
			l=dict_index_coord[start_node]
			# print(l)

			# print("coord - to - int ")
			# print(dict_coord_index)
			# print("int - to - coord ")
			# print(dict_index_coord)
			# print("###################################")
			goal_node =  dict_coord_index[(self.goal_x,self.goal_y)]
			self.adj_list_1 = adjacency_list
			ret_values = Algorithm(adjacency_list,int(self.lrText.toPlainText()))
			self.path_t,self.best_t = ret_values[0],ret_values[1] 
			# print(node_colour)
			# print("###################################")
			# print (len(node_colour))
			self.init_phase=3
		elif(self.init_phase==3):#show the bfs
			self.init_phase=4			#init_phase=4 is the default end phase of all types of graph
		elif(self.init_phase==6):#initial phase for tsp
			self.init_phase=7
		elif(self.init_phase==7):#final phase for tsp
			self.init_phase=4						



	def paintEvent(self, e):

		qp = QPainter()
		qp.begin(self)
		# self.print_s(qp)
		if (self.init_phase!=-1):
			self.drawPoints(qp)
			if(self.init_phase!=6):
				self.drawLines(qp)
			qp.end()


	def drawPoints(self, qp):

		qp.setPen(Qt.red)
		size = self.size()


		edg=ret_edg()
		myset = set()
		for e in edg: 
			myset.add(e[0])
			myset.add(e[1])

		xx = list(myset)	
		self.dict_index_coord =	make_dict_index_coord(xx)

		# print(self.init_phase)
		# print(self.start_x)
		# print(self.start_y)



		if (self.init_phase == 0):#draw points to create planar graph

			for e in edg :
				center = QPoint(e[0][0],e[0][1])
				qp.setBrush(Qt.yellow)
				qp.drawEllipse(center,5,5)
			   #qp.drawPoint(e[0][0],e[0][1])
			
			self.startAct.setEnabled(True)
			
			self.update()

		# else:
		elif (self.init_phase == 1):#draw start node
			for e in edg :
				center = QPoint(e[0][0],e[0][1])
				qp.setBrush(Qt.yellow)
				qp.drawEllipse(center,5,5)

			center = QPoint(self.start_x,self.start_y)
			qp.setBrush(Qt.green)
			qp.drawEllipse(center,10,10)	
			

			self.startAct.setEnabled(False)
			self.goalAct.setEnabled(True)

			self.update()
		elif (self.init_phase == 5):#draw goal node
			for e in edg :
				center = QPoint(e[0][0],e[0][1])
				qp.setBrush(Qt.yellow)
				qp.drawEllipse(center,5,5)

			center = QPoint(self.start_x,self.start_y)
			qp.setBrush(Qt.green)
			qp.drawEllipse(center,10,10)
				
			center = QPoint(self.goal_x,self.goal_y)
			qp.setBrush(Qt.red)
			qp.drawEllipse(center,10,10)	

			self.startAct.setEnabled(False)
			self.goalAct.setEnabled(False)
			self.update()
			# self.init_phase = 5
		elif (self.init_phase == 3):#draw the path and color different nodes as per the color coding mentioned
			i=1
			# print("node col size "+str(len(node_colour)))
			for i in range(1,len(node_colour)):
				point=dict_index_coord[i]
				e = node_colour[i]
				center = QPoint(point[0],point[1])
				if(e==0) :
					# qp.setBrush(Qt.gray)
					qp.setBrush(Qt.yellow)
					qp.drawEllipse(center,8,8)
				if(e==1) :
					qp.setBrush(Qt.magenta)
					qp.drawEllipse(center,8,8)
				if(e==2) :
					qp.setBrush(Qt.blue)
					qp.drawEllipse(center,8,8)
				if(e==3) :
					qp.setBrush(Qt.cyan)
					qp.drawEllipse(center,15,15)
				if(e==4) :
					qp.setBrush(Qt.red)
					qp.drawEllipse(center,15,15)
				# i=i+1					
				# qp.drawEllipse(center,5,5)


				
				self.update()
	
		elif(self.init_phase == 4):	#default final state of all graph
			self.update()
		elif(self.init_phase == 6): #initial phase of tsp
			for e in edg :
				center = QPoint(e[0][0],e[0][1])
				qp.setBrush(Qt.yellow)
				qp.drawEllipse(center,5,5)
			   #qp.drawPoint(e[0][0],e[0][1])
			
			self.startAct.setEnabled(True)
			
			self.update()
		elif(self.init_phase == 7):	#finall tour plot
			for e in edg :#you have to use the tour returned by TSP in place of edg
				center = QPoint(e[0][0],e[0][1])
				qp.setBrush(Qt.red)
				qp.drawEllipse(center,5,5)
			   #qp.drawPoint(e[0][0],e[0][1])
			
			self.startAct.setEnabled(True)
			
			self.update()


	def drawLines(self, qp):

		pen = QPen(Qt.black, 2, Qt.SolidLine)

		qp.setPen(Qt.gray)

		with open('edge_list.pkl', 'rb') as f:
			edg = pickle.load(f)
		
		   #main()
		for e in edg :
		   qp.drawLine(e[0][0],e[0][1],e[1][0],e[1][1])	

		if(self.init_phase == 3):
			# qp.setPen(Qt.red)
			# qp.setWidth(10)
			pen = QPen(Qt.black, 5, Qt.DashDotLine)
			pen.setBrush(Qt.blue)
			pen.setWidth(5)
			qp.setPen(pen)
			for i in range(len(self.path_t)-1):
				a=self.path_t[i]
				a_pos=dict_index_coord[a]
				b=self.path_t[i+1]
				b_pos=dict_index_coord[b]
				if(b in self.adj_list_1[a] or a in self.adj_list_1[b]):
					qp.drawLine(a_pos[0],a_pos[1],b_pos[0],b_pos[1])
			pen.setBrush(Qt.red)
			pen.setWidth(3)
			qp.setPen(pen)
			for i in range(len(self.best_t)-1):
				a=self.best_t[i]
				a_pos=dict_index_coord[a]
				b=self.best_t[i+1]
				b_pos=dict_index_coord[b]
				if(b in self.adj_list_1[a] or a in self.adj_list_1[b]):
					qp.drawLine(a_pos[0],a_pos[1],b_pos[0],b_pos[1])
		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	ex.show()
	sys.exit(app.exec_())
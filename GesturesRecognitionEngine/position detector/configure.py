import numpy as np
import os

class Config():
	"""
	"""
	# FHN params
	a        = 0.5
	b        = 0.1
	gamma    = 0.1

	freq_ctrl= 0.08
	dt       = 0.01
	sdt      = 0.1
	T        = 100
	transtime= 100

	nSimulations  = 10
	nOrientations = 4
	nPrototype    = 1

	# regime params
	# >> note:
	#   >>  for ai = 0.2, bi = 0.05
	#   >>  ci:- oscillatory = (0.1, 0.5)
	#   >>  ci: excitatory   = (-0.2, 0)
	#

	ai = 0.2
	bi = 0.5
	fr = 0.2
	ci = 0.5
	ci_range = np.array(range(-20,51,1))/100

	# Gaussian params
	N    = 10
	iRad = 4
	eRad = 2
	iA   = 10
	eA   = 20

	# for orientation bar
	mu  = np.array([5., 5.])
	std = np.array([[20., -19.8], [-19.8, 20]])

	savepath = './logs'
	SOM_weights_path = './SOM_weights.npy'
	Orientation_path = './Orientation_bars.npy'

	# if not os.path.exists(savepath):
	# 	os.mkdir(savepath)

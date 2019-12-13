#! /usr/bin/env python

import random

class WorldObject:
	def __init__(self):	
 		raise NotImplementedError('Please use child classes, not this base class.')

	x_low = -6
	x_high = 6
	y_low = -8
	y_high = 8
	z_low = .2
	z_high = 2.5
	unity_scale = 0.1
	use3d = False			# should we allow objects to float in space, or keep them on the ground?

	def scale_values(self, num, min, max):
		return (num * (max - min) + min)

	def unscale_values(self, num, min, max): 
		return (num - min) / (max - min)


class RosObject(WorldObject):
	def __init__(self):
		if random.random() > 0.5:		# dont need to block middle of x AND y, but one or the other
			self.x = random.choice([random.uniform(0, .4), random.uniform(.6, 1)])		# don't put objects in robot space
			self.y = random.random()
		else:
			self.x = random.random()
			self.y = random.choice([random.uniform(0, .4), random.uniform(.6, 1)])
		self.z = random.uniform(0, 1) if self.use3d else self.z_low
		self.unity_x = self.scale_values(self.x, self.x_low, self.x_high) * self.unity_scale
		self.unity_y = self.scale_values(self.y, self.y_low, self.y_high) * self.unity_scale
		self.unity_z = self.scale_values(self.z, self.z_low, self.z_high) * self.unity_scale if self.use3d else self.z_low * self.unity_scale


class UnityObject(WorldObject):
	def __init__(self, X, Y, Z):
		self.unity_x = X
		self.unity_y = Y
		self.unity_z = Z
		self.x = self.unscale_values(X, self.x_low, self.x_high) #/ self.unity_scale
		self.y = self.unscale_values(Y, self.y_low, self.y_high) #/ self.unity_scale
		self.z = self.unscale_values(Z, self.z_low, self.z_high) #/ self.unity_scale



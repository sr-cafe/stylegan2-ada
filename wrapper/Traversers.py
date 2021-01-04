import random

class Linear:
	def build_path(self, waypoints, steps, shuffle=False):
		if shuffle:
			random.shuffle(waypoints)

		self.points = []
		for i in range(len(waypoints) - 1):
			for j in range(steps):
				fraction = j / float(steps)
				self.points.append(waypoints(i + 1) * fraction + waypoints[i] * (1 - fraction))
		return self

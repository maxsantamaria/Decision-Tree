NUM_CLASSES = 3


class Node:
	def __init__(self, x, y, parent=None, threshold=None, feature_index=None):
		self.x = x
		self.y = y
		self.parent = parent
		self.left = None
		self.right = None
		self.threshold = threshold
		self.feature_index = feature_index

	def gini_impurity(self):
		features = self.x.shape[1]
		n = self.x.shape[0]
		count_classes = [0] * NUM_CLASSES
		for elem in y:
			for i in range(NUM_CLASSES):
				if elem == i:
					count_classes[i] += 1
		gini = 1 - sum((c / n) ** 2 for c in count_classes)
		return gini

	def count_classes(self):
		counter = [0] * NUM_CLASSES
		for elem in self.y:
			counter[int(elem)] += 1
		return counter

	def is_leaf(self):
		if self.left is None or self.right is None:
			return True
		else:
			return False

	def predict(self):
		count = self.count_classes()
		max_index = 0
		max_value = 0
		for i in range(len(count)):
			if count[i] > max_value:
				max_index = i
				max_value = count[i]
		#print(max_index, max_value)
		return max_index




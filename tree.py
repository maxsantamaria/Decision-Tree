import numpy as np
from node import NUM_CLASSES, Node



def gini_impurity(x, y):
	features = x.shape[1]
	n = x.shape[0]
	count_classes = [0] * NUM_CLASSES
	for elem in y:
		for i in range(NUM_CLASSES):
			if elem == i:
				count_classes[i] += 1
	gini = 1 - sum((c / n) ** 2 for c in count_classes)
	return gini


def gini(samples, count_classes):
	gini = 1 - sum((c / samples) ** 2 for c in count_classes)
	return gini


def best_split(x, y):
	features = x.shape[1]
	n = x.shape[0]
	if n <= 1:
		return None, None
	count_classes = [0] * NUM_CLASSES
	for elem in y:
		for i in range(NUM_CLASSES):
			if elem == i:
				count_classes[i] += 1
	root_gini = gini(n, count_classes)
	best_gini = 0
	threshold = None
	feature = None
	
	for i in range(features):
		thresholds, classes = zip(*sorted(zip(x[:, i], y)))
		#print(classes)
		count_classes_left = [0] * NUM_CLASSES
		count_classes_right = [elem for elem in count_classes]
		for j in range(n - 1):
			this_class = int(classes[j][0])
			count_classes_left[this_class] += 1  # add one sample to the left node
			count_classes_right[this_class] -= 1  # don't add that sample to the right node
			gini_left = gini(j + 1, count_classes_left)
			gini_right = gini(n - (j + 1), count_classes_right)
			delta_gini = root_gini - ((j + 1) * gini_left + (n - (j + 1)) * gini_right) / n  # CHECK
			#delta_gini = ((j + 1) * gini_left + (n - (j + 1)) * gini_right) / n
			#with open("gini.txt", "a") as data:
			#	data.write(str(i) + " " +  str(thresholds[j]) + " " + str(delta_gini) + "\n")
			
			#print("threshold: ", thresholds[j], "feature: ", i, "gini: ", delta_gini)

			### If data <= threshold, it goes to the left node, else right node
			if thresholds[j] != thresholds[j + 1] and delta_gini > best_gini:
				best_gini = delta_gini
				feature = i
				threshold = thresholds[j + 1]
			
		

	#print(feature, threshold, best_gini)
	return feature, threshold 





def build_tree(node, max_depth, depth=0):
	if depth < max_depth:
		feature, threshold = best_split(node.x, node.y)
		if feature is not None:
			new_x_left = []
			new_y_left = []
			new_x_right = []
			new_y_right = []
			for i in range(node.x.shape[0]):
				if node.x[i, feature] <= threshold:
					new_x_left.append(node.x[i, :])
					new_y_left.append(node.y[i])
				else:
					new_x_right.append(node.x[i, :])
					new_y_right.append(node.y[i])
			new_x_left = np.array(new_x_left)
			new_y_left = np.array(new_y_left)
			new_x_right = np.array(new_x_right)
			new_y_right = np.array(new_y_right)
			#print(new_x_left.shape)
			#print(new_y_left.shape)
			#print(new_x_right.shape)
			#print(new_y_right.shape)
			node.threshold = threshold
			node.feature_index = feature
			if len(new_x_left) > 0 and len(new_x_left) < len(node.x):
				left_node = Node(new_x_left, new_y_left, node)
				node.left = build_tree(left_node, max_depth, depth + 1)
			else:
				node.left = None
			if len(new_x_right) > 0 and len(new_x_right) < len(node.x):
				right_node = Node(new_x_right, new_y_right, node)
				node.right = build_tree(right_node, max_depth, depth + 1)
			else:
				node.right = None

	return node


		
		
	
def DecisionTree(phase, x, y=None, max_depth=None, tree=None):
	if phase == "Training":
		y = parse_y(y)
		root = Node(x, y)
		feature, threshold = best_split(root.x, root.y)
		#feature, threshold = best_split2(root.x, root.y)
		#print(feature, threshold)
		build_tree(root, max_depth)
		#print_tree(root, 0)
		return root

	elif phase == "Validation":
		tree = y
		return predict(tree, x)



def predict(root, x):
	y_predicted = np.zeros(x.shape[0]).reshape(-1, 1)
	for i, sample in enumerate(x):
		node = root
		while not node.is_leaf():
			#print("hola", sample[root.feature_index], root.threshold)
			if sample[node.feature_index] <= node.threshold:
				
				node = node.left
			else:
				node = node.right
		
		prediction = node.predict()
		y_predicted[i] = [prediction]
	#print(y_predicted)
	#print(y_predicted.shape)
	return y_predicted




def reader(file_name):
	with open(file_name, 'r') as file:
		x = None
		y = None
		for line in file:
			row = line.strip().split(',')
			if len(row) > 1:
				row[:-1] = list(map(float, row[:-1]))
				if x is None:
					x = np.array([row[:-1]])
					y = np.array([row[-1]])
				else:
					x = np.append(x, [row[:-1]], axis=0)
					y = np.append(y, [row[-1]], axis=0)
		
		n = x.shape[0]
		y = y.reshape(n, 1)  # make it a matrix nx1
	return x, y


def parse_y(y):
	new_y = np.zeros(y.shape)
	for i in range(y.shape[0]):
		if y[i] == 'Iris-setosa':
			new_y[i] = 0
		elif y[i] == 'Iris-versicolor':
			new_y[i] = 1
		else:
			new_y[i] = 2
	return new_y


def print_tree(root, space):

    # Base case  
	if (root == None) : 
		return

	# Increase distance between levels  
	space += [10][0] 

	# Process right child first  
	print_tree(root.right, space)  

	# Print current node after space  
	# count  
	print()  
	for i in range([10][0], space): 
		print(end = " ")  
	print(root.y.shape, root.count_classes())

	# Process left child  
	print_tree(root.left, space)


def partition_data(x, y):
	x_partitions = []
	y_partitions = []
	for i in range(5):
		new_x = np.zeros((30, x.shape[1]))
		new_y = [0] * 30
		for class_index in range(3):
			new_x[class_index * 10:(class_index + 1) * 10] = x[i * 10 + 50 * class_index:(i+1) * 10 + 50 * class_index]
			new_y[class_index * 10:(class_index + 1) * 10] = y[i * 10 + 50 * class_index:(i+1) * 10 + 50 * class_index]
		new_y = np.array(new_y).reshape(-1, 1)
		x_partitions.append(new_x)
		y_partitions.append(new_y)
	return x_partitions, y_partitions
		

def accuracy(predicted, y):
	correct_classifications = 0
	for i in range(y.shape[0]):
		
		if predicted[i] == y[i]:
			correct_classifications += 1
	return correct_classifications / y.shape[0]




if __name__ == '__main__':

	# Test 1 with all data
	x, y = reader('iris/iris.data')
	#y = parse_y(y)

	tree = DecisionTree('Training', x, y, 4)
	#print_tree(tree, 0)
	#exit()
	prediction = DecisionTree('Validation', x, tree)
	#print(prediction)
	#y = parse_y(y)
	#acc = accuracy(prediction, y)

	# n-fold Validation
	x_partitions, y_partitions = partition_data(x, y)
	accuracies = []
	for i in range(5):
		y = [0] * (x.shape[0] - 30)
		x_testing = None
		y_testing = None
		#x = np.zeros((x.shape[0] - 30, x.shape[1]))
		
		for j in range(5):
			if i != j:
				if x_testing is None:
					x_testing = x_partitions[j]
					y_testing = y_partitions[j]
				else:
					x_testing = np.concatenate((x_testing, x_partitions[j]), axis=0)
					y_testing = np.concatenate((y_testing, y_partitions[j]), axis=0)
			
		x_validation = x_partitions[i]
		y_validation = y_partitions[i]
		
		tree = DecisionTree('Training', x_testing, y_testing, 20)
		prediction = DecisionTree('Validation', x_validation, tree)
		real_y = parse_y(y_validation)
		acc = accuracy(prediction, real_y)
		print(acc)
		accuracies.append(acc)
	print("Average accuracy: ", np.mean(accuracies))



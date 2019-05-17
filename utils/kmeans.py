import numpy as np


def step(boxes, centroids, cluster_labels):
	avg_loss = 0
	cluster_counts = np.zeros((len(centroids)))
	cluster_examples = np.zeros((len(centroids), 2))

	for i in range(len(boxes)):
		dist = []
		for j in range(len(centroids)):
			dist.append(distance(boxes[i], centroids[j]))
		cluster_labels[i] = int(np.argmin(dist))

	for i in range(len(boxes)):
		cluster_counts[int(cluster_labels[i])] = cluster_counts[int(cluster_labels[i])] + 1

		for j in range(len(centroids)):
			if int(cluster_labels[i]) == j:
				cluster_examples[j] = cluster_examples[j] + boxes[i]

	clusters = []
	for i in range(len(cluster_examples)):
		clusters.append(cluster_examples[i]/cluster_counts[i])

	return cluster_labels, clusters, avg_loss

def kmeans(boxes, max_clusters):
	i = 0
	centroids_idxs = []
	while i < max_clusters:
		idx = int(np.random.rand()*len(boxes))
		if not idx in centroids_idxs:
			centroids_idxs.append(idx)
			i = i + 1


	centroids = [boxes[int(i)] for i in centroids_idxs]

	cluster_labels = np.ones(len(boxes))*-1

	prev_cluster_labels = np.zeros((len(boxes)))

	while i < 10:
		cluster_labels, centroids, _ = step(boxes, centroids, cluster_labels)
		i = i+1

	aspect_ratios = [i[0]/i[1] for i in centroids]

	return aspect_ratios, centroids

def iou(box, centroid):
	bh, bw = box
	ch, cw = centroid
	if cw >= bw and ch >= bh:
			similarity = bw*bh/(cw*ch)
	elif cw >= bw and ch <= bh:
		similarity = bw*ch/(bw*bh + (cw-bw)*ch)
	elif cw <= bw and ch >= bh:
		similarity = cw*bh/(bw*bh + cw*(ch-bh))
	else: #means both w,h are bigger than cw and ch respectively
		similarity = (cw*ch)/(bw*bh)

	return similarity

def distance(a, b):
	return 1 - iou(a,b)
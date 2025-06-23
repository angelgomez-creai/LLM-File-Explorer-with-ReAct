import heapq

class TopKTracker:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def add(self, score, data):
        item = (score, data)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heappushpop(self.heap, item)

    def get_top(self):
        return sorted(self.heap, key=lambda x: -x[0])
class DisjointSet:
    def __init__(self, sets_cnt):
        self.level_parent = [i for i in range(2 * sets_cnt - 1)]
        self.highest = sets_cnt

    def parent(self, node):
        if node != self.level_parent[node]:
            self.level_parent[node] = self.parent(self.level_parent[node])
        return self.level_parent[node]

    def join(self, a, b):
        a = self.parent(a)
        b = self.parent(b)

        if a == b:
            raise Exception('joining the same set')

        self.level_parent[a] = self.highest
        self.level_parent[b] = self.highest
        self.highest += 1

    def common(self, a, b):
        return self.parent(a) == self.parent(b)
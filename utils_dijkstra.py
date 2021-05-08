#  MIT License
#
#  Copyright (c) 2020 WGCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from collections import defaultdict
import numpy as np

class Graph:
    def __init__(self, vertices, inf):
        self.V = vertices  # No. of
        self.inf = inf
        self.V_org = vertices
        self.graph = defaultdict(list)  # default dictionary to store graph
        self.weight = np.zeros([vertices, vertices])  # default dictionary to store graph
    def addEdge(self,u,v,w):
        if v not in self.graph[u]:
            self.graph[u].append(v)
            self.weight[u][v]=w
        else:
            self.weight[u][v] = w
    def printPath(self, parent, j):
        Path_len = 1
        if parent[j] == -1 and j < self.V_org : #Base Case : If j is source
            print (j),
            return 0 # when parent[-1] then path length = 0
        l = self.printPath(parent , parent[j])
        Path_len = l + Path_len
        if j < self.V_org :
            print (j),
        return Path_len
    def findShortestPath(self,src, dest):
        visited =[False]*(self.V) # Initialize parent[] and visited[]
        parent =[-1]*(self.V)
        queue=[] # Create a queue for BFS
        queue.append(src) # Mark the source node as visited and enqueue it
        visited[src] = True
        while queue :
            s = queue.pop(0) # Dequeue a vertex from queue
            if s == dest:   # if s = dest then print the path and return
                return self.printPath(parent, s)
            # Get all adjacent vertices of the dequeued vertex s
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                    parent[i] = s

    def findLkNeighboursDir(self,src, k):
        idx = 1
        res = [[] for i in range(k)]
        marked = np.zeros([self.V])
        visited =[False]*(self.V) # Initialize parent[] and visited[]
        distant =[0]*(self.V)
        queue=[] # Create a queue for BFS
        queue.append(src) # Mark the source node as visited and enqueue it
        visited[src] = True
        for i in self.graph[src]:
            if i in self.inf[src] and src in self.inf[i]:
                marked[i] = 2 #mutual
            if i not in self.inf[src]:
                self.addEdge(src, i, 0.9)
        while queue:
            s = queue.pop(0) # Dequeue a vertex from queue
            for i in range(0, k):
                if len(res[i]) == 0:
                    idx = i
                    break
                else:
                    if s in res[i]:
                        idx = i
                        break

            # Get all adjacent vertices of the dequeued vertex s
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for item in self.graph[s]:
                if visited[item] == False:
                    queue.append(item)
                    if s == src:
                        res[idx].append(item)
                        distant[item]=self.weight[s][item]
                    else:
                        if idx == k-1:
                            for i in self.graph[src]:
                                if i not in self.inf[src]:
                                    self.addEdge(src, i, 1)
                            for i in range(0, len(marked)):
                                if distant[i] != 0:
                                    if distant[i]%1 == 0:
                                        if marked[i] != 2:
                                            marked[i] = -1
                                    else:
                                        marked[i] = 1
                            return res, marked
                        else:
                            res[idx+1].append(item)
                            distant[item] = self.weight[s][item] + distant[s]
                    visited[item] = True
        for i in self.graph[src]:
            if i not in self.inf[src]:
                self.addEdge(src, i, 1)
        for i in range(0, len(marked)):
            if distant[i] != 0:
                if distant[i] % 1 == 0:
                    marked[i] = -1
                else:
                    marked[i] = 1
        return res, marked

    def findLkNeighboursUndir(self,src, k):
        idx = 1
        res = [[] for i in range(k)]
        visited =[False]*(self.V) # Initialize parent[] and visited[]
        queue=[] # Create a queue for BFS
        queue.append(src) # Mark the source node as visited and enqueue it
        visited[src] = True
        while queue:
            # Dequeue a vertex from queue
            s = queue.pop(0)
            for i in range(0, k):
                if len(res[i]) == 0:
                    idx = i
                    break
                else:
                    if s in res[i]:
                        idx = i
                        break
            # Get all adjacent vertices of the dequeued vertex s
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for item in self.graph[s]:
                if visited[item] == False:
                    queue.append(item)
                    if s == src:
                        res[idx].append(item)
                    else:
                        a = (idx == k-1)
                        if idx == k-1:
                            return res
                        else:
                            res[idx+1].append(item)
                    visited[item] = True
        return res

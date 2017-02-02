"""
cluster.py
"""

import json
import networkx as nx
from collections import Counter, defaultdict, deque
import sys
import glob
import os
import time
import matplotlib.pyplot as plt
import community
from sklearn.cluster import spectral_clustering
import scipy


def read_data(path):
    return [json.load(open(f)) for f in glob.glob(os.path.join(path, '*.txt'))]


def build_graph(users, ifremove = False, mindegree = 2):
    G = nx.Graph()
    for u in users:
        G.add_edges_from([u['id'], fid] for fid in u['friends'])
        G.add_edges_from([u['id'], fid] for fid in u['followers'])

    if ifremove == True:
        degree_less_two = [node for node, degree in G.degree().items() if degree < mindegree]
        G.remove_nodes_from(degree_less_two)
        degree_less_one = [node for node, degree in G.degree().items() if degree < 1]
        G.remove_nodes_from(degree_less_one)

    return G

def draw_network(graph, users, filename):
    node_labels = {u['id']:u['screen_name'] for u in users}
    fig = plt.figure()
    plt.axis('off')
    G = nx.draw_networkx(graph, pos=nx.spring_layout(graph), node_size = 20, alpha = 0.3, width = 0.05, edge_color = 'k', labels = node_labels, font_size = 10)
    #plt.show()
    plt.savefig(filename, dpi = 600)


def bfs(graph, root, max_depth):
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)

    nodequeue = deque([root])
    visited = set()
    visited.add(root)

    node2distances[root] = 0
    node2num_paths[root] = 1
    depth = 0
    parent = root
    while (len(nodequeue) > 0):
          #node = "".join(nodequeue.popleft())
          node = str(nodequeue.popleft())
          depth = node2distances[node] + 1
          if depth > max_depth:
            continue
          for n in graph.neighbors(node):
              if (n not in visited):
                  nodequeue.append ([n])
                  node2distances[n] = depth
                  node2parents[n].append(node)
                  node2num_paths[n] = 1
                  visited.add(n)
              elif (node2distances[n] == depth):
                  node2num_paths[n] += 1
                  node2parents[n].append(node)
              
    return dict(sorted(node2distances.items())), dict(sorted(node2num_paths.items())), dict(sorted((_node, sorted(_parents)) for _node, _parents in node2parents.items()))


def bottom_up(root, node2distances, node2num_paths, node2parents):
    result = defaultdict(float)
    node2children = defaultdict(list)
   
    for n, parents in node2parents.items():
        for np in parents:
            node2children[np].append(n)

    def getallnodescore(p, scores):
        score = 0
        for n in node2children[p]:
            if (len(node2children[n]) == 0):  #leaf node
                scores[n] = 1
            elif (scores[n] == 0.0):
                scores[n] = 1 + getallnodescore(n, scores)
            edge = tuple(sorted([p, n]))
            edgescore = scores[n] * node2num_paths[p] / node2num_paths[n]
            result[edge] = edgescore
            score += edgescore
        return score

    scores = defaultdict(float)
    for n in node2distances.keys():
        scores[n] = 0.0
    getallnodescore(root, scores)

    return dict(sorted(result.items()))


def approximate_betweenness(graph, max_depth):
    edge2betweenness = defaultdict(float)
    for n in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, n, max_depth)
        #print(n)
        #print(node2distances)
        edge2score = bottom_up(n, node2distances, node2num_paths, node2parents)
        #print(edge2score)
        for k,v in edge2score.items():
            edge2betweenness[k] += v
    for k,v in edge2betweenness.items():
        edge2betweenness[k] /= 2

    return dict(sorted(edge2betweenness.items()))


def partition_girvan_newman(graph, max_depth):
    tmpgraph = graph.copy()
    edge2betweenness = approximate_betweenness(tmpgraph, max_depth)
    edge2betweenness = sorted(edge2betweenness.items(), key=lambda x : (-x[1], x[0]))

    edgeiter = iter(edge2betweenness)
    while nx.is_connected(tmpgraph):
        [k, v] = next(edgeiter)
        tmpgraph.remove_edge(*k)

    return list(nx.connected_component_subgraphs(tmpgraph))


def get_subgraph(graph, min_degree):
    node2degree = graph.degree()
    tmpgraph = graph.copy()

    for e in filter(lambda x:node2degree[x] < min_degree, node2degree):
        tmpgraph.remove_node(e)
    return tmpgraph


def volume(nodes, graph):
    inset = set(nodes)
    involume = 0
    outvolume = 0
    for n in nodes:
        for nn in graph.neighbors(n):
            if nn in inset:
                involume += 1
            else:
                outvolume += 1

    return int(involume/2 + outvolume)        



def cut(S, T, graph):
    num = 0
    for s, t in [(s,t) for s in S for t in T]:
        if graph.has_edge(s, t):
            num += 1

    return num


def norm_cut(S, T, graph):
    ct = cut(S, T, graph)
    vs = volume(S, graph)
    vt = volume(T, graph)

    return ct/vs + ct/vt



def score_max_depths(graph, max_depths):
    depth2cut = []
    for d in max_depths:
        [c1, c2] = partition_girvan_newman(graph, d)
        nc = norm_cut(c1, c2, graph)
        depth2cut.append((d, nc))

    return sorted(depth2cut, key = lambda x: x[0])


def community_detection(G, users, savegraph = False):
    color = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w'}
    node_labels = {u['id']:u['screen_name'] for u in users}
    partition = community.best_partition(G)
    #drawing
    size = float(len(set(partition.values())))
    print("Totally %d communities are discovered" %(size))
    pos = nx.spring_layout(G)

    if savegraph:
        count = 0.
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
            if size > len(color):
                nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))
            else:
                nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, node_color = color[count-1], labels = node_labels)

        nx.draw_networkx_edges(G,pos, alpha = 0.3, width = 0.05, edge_color = 'k', labels = node_labels, font_size = 10)
        #plt.show()
        plt.savefig('communities.png', dpi = 600)

    return partition


def community_stat(partition):
    print("Totally %d communities" %len(set(partition.values())))
    print("Average number of users per community: %d" %(len(partition) / len(set(partition.values()))))
    com2num = Counter(partition.values())
    for key in com2num:
        print("community %d has %d nodes" %(key, com2num[key]))
    json.dump(partition, open("./partition.txt", 'w'))


# def print_degree_hist(G):
#     degree_sequence = sorted([d for n,d in G.degree(G.nodes()).items()], reverse=True) # degree sequence
#     degreeCount = Counter(degree_sequence)
#     #print(degreeCount)
#     deg, cnt = zip(*degreeCount.items())
#     fig, ax = plt.subplots()
#     plt.bar(deg, cnt, width=0.80, color='b')
#     plt.title("Degree Histogram")
#     plt.ylabel("Count")
#     plt.xlabel("Degree")
#     ax.set_xticks([d+0.4 for d in deg])
#     ax.set_xticklabels(deg)
#     plt.savefig("degree_histogram.png")


def main():
    old_data_path = './old_users'
    new_data_path = './new_users'
    choice = input("\nRead original data or newly downloaded data? \n 1: original \t\t 2: new")
    #print(choice)
    while (choice != '1' and choice != '2'):
        choice = input("Only accept '1' or '2', try again. \n 1: original \t\t 2: new")

    users = []
    print("Loading data...")
    if choice == '1':
        users = read_data(old_data_path)
    else:
        users = read_data(new_data_path)

    print("Building graph with degree >= 2...")
    graph = build_graph(users, ifremove = True)
    print("Graph contains %d nodes, %d edges" %(graph.number_of_nodes(), graph.number_of_edges()))
    choice = input("\n Saving the network graph into file? \n y: yes \t\t n: no")
    while (choice != 'y' and choice != 'Y' and choice != 'n' and choice != 'N'):
        choice = input("Only accept 'y' or 'Y and 'n' or 'N', try again.\n y: yes \t\t n: no")
    
    if choice == 'y' or choice == 'Y':    
        print("Saving original network graph")
        draw_network(graph, users, 'network.png')

    # print_degree_hist(graph)
    degree_sequence = sorted([d for n,d in graph.degree(graph.nodes()).items()], reverse=True) # degree sequence
    degreeCount = Counter(degree_sequence)
    # degreeCount = sorted(degreeCount, key = lambda x : x[0])
    print("Degree distribution:")
    for i in dict(degreeCount.most_common()):
        print("degree: %d, num: %d" %(i, degreeCount[i]))

    if nx.is_connected(graph):
        print("Graph is connected")

    json.dump(dict(degreeCount.most_common()), open("./degree.txt", 'w'))

    # print('norm_cut scores by max_depth:')
    # print(score_max_depths(graph, range(1,5)))

    # clusters = partition_girvan_newman(graph, 3)
    # print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
    #       (clusters[0].order(), clusters[1].order()))
    # print('cluster 2 nodes:')
    # print(clusters[1].nodes())


    print("Now implement community discovering with louvain method...")
    if choice == 'y' or choice == 'Y':
        partition = community_detection(graph, users, savegraph = True)
    else:
        partition = community_detection(graph, users, savegraph = False)

    print("Show the statistics of discovered communities...")
    community_stat(partition)


if __name__ == '__main__':
    main()

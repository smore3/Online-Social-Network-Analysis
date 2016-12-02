import pickle
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict, deque, Counter

def bfs(graph, root, max_depth):
    queue = deque()
    node2distances, node2num_paths, node2parents = dict(), Counter(), dict()
    queue.append((root, 0))
    node2distances[root] = 0
    node2num_paths[root] += 1

    while queue:
        node, level = queue.popleft()

        if level >= max_depth:
            break
        for neighbor in sorted(graph.neighbors(node)):
            if node == root:
                node2distances[neighbor] = 1 + node2distances[node]
                if neighbor in node2parents:
                    node2parents[neighbor].append(node)
                else:
                    node2parents[neighbor] = [node]
                queue.append((neighbor, level + 1))
                node2num_paths[neighbor] += 1

            elif neighbor not in node2parents[node] and (
                    neighbor not in node2distances or node2distances[neighbor] > node2distances[node]):

                node2distances[neighbor] = 1 + node2distances[node]
                if neighbor in node2parents:
                    node2parents[neighbor].append(node)
                else:
                    node2parents[neighbor] = [node]
                queue.append((neighbor, level + 1))
                node2num_paths[neighbor] += 1

    return node2distances, node2num_paths, node2parents

def bottom_up(root, node2distances, node2num_paths, node2parents):
    parents = set()
    node_credit = defaultdict(float)
    node_credit[root] = 1.0
    queue = deque()
    result = defaultdict(float)

    for key, value in node2parents.items():
        node_credit[key] = 1.0
        for j in value:
            parents.add(j)

    for node in node_credit.keys():
        if node not in parents:
            queue.append(node)

    while queue:
        popped = queue.popleft()
        if popped != root:
            p = node2parents[popped]
            if len(p) == 1:
                node_credit[p[0]] = node_credit[p[0]] + node_credit[popped]
                if popped < p[0]:
                    result[(popped, p[0])] = node_credit[popped]
                else:
                    result[(p[0], popped)] = node_credit[popped]

                if p[0] not in queue:
                    queue.append(p[0])
            else:
                total = 0
                for i in p:
                    total += node2num_paths[i]
                for i in p:
                    node_credit[i] = node_credit[i] + (node2num_paths[i] * node_credit[popped] / total)
                    if popped < i:
                        result[(popped, i)] = (node2num_paths[i] * node_credit[popped] / total)
                    else:
                        result[(i, popped)] = (node2num_paths[i] * node_credit[popped] / total)
                    if i not in queue:
                        queue.append(i)
    return result

def approximate_betweenness(graph, max_depth):
    result = dict()
    for n in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, n, max_depth)
        r = bottom_up(n, node2distances, node2num_paths, node2parents)
        for key, value in sorted(r.items()):
            if key not in result:
                result[key] = value
            else:
                result[key] = result[key] + value

    for key, value in result.items():
        result[key] = value / 2.0

    return result

def partition_girvan_newman(graph, max_depth):
    copied_graph = graph.copy()
    #betweenness = sorted(sorted(approximate_betweenness(graph, max_depth).items(), key=lambda x: x[0]), key = lambda x:x[1], reverse= True)
    betweenness=sorted(sorted(nx.edge_betweenness_centrality(graph).items(), key=lambda x: x[0]), key = lambda x:x[1], reverse= True)
    graphs = list()
    i = 0
    while len(graphs) <= 1:
        copied_graph.remove_edge(betweenness[i][0][0], betweenness[i][0][1])
        graphs = list(nx.connected_component_subgraphs(copied_graph))
        i += 1
    return graphs

def girvan_newman(graph):


    clusters = partition_girvan_newman(graph, 1)

    orders=[clusters[i].order() for i in range(len(clusters))]

    return len(clusters)

def cluster():
    tweets = pickle.load(open("tweets.pkl", "rb"))
    graph = nx.Graph()
    cluster_file = open('cluster_output.txt', 'w')
    for tweet in tweets:
        if '@' in tweet['text']:
            mentions = re.findall(r'[@]\S+', tweet['text'])
            for mention in mentions:
                graph.add_node(tweet['user']['screen_name'])
                graph.add_node(mention[1:])
                graph.add_edge(tweet['user']['screen_name'], mention[1:])

    # drawing graph
    remove=[node for node,degree in graph.degree().items() if degree < 2]
    graph.remove_nodes_from(remove)
    nx.draw_networkx(graph, pos=None, with_labels=False, node_color='b', node_size=10, alpha=0.5, )
    plt.savefig('Un-clustered.png')
    num_clusters=girvan_newman(graph)

    cluster_file.write(str(num_clusters))
    cluster_file.write('\n'+str(graph.order()))
def main():
    cluster()

if __name__ == '__main__':
    main()
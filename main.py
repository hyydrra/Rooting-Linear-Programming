import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pulp
from math import sin, cos, sqrt, atan2, radians
import utm
from tabulate import tabulate
from projection import *
import folium

data = pd.read_csv("/home/amirhosein/PycharmProjects/OR_rooting/Data.csv")


def distance(p1, p2):
    return ((p1[1] - p2[1]) ** 2 + (p1[0] - p2[0]) ** 2) ** 0.5


def create_network(opt_path=[]):
    num_points = data.shape[0]
    lat = list(data['latitude'])
    lon = list(data['longitude'])
    x_utm = []
    y_utm = []
    for i in range(num_points):
        temp = utm.from_latlon(lat[i], lon[i])
        x_utm.append(temp[0])
        y_utm.append(temp[1])
    G = nx.MultiDiGraph()
    for i in range(num_points):
        curr_index = int(data.iloc[i]["Place index"])
        G.add_node(str(curr_index), pos=(y_utm[i], x_utm[i]))

    for i in range(num_points):
        curr_index = int(data.iloc[i]["Place index"])
        temp_neighbours = data.iloc[i]["Neighbors indice"].split(",")
        temp_neighbours_w = data.iloc[i]["Neighbors weight"].split(",")

        for j in range(len(temp_neighbours)):
            if (str(curr_index), temp_neighbours[j]) in opt_path:
                G.add_edge(str(curr_index), temp_neighbours[j], color='r', width=5, length=float(temp_neighbours_w[j]))
            else:
                G.add_edge(str(curr_index), temp_neighbours[j], color='b', width=2, length=float(temp_neighbours_w[j]))
    return G


def draw_map(G):
    # plt.figure(figsize=(18, 18))
    pos = {city: (long, lat) for (city, (lat, long)) in nx.get_node_attributes(G, 'pos').items()}
    edge_labels = dict([((u, v,), d['length'])
                        for u, v, d in G.edges(data=True)])
    colors = [d['color']
              for u, v, d in G.edges(data=True)]
    width = [d['width']
             for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, with_labels=True, edge_color=colors, width=width)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
    plt.show()

    return edge_labels


def solver(G, edge_labels, source, target):
    nodes = G.nodes
    # define Cij
    prob = pulp.LpProblem("Shortest_Path_Problem", pulp.LpMinimize)
    cij = {}
    for node in edge_labels:
        i = node[0]
        j = node[1]
        x = pulp.LpVariable("x_(%s_%s)" % (i, j), cat=pulp.LpBinary)
        cij[i, j] = x
    prob += pulp.lpSum([float(edge_labels[x]) * cij[x] for x in edge_labels]), "Total_Hop_Count"
    # define constraints
    for node in nodes:
        if node == source:
            prob += pulp.lpSum([cij[x] for x in edge_labels if x[1] == node]) - \
                    pulp.lpSum([cij[x] for x in edge_labels if x[0] == node]) == 1
        elif node == target:
            prob += pulp.lpSum([cij[x] for x in edge_labels if x[1] == node]) - \
                    pulp.lpSum([cij[x] for x in edge_labels if x[0] == node]) == -1
        else:
            prob += pulp.lpSum([cij[x] for x in edge_labels if x[1] == node]) - \
                    pulp.lpSum([cij[x] for x in edge_labels if x[0] == node]) == 0

    # solve
    prob.solve()

    # prepare the answer
    links = []
    #print(pulp.LpStatus[prob.status])
    #print(pulp.value(prob.objective))
    for x in edge_labels:
        if cij[x].value() == 1:
            links.append(x)
    # print the answer in right order
    result = []
    tmp = source
    while tmp != target:
        for link in links:
            if link[0] == tmp:
                result.append((tmp, link[1]))
                tmp = link[1]
                links.remove(link)
                break
            if link[1] == tmp:
                result.append((tmp, link[0]))
                tmp = link[0]
                links.remove(link)
                break

    return result


def ax_by_c(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    m = (y2 - y1) / (x2 - x1)

    b = 1
    a = -m
    c = m * x1 - y1
    return a, b, c


def findfoot(p1, p2, p3):
    x1 = p3[0]
    y1 = p3[1]
    a, b, c = ax_by_c(p1, p2)
    temp = (-1 * (a * x1 + b * y1 + c) //
            (a * a + b * b))
    x = temp * a + x1
    y = temp * b + y1
    return (x, y)


def distance_from_lat_lon(latlon1, latlon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(latlon1[0])
    lon1 = radians(latlon1[1])
    lat2 = radians(latlon2[0])
    lon2 = radians(latlon2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    # answer in Km
    return distance


def find_k_nearest_node(ps, k):
    distances = []
    num_points = data.shape[0]
    for i in range(num_points):
        nodei = (data.iloc[i]["latitude"], data.iloc[i]["longitude"])
        tempdis = distance_from_lat_lon(nodei, ps)
        distances.append(tempdis)
    distances = np.asanyarray(distances)
    ind = np.argsort(distances)[:k]
    # print(distances)
    return ind


def find_projection(ps, k):
    knearstNode = find_k_nearest_node(ps, k)
    all_projections_distances = {}
    all_projections = {}
    for node in knearstNode:
        curr_index = int(data.iloc[node]["Place index"])
        temp_neighbours = data.iloc[node]["Neighbors indice"].split(",")
        for neigh in temp_neighbours:
            p1 = utm.from_latlon(data.iloc[node]["latitude"], data.iloc[node]["longitude"])
            p2 = utm.from_latlon(data.iloc[int(neigh) - 1]["latitude"], data.iloc[int(neigh) - 1]["longitude"])
            p3 = utm.from_latlon(ps[0], ps[1])
            cost, nearest = pnt2line(p3[0:2], p1[0:2], p2[0:2])

            all_projections_distances[(node + 1, int(neigh))] = cost
            all_projections[(node + 1, int(neigh))] = nearest

    # print(all_projections_distances)
    minp = min(all_projections_distances, key=all_projections_distances.get)
    # print(minp)
    # print(all_projections[minp])
    return minp, all_projections[minp]


def google_map_visual(G, path=[]):
    edge_labels = dict([((u, v,), d['length']) for u, v, d in G.edges(data=True)])
    pos = {city: (long, lat) for (city, (lat, long)) in nx.get_node_attributes(G, 'pos').items()}

    center_lat = data.mean(axis=0)["latitude"]
    center_lon = data.mean(axis=0)["longitude"]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    tooltip = 'Click me!'
    for node in G.nodes:
        lat_lan = utm.to_latlon(pos[node][0], pos[node][1], 39, 's')
        if node.isdigit():
            folium.Marker([lat_lan[0], lat_lan[1]],
                          popup='<i>' + str(data.iloc[int(node) - 1]["Place index"]) + "-" + data.iloc[int(node) - 1][
                              "Place Name"] + '</i>',
                          icon=folium.Icon(color='green'),
                          tooltip=tooltip).add_to(m)
        else:
            folium.Marker([lat_lan[0], lat_lan[1]], popup='<i>' + node + '</i>',
                          icon=folium.Icon(color='red'),
                          tooltip=tooltip).add_to(m)
    m.add_child(folium.LatLngPopup())
    # m.add_child(folium.ClickForMarker(popup='Waypoint'))

    for edge in edge_labels:
        node1 = edge[0]
        node2 = edge[1]
        lat_lan1 = utm.to_latlon(pos[node1][0], pos[node1][1], 39, 's')
        lat_lan2 = utm.to_latlon(pos[node2][0], pos[node2][1], 39, 's')
        if edge in path or (edge[1], edge[0]) in path:
            folium.PolyLine([list(lat_lan1), list(lat_lan2)], color='red', weight=6).add_to(m)
        else:
            folium.PolyLine([list(lat_lan1), list(lat_lan2)], color='blue', weight=2).add_to(m)

    m.save('map_district5.html')


run = 1
while run:
    print(" ")
    print("Choose one option from below menu:")

    while True:
        print("-enter 1 to select Origin and destination by point's ID")
        print("-enter 2 to select Origin and destination by latitude and longitude")
        print("-enter 0 to end the program")
        method = input("Enter one of the options: ")
        if method not in ["1", "2", "0"]:
            print("please try again.")
            continue
        else:
            print("")
            # we're happy with the value given.
            # we're ready to exit the loop.
            break
    if method == "1":
        print("select Origin and destination from below points:")
        print(tabulate(data.drop(['Neighbors indice', 'Neighbors weight'], axis=1), headers='keys', tablefmt='psql',
                       showindex=False))
        G = create_network()
        edge_labels = draw_map(G)
        print("you can view the points on the real map in map.html file")
        google_map_visual(G)
        while True:
            source = int(input("Enter Origin node:\n"))
            if source not in list(range(1, 26)):
                print("please enter a number between 1 and 25.")
                continue
            else:
                break
        while True:
            target = int(input("Enter destination node:\n"))
            if target not in list(range(1, 26)):
                print("please enter a number between 1 and 25.")
                continue
            else:
                break

        path = solver(G, edge_labels, str(source), str(target))
        cost = 0
        for x in path:
            G.remove_edge(x[0], x[1])
            G.add_edge(x[0], x[1], color='r', width=5,
                       length=edge_labels[x[0], x[1]])
            cost = cost + edge_labels[x[0], x[1]]

        print("The shortest path with Cost = " + str(cost) + ", is: ")
        print(path)
        google_map_visual(G, path)
        draw_map(G)

    if method == "2":
        G = create_network()
        print("you can see the latitude and longitude of your preferred points by clicking on the map.html ")
        google_map_visual(G)

        while True:
            source = input("please enter latitude and longitude of source:(lat lon)").split()
            if len(source) != 2:
                print("you should enter 2 number.")
                continue
            else:
                break
        while True:
            destination = input("please enter latitude and longitude of destination:(lat lon)").split()
            if len(source) != 2:
                print("you should enter 2 number.")
                continue
            else:
                break

        #destination = (35.7322 51.2871)
        #destination = (35.7393 ,51.3417)
        #destination = (35.8006, 51.3043)
        #source = (35.7451 51.3283)

        source = [float(i) for i in source]
        destination = [float(i) for i in destination]
        utm_source = utm.from_latlon(source[0], source[1])
        utm_destination = utm.from_latlon(destination[0], destination[1])

        edge_src, p_source = find_projection(source, 5)
        edge_des, p_destination = find_projection(destination, 5)
        lat_lon_p_source = utm.to_latlon(p_source[0], p_source[1], 39, 's')
        lat_lon_p_destination = utm.to_latlon(p_destination[0], p_destination[1], 39, 's')

        lat_lon_edge_src = [(data.iloc[edge_src[0] - 1]["latitude"], data.iloc[edge_src[0] - 1]["longitude"]),
                            (data.iloc[edge_src[1] - 1]["latitude"], data.iloc[edge_src[1] - 1]["longitude"])]
        lat_lon_edge_des = [(data.iloc[edge_des[0] - 1]["latitude"], data.iloc[edge_des[0] - 1]["longitude"]),
                            (data.iloc[edge_des[1] - 1]["latitude"], data.iloc[edge_des[1] - 1]["longitude"])]

        if distance_from_lat_lon(lat_lon_edge_src[0], lat_lon_p_source) < 10 ** -3:
            G.add_node("start", pos=(utm_source[1], utm_source[0]))
            G.add_edge("start", str(edge_src[0]), color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_source, source))
            src_node = "start"

        elif distance_from_lat_lon(lat_lon_edge_src[1], lat_lon_p_source) < 10 ** -3:
            G.add_node("start", pos=(utm_source[1], utm_source[0]))
            G.add_edge("start", str(edge_src[1]), color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_source, source))
            src_node = "start"

        else:
            G.add_node("start", pos=(utm_source[1], utm_source[0]))
            G.add_node("pstart", pos=(p_source[1], p_source[0]))
            G.add_edge("start", "pstart", color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_source, source))
            src_node = "start"

            if (str(edge_src[0]), str(edge_src[1])) in G.edges:
                #G.remove_edge(str(edge_src[0]), str(edge_src[1]))
                G.add_edge("pstart", str(edge_src[1]), color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_source, lat_lon_edge_src[1]))
                G.add_edge(str(edge_src[0]), "pstart", color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_source, lat_lon_edge_src[0]))

            if (str(edge_src[1]), str(edge_src[0])) in G.edges:
                #G.remove_edge(str(edge_src[1]), str(edge_src[0]))
                G.add_edge("pstart", str(edge_src[0]), color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_source, lat_lon_edge_src[0]))
                G.add_edge(str(edge_src[1]), "pstart", color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_source, lat_lon_edge_src[1]))


        if distance_from_lat_lon(lat_lon_edge_des[0], lat_lon_p_destination) < 10 ** -3:
            G.add_node("destination", pos=(utm_destination[1], utm_destination[0]))
            G.add_edge(str(edge_des[0]), "destination", color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_destination, destination))

        elif distance_from_lat_lon(lat_lon_edge_des[1], lat_lon_p_destination) < 10 ** -3:
            G.add_node("destination", pos=(utm_destination[1], utm_destination[0]))
            G.add_edge(str(edge_des[1]), "destination", color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_destination, destination))

        else:
            G.add_node("destination", pos=(utm_destination[1], utm_destination[0]))
            G.add_node("pdestination", pos=(p_destination[1], p_destination[0]))
            G.add_edge("pdestination", "destination", color='b', width=2,
                       length=distance_from_lat_lon(lat_lon_p_destination, destination))
            des_node = "destination"
            if (str(edge_des[0]), str(edge_des[1])) in G.edges:
                #G.remove_edge(str(edge_des[0]), str(edge_des[1]))

                G.add_edge("pdestination", str(edge_des[1]), color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_destination, lat_lon_edge_des[1]))

                G.add_edge(str(edge_des[0]), "pdestination", color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_destination, lat_lon_edge_des[0]))

            if (str(edge_des[1]), str(edge_des[0])) in G.edges:
                #G.remove_edge(str(edge_des[1]), str(edge_des[0]))

                G.add_edge("pdestination", str(edge_des[0]), color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_destination, lat_lon_edge_des[0]))

                G.add_edge(str(edge_des[1]), "pdestination", color='b', width=2,
                           length=distance_from_lat_lon(lat_lon_p_destination, lat_lon_edge_des[1]))



        edge_labels = dict([((u, v,), d['length']) for u, v, d in G.edges(data=True)])
        path = solver(G, edge_labels, "destination", "start")
        Path = []
        cost = 0
        for x in path:
            Path.append((x[1], x[0]))
            G.remove_edge(x[1], x[0])
            G.add_edge(x[1], x[0], color='r', width=5,
                       length=edge_labels[x[1], x[0]])
            cost = cost + edge_labels[x[1], x[0]]

        Path.reverse()
        print("The shortest path with Cost = " + str(cost) + ", is: ")
        print(Path)
        google_map_visual(G, path)
        draw_map(G)

    if method == "0":
        run = 0
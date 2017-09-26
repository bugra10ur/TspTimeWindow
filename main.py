import numpy as np
from tsp_solver.greedy import solve_tsp as tsp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from math import radians, cos, sin, asin, sqrt



def make_dist_matrix(x, y):
    """Creates distance matrix for the given coordinate vectors"""
    N = len(x)
    xx = np.vstack( (x,)*N )
    yy = np.vstack( (y,)*N )
    return np.sqrt( (xx - xx.T)**2 + (yy - yy.T)**2 )


def read_data(filename):
    cities = []

    path = 'C:/Users/Bugra/Desktop/ntnu-som-master/ntnu-som-master/assets/{}.txt'.format(filename)
    with open(path, 'r') as f:
        for line in f:
            city = list(map(float, line.split()[0:]))
            cities.append((city[0], city[1],city[2],city[3],city[4],city[5]))

    return cities

def vectorized_haversine(lat,lng):
    lat = np.deg2rad(lat)
    lng = np.deg2rad(lng)
    dflat = lat[:,None] - lat
    dflng = lng[:,None] - lng
    d = np.sin(dflat/2)**2 + np.cos(lat[:,None])*np.cos(lat) * np.sin(dflng/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


def km_to_second(matrix):
    l=np.reshape(matrix, matrix.shape[0]*matrix.shape[1])
    l=[x*3 if x <= 3 else x for x in l]
    return np.reshape(l, (matrix.shape[0],matrix.shape[1]))


def path_duration(time_matrix,path,delivery_time):
    duration=0
    for i in range(len(path)-1):
        duration=duration+delivery_time+time_matrix[path[i],path[i+1]]
    return duration


cities = read_data('Coords')
id    = np.array(cities)[:, 0]
x     = np.array(cities)[:, 1]
y     = np.array(cities)[:, 2]
start = np.array(cities)[:, 3]
end   = np.array(cities)[:, 4]
#group = np.array(cities)[:, 5]

newcoloumn=[]
temp=[]

for row in cities:
   search = str(str(row[3]) + '-' + str(row[4]))
   index =temp.index(search) if search in temp else -1
   if index == -1:
       newcoloumn.append(len(temp))
       temp.append(search)
   else:
       newcoloumn.append(index)

group=np.array(newcoloumn)


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
matris = np.column_stack((id,x,y,start,end,group))

for i in range(len(temp)):
    indices = group == i
    data = matris[indices,:]
    distances = vectorized_haversine(data[:,1], data[:,2])
    path = tsp(distances)
    time=km_to_second(distances)
    print(path)
    print("Toplam süre:"+"  "+str(path_duration(time,path,10))+" dk")

    plt.plot(data[path, 1], data[path, 2], list(colors.keys())[i])
    for k, txt in enumerate(data[path, 0]):
        plt.annotate(int(txt), (data[path[k], 1], data[path[k], 2]))


plt.scatter(matris[:, 1], matris[:, 2], c=matris[:,5], cmap=plt.cm.Set1,edgecolor='k')
plt.show()


#birleştirme
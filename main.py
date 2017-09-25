import numpy as np
from tsp_solver.greedy import solve_tsp as tsp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from geopy.distance import vincenty


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


cities = read_data('Coords')
id    = np.array(cities)[:, 0]
x     = np.array(cities)[:, 1]
y     = np.array(cities)[:, 2]
start = np.array(cities)[:,3]
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
    distances = make_dist_matrix(data[:,1], data[:,2])
    path = tsp(distances)
    print(path)
    plt.plot(data[path, 1], data[path, 2], list(colors.keys())[i])
    for k, txt in enumerate(data[path, 0]):
        plt.annotate(int(txt), (data[path[k], 1], data[path[k], 2]))


plt.scatter(matris[:, 1], matris[:, 2], c=matris[:,5], cmap=plt.cm.Set1,edgecolor='k')
plt.show()

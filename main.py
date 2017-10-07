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


def which_outer_points(tw1,tw2,matris):

    tw11 = np.array(([matris[int(tw1[0]),1] ,matris[int(tw1[0]) ,2]]))
    tw12 = np.array(([matris[int(tw1[-1]),1],matris[int(tw1[-1]),2]]))
    tw21 = np.array(([matris[int(tw2[0]),1] ,matris[int(tw2[0]) ,2]]))
    tw22 = np.array(([matris[int(tw2[-1]),1],matris[int(tw2[-1]),2]]))
    euclidean=np.zeros(4, dtype = object)
    euclidean[0] = [np.linalg.norm(tw11 - tw21),int(tw1[0]),int(tw2[0])]
    euclidean[1] = [np.linalg.norm(tw11 - tw22),int(tw1[0]),int(tw2[-1])]
    euclidean[2] = [np.linalg.norm(tw12 - tw21),int(tw1[-1]),int(tw2[0])]
    euclidean[3] = [np.linalg.norm(tw12 - tw22),int(tw1[-1]),int(tw2[-1])]
    return min(euclidean)


#Okunan dosyanın arraylara paylaşılması
cities = read_data('Coords')
id    = np.array(cities)[:, 0]
x     = np.array(cities)[:, 1]
y     = np.array(cities)[:, 2]
start = np.array(cities)[:, 3]
end   = np.array(cities)[:, 4]

newcoloumn=[]
temp=[]

#datayı okunan zaman gruplarına göre gruplama işlemi. (Başlangıç- Bitiş zamanlarına göre)
for row in cities:
   search = str(str(row[3]) + '-' + str(row[4]))
   index =temp.index(search) if search in temp else -1
   if index == -1:
       newcoloumn.append(len(temp))
       temp.append(search)
   else:
       newcoloumn.append(index)

group=np.array(newcoloumn)
#çizimde her grubu farklı renkle ifade edebilmek için renk kodları tanımlıyoruz.
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#okunup ayrı ayrı arraylara atanan bilgileri tek bir matris de birleştiriyoruz.
matris = np.column_stack((id,x,y,start,end,group))
#her zaman penceresinin toplam süresini tutar
twd_size=np.max(group)+1
#birleştirme aşanasında kullanmak üzere her bir grubun toplam tamamlanma süresini,
# tamamlanma hedefini (Saat 8-10 arası 2 saat toplam 120dk gibi)
# çizilen rotasını [1 2 34 22] 1 nolu id den başla 22 nolu id de bit gibi bilgileri tutmak için arraylar oluşturma kısmı
time_window_durations  =np.zeros(twd_size, dtype = object)
time_window_goal=np.zeros(twd_size, dtype = object)
time_window_routes=np.zeros(twd_size, dtype = object)

#her grup için ayrı ayrı yol çizdirme kısmı, yukarda oluşturduğumuz arrayleride dolduruyoruz. print sonuçlarından takip edilebilir.
for i in range(len(temp)):
    indices = group == i
    data = matris[indices,:]
    distances = vectorized_haversine(data[:,1], data[:,2])
    path = tsp(distances)
    time=km_to_second(distances)
    print("Group:"+" "+str(i)+"  Time Window: "+str(int(data[1,3]))+"-"+str(int(data[1,4]))+"-"+str((data[1,4]-data[1,3])*60)+"  dk")
    print("Yol sırası->"+str(path))
    print("Id lere göre sırası->"+str(data[path,0]))
    time_window_durations[i]=path_duration(time,path,10)
    time_window_goal[i]=(data[1,4]-data[1,3])*60
    time_window_routes[i]=data[path,0]
    print("Total duration:"+"  "+str(path_duration(time,path,10))+" dk")

    plt.plot(data[path, 1], data[path, 2], list(colors.keys())[i])
    for k, txt in enumerate(data[path, 0]):
        plt.annotate(int(txt), (data[path[k], 1], data[path[k], 2]))

#Oluşturulan yolların çizdirilmesi
plt.scatter(matris[:, 1], matris[:, 2], c=matris[:,5], cmap=plt.cm.Set1,edgecolor='k')
plt.show()


#birleştirme öncesi zaman periyotlarını sıralayıp gruplama
print(time_window_durations)
print(time_window_goal)
print(time_window_routes)
tw_groups = np.column_stack((start,end,group))
tw_order=np.lexsort((tw_groups[:,1],tw_groups[:,0]))
tw_groups=np.unique(tw_groups[tw_order], axis=0)
print(tw_groups)

#ilk 2 grubun hangi noktalarından birleştirilebileceğini sorgulama kısmı. bu bilgileri birleştirme aşamasında kullanacağız.
print("Grup 1-> "+str(time_window_routes[0]))
print("Grup 2-> "+str(time_window_routes[1]))
a=which_outer_points(time_window_routes[0],time_window_routes[1],matris)
print("Grupların birleştirileceği noktalar -> "+str(a))

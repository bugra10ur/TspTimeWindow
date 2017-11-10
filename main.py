import numpy as np
from tsp_solver.greedy import solve_tsp as tsp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from random import randint
from math import radians, cos, sin, asin, sqrt
import haversine as hv
import itertools as it



def make_dist_matrix(x, y):
    """Creates distance matrix for the given coordinate vectors"""
    N = len(x)
    xx = np.vstack( (x,)*N )
    yy = np.vstack( (y,)*N )
    return np.sqrt( (xx - xx.T)**2 + (yy - yy.T)**2 )


def read_data(filePath, delimiter=' '):
    return np.genfromtxt(filePath, delimiter=delimiter)


def vectorized_haversine(lat,lng):
    lat = np.deg2rad(lat)
    lng = np.deg2rad(lng)
    dflat = lat[:,None] - lat
    dflng = lng[:,None] - lng
    d = np.sin(dflat/2)**2 + np.cos(lat[:,None])*np.cos(lat) * np.sin(dflng/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


def km_to_second(matrix):
    if np.isscalar(matrix) == True:
        if float(matrix) <= 3:
           return float(matrix) * 3
        else:
           return float(matrix)
    else:
        l=np.reshape(matrix, matrix.shape[0]*matrix.shape[1])
        l=[x*3 if x <= 3 else x for x in l]
        return np.reshape(l, (matrix.shape[0],matrix.shape[1]))


def path_duration(time_matrix,path,ids,matris):
    waitingTimes=getElementByID(matris, ids, 0, 5)
    duration = sum(waitingTimes)
    for i in range(len(path)-1):
        duration=duration+time_matrix[path[i],path[i+1]]
    return duration


def which_outer_points(tw1,tw2,matris):

    tw11 = np.array(([matris[int(tw1[0]),1] ,matris[int(tw1[0]) ,2]]))
    tw12 = np.array(([matris[int(tw1[-1]),1],matris[int(tw1[-1]),2]]))
    tw21 = np.array(([matris[int(tw2[0]),1] ,matris[int(tw2[0]) ,2]]))
    tw22 = np.array(([matris[int(tw2[-1]),1],matris[int(tw2[-1]),2]]))
    distances=np.zeros(4, dtype = object)
    distances[0] = [hv.haversine(tw11, tw21),int(tw1[0]),int(tw2[0])]
    distances[1] = [hv.haversine(tw11, tw22),int(tw1[0]),int(tw2[-1])]
    distances[2] = [hv.haversine(tw12, tw21),int(tw1[-1]),int(tw2[0])]
    distances[3] = [hv.haversine(tw12, tw22),int(tw1[-1]),int(tw2[-1])]
    print("En yakın uç noktalar->"+str(min(distances)))
    return min(distances)


def merge_groups(tw1,tw2,durations,matris):
    print("Grup 1-> " + str(tw1))
    print("Grup 2-> " + str(tw2))
    mergePoints = which_outer_points(tw1, tw2, matris)
    #mergePoints'in ilk elemanı noktalar arasındaki km cinsinden uzaklığı dönüyordu, onu dk cinsine çeviriyoruz
    mergePointsDuration=km_to_second(mergePoints[0])
    #birleştirilecek noktalara göre rotayı sıralama kısmı Birleştirilecek nokta
    # 1. grubun ilk elemanı ise 1.grup ters çevrilir,
    # Birleştirilecek noktalardan 2. gruba ait olan nokta   2. grubun son elemanı ise 2. grup ters çevrilir
   # if mergePoints[1] == tw1[0]:
      #  tw1=tw1[::-1]
  #  if mergePoints[2] != tw2[0]:
     #   tw2=tw2[::-1]
    merged_routes = list(it.chain(tw1, tw2))
    merged_duration=durations[0]+durations[1]+mergePointsDuration
    print("Grupların birleştirilmesi ->"+str(merged_routes))
    return merged_routes,merged_duration


def combine_merge_groups(twr,twd,tw_groups,matris):
    mergedGroups=[twr[0],0]
    durations=twd[0:2]
    print(range(len(tw_groups)-1))
    for i in range(len(tw_groups)-1):
        indices = i
        mergedGroups = merge_groups(mergedGroups[0],twr[i+1],durations,matris)
        durations=[twd[i+1],mergedGroups[1]]
    return mergedGroups


def plot_route(mergedIds,matris):
    results = np.array(list(map(int, mergedIds[0])))
    x = []
    y = []
    ids = []
    for i in range(len(matris)):
        id = np.where(matris[:, 0] == results[i])
        ids.append(results[i])
        x.append(matris[id, 1][0][0])
        y.append(matris[id, 2][0][0])

    g = plt.figure(randint(100, 1000))
    plt.plot(x, y)
    #id  leri  grafik  üzerinde görüntüleme  kısmı
    for t, txt in enumerate(ids):
        plt.annotate(int(txt), (x[t], y[t]))
    plt.show()


def find_index(ids,matrix):
    results = np.array(list(map(int, ids)))
    idIndex=[]
    for i in range(len(results)):
        id = np.where(matrix[:, 0] == results[i])[0][0]
        idIndex.append(id)
    return idIndex


def getElementByID(matrix,searched,x,y):
    # parametre olarak aldığı matriste x indisi değerlerinin sahip olduğu y değerlerini döndürür,
    # *** searched olarak verilen değerlerin ve matrisin tekil olduğu varsayılmıştır
    results = []
    for i in range(len(searched)):
        indice = np.where(matrix[:, x] == searched[i])[0]
        temp = matrix[indice, y]
        results.append(temp)
    return results


def time_window_elemination(time_window_durations,time_window_goal,time_window_routes,matris):
    print("len->"+str(len(time_window_durations)))
    time_window_removed =[]
    for i in range(len(time_window_durations)):
        print("twg->"+str(i)+"-"+str(time_window_durations[i]))
        #küçük zaman penceresi içerisindeki noktalar yeterli değil ise eleme yapılır
        goal=time_window_goal[i]
        duration= time_window_durations[i]
        route=time_window_routes[i]
        while (goal < duration):
            if len(list(route)) == 1:
                i=i+1
            else:
                newRoute= list(route)
                removedId=newRoute[-1]
                time_window_removed.append(removedId)
                del newRoute[-1]
                data = matris[find_index(newRoute,matris), :]
                distances = vectorized_haversine(data[:, 1], data[:, 2])
                path = tsp(distances)
                time = km_to_second(distances)
                duration=path_duration(time,path,data[path,0],matris)
                route=data[path, 0]
                print("Yeni Yol sırası->" + str(path))
                print("Id lere göre Yeni yol sırası->" + str(data[path, 0]))
                print("Çıkartılan noktalar->"+str(time_window_removed))
    return time_window_removed


def estimate_Mean_2D(dataset,x,y):
    #Parametre olarak; Data ve x,y bilgilerini içeren kolon indexlerini alır,
    #Çıktı olarak datadaki tüm noktaların ortalama x,y bilgilerini döner...
    data=dataset[:, [x, y]]
    mu = np.mean(data, axis=0)
    return mu


def compute_distances_no_loops(dataset,x,y,idIndex):
    #Parametre olarak sıralacak data,datanın x,y ve id kolon index bilgilerini alır..
    #çıktı olarak id,x,y şeklinde girdi olarak aldığı datayı ortalama noktasına en yakından en uzak olacak şekilde sıralanmış halini döner...
    mu=estimate_Mean_2D(dataset,x,y)
    X=dataset[:, [x, y]]
    ids=dataset[:,idIndex]
    difference =mu-X
    differenceRoot=np.square(difference)
    sumDifs=differenceRoot.sum(axis=1, keepdims=True)
    #dists=((ids,sumDifs[:,0]))
    #dists=np.concatenate([ids,sumDifs])
    print(ids)
    dists=np.column_stack((ids,sumDifs))
    sorted=dists[np.argsort(dists[:,1])]
    return sorted
#Dosyayı oku  id,x,y,start,end,waitingTime
cities = read_data('Coords.txt')
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
matris = cities#np.column_stack((id,x,y,start,end,waiting,group))
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
    totalDuration=time_window_durations[i]=path_duration(time,path,data[path,0],matris)
    time_window_goal[i]=(data[1,4]-data[1,3])*60
    time_window_routes[i]=data[path,0]
    print("Total duration:"+"  "+str(totalDuration)+" dk")

    f = plt.figure(1)
    plt.plot(data[path, 1], data[path, 2], list(colors.keys())[i])
    for k, txt in enumerate(data[path, 0]):
        plt.annotate(int(txt), (data[path[k], 1], data[path[k], 2]))

#Oluşturulan yolların çizdirilmesi
plt.scatter(matris[:, 1], matris[:, 2], c=matris[:,5], cmap=plt.cm.Set1,edgecolor='k')
f.show()

#birleştirme öncesi zaman periyotlarını sıralayıp gruplama

tw_groups = np.column_stack((matris[:,3],matris[:,4],group))
tw_order=np.lexsort((tw_groups[:,1],tw_groups[:,0]))
tw_groups=np.unique(tw_groups[tw_order], axis=0)

print(time_window_durations)
print(time_window_goal)
print(time_window_routes)
print(tw_groups)

a=time_window_elemination(time_window_durations,time_window_goal,time_window_routes,matris)
print(a)
plt.show()
mergedGroups=combine_merge_groups(time_window_routes,time_window_durations,tw_groups,matris)
plot_route(mergedGroups,matris)
#Zaman grubu sayısı kadar dör (bizim örnek için 8)  ilklendirme olarak ilk zaman grubunu atıyoruz
#mergedGroups=[time_window_routes[0],0]
#durations=time_window_durations[0:2]
#print(range(len(tw_groups)-1))
#for i in range(len(tw_groups)-1):
    #indices = i
    #mergedGroups = merge_groups(mergedGroups[0],time_window_routes[i+1],durations,matris)
    #durations=[time_window_durations[i+1],mergedGroups[1]]





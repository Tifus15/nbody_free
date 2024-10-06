import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from nbody import simulate
import os
from tqdm import tqdm
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def simulation_start(times=25, bodies = 7,sim=True):
    for i in tqdm(range(times)):
        
        flag = False
        while not flag:
            print("######{}#######".format(i))
            name1,H1,x1 = simulate(N=bodies, t= 0, tEnd= 1.27,dt = 0.01, r=0.5, softening= 0.001, G = 1, plotRealTime=sim)
            extra_x = -0.5+np.random.rand(1,3)
            #print(extra_x.shape)
            #print(x1.shape)
            x_n = np.concatenate((x1[0,:,:],extra_x),axis=0)
            #print(x_n.shape)
            name2,H2,x2 = simulate(N=bodies+1, t= 0, tEnd= 1.27,dt = 0.01, r=1.0, softening= 0.001, G = 0.001,x =x_n , plotRealTime=sim)
            CRIT1 = torch.std(torch.tensor(H1))
            CRIT2 = torch.std(torch.tensor(H2)) 
            #print(x_n.shape) 
            #print(x1.shape)
            #print(x2.shape)
            H1 = H1.reshape((128,1))
            print(H1.shape)
            H2 = H2.reshape((128,1))
            print(H2.shape)
            
            
            q_x1 = x1[:,:,0]
            q_y1 = x1[:,:,1]
            q_z1 = x1[:,:,2]
            max_x = np.max(q_x1.flatten())
            min_x = np.min(q_x1.flatten())
            max_y = np.max(q_y1.flatten())
            min_y = np.min(q_y1.flatten())
            max_z = np.max(q_z1.flatten())
            min_z = np.min(q_z1.flatten())
            if CRIT1 > 0.1 or max_x > 1.5 or min_x < -1.5 or max_y > 1.5 or min_y < -1.5 or max_z > 1.5 or min_z < -1.5:
                print("{} > 0.1  FAILED".format(CRIT1))
                print("{} max x > 1".format(max_x))
                print("{} min x < -1".format(min_x))
                print("{} max y > 1".format(max_y))
                print("{} min y < -1".format(min_y))
                print("{} max z > 1".format(max_z))
                print("{} min z < -1".format(min_z))
              
                os.remove(name1)
                os.remove(name2)
                time.sleep(0.5)
                flag = False
                continue
            else:
                print("###############")
                print("{} < 0.1 PASSED".format(CRIT1))
                print("{} max x < 1".format(max_x))
                print("{} min x > -1".format(min_x))
                print("{} max y < 1".format(max_y))
                print("{} min y > -1".format(min_y))
                print("{} max z < 1".format(max_z))
                print("{} min z > -1".format(min_z))
                print("###############")
        
        q_x2 = x2[:,:,0]
        q_y2 = x2[:,:,1]
        q_z2 = x2[:,:,2]
        max_x = np.max(q_x2.flatten())
        min_x = np.min(q_x2.flatten())
        max_y = np.max(q_y2.flatten())
        min_y = np.min(q_y2.flatten())
        max_z = np.max(q_z2.flatten())
        min_z = np.min(q_z2.flatten())
        if CRIT2 > 0.1 or max_x > 1.5 or min_x < -1.5 or max_y > 1.5 or min_y < -1.5 or max_z > 1.5 or min_z < -1.5:
            print("{} > 0.1  FAILED".format(CRIT2))
            print("{} max x > 1".format(max_x))
            print("{} min x < -1".format(min_x))
            print("{} max y > 1".format(max_y))
            print("{} min y < -1".format(min_y))
            print("{} max z > 1".format(max_z))
            print("{} min z < -1".format(min_z))
          
            os.remove(name1)
            os.remove(name2)
            time.sleep(0.5)
            continue
            flag = False
        else:
            print("###############")
            print("{} < 0.1 PASSED".format(CRIT2))
            print("{} max x < 1".format(max_x))
            print("{} min x > -1".format(min_x))
            print("{} max y < 1".format(max_y))
            print("{} min y > -1".format(min_y))
            print("{} max z < 1".format(max_z))
            print("{} min z > -1".format(min_z))
            print("###############")
            flag = True
        time.sleep(3.0)         
    arr = os.listdir()
    arr_new=list(filter(lambda k: '.npy' in k, arr)) 
    print(len(arr_new))
    qp1 = []
    H1 = []
    qp2 = []
    H2 = []
    for npy in arr_new:
        with open(npy, 'rb') as f:
            pos = torch.tensor(np.load(f))
            
            vel = torch.tensor(np.load(f))
            t = torch.tensor(np.load(f))
            mass = np.load(f)
            K = torch.tensor(np.load(f))
            P = torch.tensor(np.load(f))
        print(K.shape)
        if pos.shape[1] == bodies:
            qp1.append(torch.cat((pos.unsqueeze(1),vel.unsqueeze(1)),dim=-1))
            H1.append((K+P).reshape(-1,1,1))
            time.sleep(1.0) 
            os.remove(npy)
        elif pos.shape[1] == (bodies+1):	
            qp2.append(torch.cat((pos.unsqueeze(1),vel.unsqueeze(1)),dim=-1))
            H2.append((K+P).reshape(-1,1,1))
            time.sleep(1.0) 
        else:
            print("WROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG")
    out1 = torch.cat((qp1),dim=1)
    H_out1 = torch.cat((H1),dim=1)
    out2 = torch.cat((qp2),dim=1)
    H_out2 = torch.cat((H2),dim=1)
    print(out1.shape)
    print(H_out1.shape)
    print(out2.shape)
    print(H_out2.shape)
    filename1 = "nbody_" + str(bodies) + "_traj.pt"
    H_file1 = "nbody_" + str(bodies) + "_H.pt"
    filename2 = "nbody_" + str(bodies+1) + "_traj.pt"
    H_file2 = "nbody_" + str(bodies+1) + "_H.pt"
    torch.save(out1,ROOT_PATH+"/"+filename1)
    torch.save(H_out1,ROOT_PATH+"/"+H_file1)
    torch.save(out2,ROOT_PATH+"/"+filename2)
    torch.save(H_out2,ROOT_PATH+"/"+H_file2)
    print(H_out)
    return out1, H_out1, out2, H_out2      
        


if __name__ == "__main__":
    #start if from the folder corrected
    simulation_start(25,5,sim=False)
    #simulation_start(25,6,sim=False)
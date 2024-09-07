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
            name,H,x = simulate(N=bodies, t= 0, tEnd= 1.27,dt = 0.01, r=0.5, softening= 0.001, G = 1, plotRealTime=sim)
            CRIT =torch.std(torch.tensor(H))
            print(x.shape) 
            H = H.reshape((128,1))
            print(H.shape)
            
            
            q_x = x[:,:,0]
            q_y = x[:,:,1]
            q_z = x[:,:,2]
            max_x = np.max(q_x.flatten())
            min_x = np.min(q_x.flatten())
            max_y = np.max(q_y.flatten())
            min_y = np.min(q_y.flatten())
            max_z = np.max(q_z.flatten())
            min_z = np.min(q_z.flatten())
            if CRIT > 0.1 or max_x > 1 or min_x < -1 or max_y > 1 or min_y < -1or max_z > 1 or min_z < -1:
                print("{} > 0.1  FAILED".format(CRIT))
                print("{} max x > 1".format(max_x))
                print("{} min x < -1".format(min_x))
                print("{} max y > 1".format(max_y))
                print("{} min y < -1".format(min_y))
                print("{} max z > 1".format(max_z))
                print("{} min z < -1".format(min_z))
                os.remove(name)
                flag = False
            else:
                print("###############")
                print("{} < 0.1 PASSED".format(CRIT))
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
    qp = []
    H = []
    for npy in arr_new:
        with open(npy, 'rb') as f:
            pos = torch.tensor(np.load(f))
            vel = torch.tensor(np.load(f))
            t = torch.tensor(np.load(f))
            mass = np.load(f)
            K = torch.tensor(np.load(f))
            P = torch.tensor(np.load(f))
        print(K.shape)
        qp.append(torch.cat((pos.unsqueeze(1),vel.unsqueeze(1)),dim=-1))
        H.append((K+P).reshape(-1,1,1))
        time.sleep(1.0) 
        os.remove(npy)	
    out = torch.cat((qp),dim=1)
    H_out = torch.cat((H),dim=1)
    print(out.shape)
    print(H_out.shape)
    filename = "nbody_" + str(bodies) + "_traj.pt"
    H_file = "nbody_" + str(bodies) + "_H.pt"
    torch.save(out,ROOT_PATH+"/"+filename)
    torch.save(H_out,ROOT_PATH+"/"+H_file)
    print(H_out)
    return out, H_out      
        


if __name__ == "__main__":
    #start if from the folder corrected
    simulation_start(25,4,sim=False)
    simulation_start(25,5,sim=False)
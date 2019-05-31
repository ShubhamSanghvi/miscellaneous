import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

num_epocs = 1
delay_s = 0.25

def plot_line(params,lplot):
    if lplot:
        lplot.pop(0).remove()
  
    x = np.arange(-0.1,1.1,0.1)  
    y = (params[2]/params[1]) - ((params[0]/params[1]) * x)
    lplot = plt.plot(x, y,color='black',linewidth=2)
    plt.pause(0.05)
    return lplot
    
if __name__=='__main__':
    data = pd.read_csv("data.csv",header=None)  
    data.columns = ['X','Y','LABEL']
    
    #shuffle data frame points
    data = data.sample(frac=1).reset_index(drop=True)
    print(data.head())

    colors = [ 'm' if i==1 else 'c' for i in data['LABEL']]

    plt.ion()
    fig= plt.figure()
    fig.suptitle('Naive Classifier', fontsize=20)
    plt.ylim([-0.25,1.25])
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=16)
    plt.scatter(data['X'],data['Y'],color=colors)
    plt.pause(0.05)

    d = input("key")
    
    plot_obj = None
    np.random.seed(3)
    wb = np.random.rand(3)
    plot_obj = plot_line(wb,plot_obj)
    
    for _ in range(0,num_epocs):
        for id in data.index:
            
            prev_pt = plt.scatter(data.loc[id,'X'],data.loc[id,'Y'],color='y',s=100,alpha = 0.5,marker='D')
            plt.pause(0.05)
            time.sleep(delay_s)
            
            pred_l = wb[0] * data.loc[id,'X'] + wb[1] * data.loc[id,'Y'] - wb[2] < 0
            
            if pred_l != data.loc[id,'LABEL']:
                err_pt = plt.scatter(data.loc[id,'X'],data.loc[id,'Y'],color='R',s=100,marker='D')
                plt.pause(0.05)
    
                # We got to get it above the line
                # Naive approach says, increase if false negative and decrease if false positve
                wb[0] += (data.loc[id,'X']) * (0.5 - pred_l) * 0.1
                wb[1] += (data.loc[id,'Y']) * (0.5 - pred_l) * 0.1
                wb[2] += (0.5 - pred_l) * 0.1
                
                plot_obj = plot_line(wb,plot_obj)
                time.sleep(delay_s)
    
                if 'err_pt' in globals():
                    err_pt.remove()
                
            if 'prev_pt' in globals():
                prev_pt.remove()
           
    plt.ioff()
    plt.show()    

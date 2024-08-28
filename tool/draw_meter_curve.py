import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import glob
import os
import numpy as np
from scipy.interpolate import interp1d


def read_from_txt(filename):
    with open(filename, "r") as f:
        context = f.readlines()
    meter, prob = [], []
    index = [0,1,2,3,5,7,10,20,30,40,50,60,70,80,90,99]
    for line in np.array(context)[index]:
        m, p = line.split("\n")[0].split(" ")
        m = float(m)
        p = float(p)
        meter.append(m)
        prob.append(p)
    return meter, prob


root_dir = "./result_files/circle-level"

if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 4))
    plt.grid()

    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'maroon', 'peru', 'teal']

    plt.yticks(fontproperties='Times New Roman', size=8)
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.xlim(0,100)
    plt.ylim(0,1)

    filenames = glob.glob(os.path.join(root_dir, "*.txt"))
    filenames.sort()
    for ind, filename in enumerate(filenames):
        meter, prob = read_from_txt(filename)
        cubic_interploation_model=interp1d(meter,prob,kind="cubic")
        xs=np.linspace(0,99,500)
        ys=cubic_interploation_model(xs)
        label = filename.split("/")[-1].split(".txt")[0]
        plt.plot(xs, ys, c=color_list[ind], label=label)

    # plt.plot(x,s_d_AP,c='r',marker = 's',label='Satellite -> Drone',linewidth=1.5,markersize=6)
    # x_major_locator = MultipleLocator(1)

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)

    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 5})
    # plt.ylabel("AP(%)",fontdict={'family' : 'Times New Roman', 'size': 16})
    # plt.xlabel("numbers of sampling k",fontdict={'family' : 'Times New Roman', 'size': 16})
    # plt.tight_layout()
    fig.savefig('test.png', dpi=300)
    # plt.show()

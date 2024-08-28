import matplotlib.pyplot as plt
import numpy as np

negw_x = [1,10,30,50,80,100,110,120,130,140,150,160,200]
negw_y = [0.654,0.728,0.722,0.748,0.744,0.739,0.749,0.753,0.758,0.742,0.743,0.740,0.737]

centerR_x = [1,3,5,7,11,15,21,25,29,31,41,51,61,75,101]
centerR_y = [0.682,0.664,0.674,0.677,0.664,0.678,0.695,0.707,0.698,0.736,0.714,0.721,0.718,0.713,0.675]


if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'maroon', 'peru', 'teal']
    
    ax1.grid()
    ax1.plot(negw_x, negw_y, marker='o', markersize=8, linestyle='-', c=color_list[2])
    ax1.set_xlabel('Negative Weight\n(a)', fontdict={'family' : 'Times New Roman', 'size': 16})    
    ax1.set_ylabel('RDS', fontdict={'family' : 'Times New Roman', 'size': 16})
    ax1.tick_params(axis='both', labelsize=12, which='both', direction='in', width=1)
    
    ax2.grid()
    ax2.plot(centerR_x, centerR_y, marker='o', markersize=8, linestyle='-', c=color_list[2])
    ax2.set_xlabel('Positive Range R\n(b)', fontdict={'family' : 'Times New Roman', 'size': 16})    
    ax2.set_ylabel('RDS', fontdict={'family' : 'Times New Roman', 'size': 16})
    ax2.tick_params(axis='both', labelsize=12, which='both', direction='in', width=1)

    for tick in ax1.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax1.get_yticklabels():
        tick.set_fontname('Times New Roman')
        
    for tick in ax2.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax2.get_yticklabels():
        tick.set_fontname('Times New Roman')

    plt.tight_layout()
    fig.savefig('test.jpg', dpi=300)

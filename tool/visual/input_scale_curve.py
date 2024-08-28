import matplotlib.pyplot as plt
import numpy as np

x_satellite = list(range(320,480,32))
x_uav = list(range(96,176,16))

RDS_satellite = [0.714,0.748,0.758,0.748,0.764]
RDS_uav = [0.727,0.701,0.758,0.710,0.760]

MA3_uav = [0.109, 0.103, 0.131, 0.138, 0.144]
MA3_Satellite = [0.103, 0.123, 0.131, 0.140, 0.152]



inference_time_satellite = np.array([21.8,22.0,22.1,22.8,23.0])
inference_time_uav = np.array([21.8,22.0,22.1,22.3,22.6])


if __name__ == '__main__':
    fig = plt.figure(figsize=(8,8))
    # plt.subplot(3 ,2, 1)
    # plt.grid()
    # plt.yticks(fontproperties='Times New Roman', size=10)
    # plt.xticks(fontproperties='Times New Roman', size=10)
    # plt.plot(x_uav,RDS_uav,c='b',marker = 'o',linewidth=1.5,markersize=8)
    # plt.ylabel("RDS(%)",fontdict={'family' : 'Times New Roman', 'size': 15})
    # plt.xlabel("UAV-View Image Size\n(a)",fontdict={'family' : 'Times New Roman', 'size': 15})
    # plt.tight_layout()

    # plt.subplot(3, 2, 2)
    # plt.grid()
    # plt.yticks(fontproperties='Times New Roman', size=10)
    # plt.xticks(fontproperties='Times New Roman', size=10)
    # plt.plot(x_satellite,RDS_satellite,c='b',marker = 'o',linewidth=1.5,markersize=8)
    # plt.ylabel("RDS(%)", fontdict={'family': 'Times New Roman', 'size': 15})
    # plt.xlabel("Satellite-View Image Size\n(b)", fontdict={'family': 'Times New Roman', 'size': 15})
    # plt.tight_layout()

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.plot(x_uav,MA3_uav,c='b',marker = 'o',linewidth=1.5,markersize=8)
    plt.ylabel("MA@3(%)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.xlabel("UAV-View Image Size\n(a)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.plot(x_satellite,MA3_Satellite,c='b',marker = 'o',linewidth=1.5,markersize=8)
    plt.ylabel("MA@3(%)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.xlabel("Satellite-View Image Size\n(b)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.plot(x_uav, inference_time_uav, c='r', marker='s', linewidth=1.5, markersize=8)
    plt.ylabel("Inference Times (s/1000samples)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.xlabel("UAV-View Image Size\n(c)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.plot(x_satellite, inference_time_satellite, c='r', marker='s', linewidth=1.5, markersize=8)
    plt.ylabel("Inference Times (s/1000samples)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.xlabel("Satellite-View Image Size\n(d)", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.tight_layout()

    fig.savefig('tool/visual/curve/scale_uav_satellite.eps', dpi=600, format="eps")
    fig.savefig('tool/visual/curve/scale_uav_satellite.jpg', dpi=600, format="jpg")
    # plt.show()

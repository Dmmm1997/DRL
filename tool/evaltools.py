import numpy as np
import math


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.data_config["Satellitehw"][0]
    y_rate = (pred_Y - label_Y) / opt.data_config["Satellitehw"][1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * 10 * distance)
    return result


def Distance(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a)
                                  * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


def distance(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算两个经纬度坐标之间的球面距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c * 1000  # 6371km 为地球半径，转换为米

    return distance


if __name__ == "__main__":
    l1 = Distance(100.111111, 40.222222, 120.232322, 44.2132311)
    l2 = distance(100.111111, 40.222222, 120.232322, 44.2132311)
    print(l1, l2)


import math
import matplotlib.pyplot as plt


warmup_iters = 500
total_iters = 24 * 846
warmup_start_lr = 0.1
minimun_lr = 0.01
init_lr = 1

lambda_func = lambda iter: ((1-warmup_start_lr)*iter / warmup_iters+warmup_start_lr) if iter < warmup_iters else minimun_lr \
                    if 0.5 * (1+math.cos(math.pi*(iter - warmup_iters)/(total_iters-warmup_iters))) < minimun_lr \
                    else 0.5 * (1+math.cos(math.pi*(iter - warmup_iters)/(total_iters-warmup_iters)))

x,y = [],[]
for i in range(total_iters):
    x.append(i)
    y.append(lambda_func(i))


plt.plot(x,y)

plt.savefig("1.png")




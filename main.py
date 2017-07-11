from scipy.stats import beta, bernoulli
import numpy as np
# import matplotlib.pyplot as plt

ctr_old = 0.012
rate = 0.002
epoch_size = 1000
a_size = int(epoch_size * (1 - rate))
b_size = int(epoch_size * rate)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)

means_collector_prop = []
means_collector_cur = []


for improve in [1.01, 1.02, 1.05]:
    for _ in range(100):

        ctr_new = ctr_old * improve
        stds1, means1, vals1 = [], [], []
        stds2, means2, vals2 = [], [], []

        #### initiation ####
        a1, b1 = 12, 1000
        a2, b2 = 12, 1000

        for epoch in range(1, 100000):
            cur_outcome = bernoulli.rvs(ctr_old, size=a_size)
            prop_outcome = bernoulli.rvs(ctr_new, size=b_size)

            a1 += np.sum(cur_outcome)
            a2 += np.sum(prop_outcome)

            b1 += a_size
            b2 += b_size

            vals1.append(beta.rvs(a1, b1))
            # stds1.append(np.std(vals1))
            means1.append(np.mean(vals1[-200:]))

            vals2.append(beta.rvs(a2, b2))
            # stds2.append(np.std(vals2))
            means2.append(np.mean(vals2[-200:]))

        print(_, ctr_old, ctr_new)
        means_collector_prop.append(means2)
        means_collector_cur.append(means1)

        # ax1.plot(means2, label='mean_prop', alpha=0.5)

    # ax1.plot(means1, label='mean_cur', alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.show()

    #### New models` stats ####
    for e_ in range(100):
        marker = e_ * 1000
        datum_prop = np.array([zz[marker] for zz in means_collector_prop])
        datum_cur = np.array([zz[marker] for zz in means_collector_cur])
        impr = np.mean(datum_prop > datum_cur)
        reg = np.mean(datum_prop <= datum_cur)
        bf = impr/reg
        print('Modeled improve: {}, Epoch: {}, Mean: {}, Std: {}, bf: {}, Test is better: {}'.format(improve, marker, np.mean(datum_prop), np.std(datum_prop), bf, impr))



### Здесь мы проверяем как работает модель на придуманных значениях


#### initiation #### Передам данные котоые мы наблюдали  ранее
a1, b1 = 12, 1000
a2, b2 = 12, 1000

ctr_new = ctr_old * 1.04
# ctr_new = np.random.normal(ctr_old + 0.001, 0.001)

marker_prelim = 0
for epoch in range(1, 10000):
    marker = epoch * 1000
    cur_outcome = bernoulli.rvs(ctr_old, size=a_size)
    prop_outcome = bernoulli.rvs(ctr_new, size=b_size)

    a1 += np.sum(cur_outcome)
    a2 += np.sum(prop_outcome)

    b1 += a_size
    b2 += b_size

    if marker > 1000000:
        datum_cur = beta.rvs(a1, b1, size=1000)
        datum_prop = beta.rvs(a2, b2, size=1000)
        impr = np.mean(datum_prop > datum_cur)
        reg = np.mean(datum_prop <= datum_cur)
        bf = impr / reg

        if bf > 3 and marker_prelim == 0:
            print('prelimitory', marker, bf, ctr_new)
            marker_prelim = 1
        if bf > 9:
            print('final', marker, bf, ctr_new)
            break
        if not marker % 1000000:
            print(marker)
from matplotlib import pyplot as plt

file = 'FGSM_LASS_output/epochs_110/target_model_ResNet18/lr_0.1/lr_ap_0.0003/iter_num_20/factor_0.9/epsilon_8/output.log'

values = []
cnt = 0
cnt2 = 0
is_skip = False
value = 0.

with open(file, 'r') as lines:
    for i, line in enumerate(lines):
        if i == 0 or is_skip:
            if i != 0:
                cnt2 += 1
                if cnt2 == 4:
                    cnt2 = 0
                    is_skip = False
            continue
        value += float(line.split('=')[-2].split(',')[0])
        cnt += 1
        if cnt == 20:
            values.append(value / cnt)
            value = 0.
            cnt = 0
            is_skip = True

plt.figure()
x = [i + 1 for i in range(len(values))]
plt.plot(x, values)
fontdict = {'family': 'Times New Roman', 'fontsize': 12, 'fontstyle': 'italic'}
plt.xlabel('Epoch', fontdict=fontdict)
plt.ylabel('Mean Attack Step Size', fontdict=fontdict)
plt.show()

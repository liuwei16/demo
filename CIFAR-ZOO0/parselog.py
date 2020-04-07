from matplotlib import pyplot as plt
import numpy as np
import re
train_file = "myexe/logs_18_cos200_lsr.txt"
acc_file = "myexe/acc_18_cos200_lsr.txt"
with open(train_file,'r') as f1:
    with open(acc_file, 'w') as f2:
      for line in f1:
            if 'test acc' in line:
                data = re.findall('\d{2}\.\d{3}\%', line)
                f2.write(data[0][:-1]+'\n')

# accuration = []
# with open(acc_file, 'r') as f5:
#     for line in f5:
#         accuration.append(float(line))

# plt.xlabel('Epoch')
# plt.ylabel('Accuration')
# plt.plot(np.arange(1,len(accuration)+1)*2, accuration)
# plt.savefig('a.png')



acc1 = []
acc2 = []
acc3 = []
with open( "myexe/acc_18_cos200_lsr.txt", 'r') as f5:
    for line in f5:
        acc1.append(float(line))
with open( "myexe/acc_18_cos200_mixup.txt", 'r') as f6:
    for line in f6:
        acc2.append(float(line))
with open('myexe/acc_18_cos200.txt', 'r') as f7:
    for line in f7:
        acc3.append(float(line))

plt.xlabel('Epoch')
plt.ylabel('Accuration')
plt.plot(np.arange(1,len(acc1)+1)*2, acc1,label='200_cos_lsr')
plt.plot(np.arange(1,len(acc2)+1)*2, acc2,label='200_cos_mixup')
plt.plot(np.arange(1,len(acc3)+1)*2, acc3,label='200_cos')
plt.legend(loc = 'best')
plt.grid()
plt.savefig('a.png')
print('200_cos_lsr',np.argmax(acc1)*2, np.max(acc1))
print('200_cos_mixup',np.argmax(acc2)*2, np.max(acc2))
print('200_cos',np.argmax(acc3)*2, np.max(acc3))
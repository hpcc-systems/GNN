import string
xtest = open('../Datasets/HAR/Raw/test/Inertial Signals/body_acc_x_test.txt')
for i in range(2):
  line = xtest.readline()
  tokens = line.split()
  line = ','.join(tokens)
  print(line)
  print('nTokens = ', len(tokens))
recs = 0
xtest = open('../Datasets/HAR/Raw/test/Inertial Signals/body_acc_x_test.txt')
while 1:
  line = xtest.readline()
  if not line:
    break
  recs += 1

print('recCount = ', recs)
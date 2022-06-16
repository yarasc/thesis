import os

folders = os.listdir('/Users/zeehondje/Documents/GitHub/thesis/dvs_data/DVSGesture/ibmGestureTest')

for user in folders:
    path = os.path.join("/Users/zeehondje/Documents/GitHub/thesis/dvs_data/DVSGesture/ibmGestureTest",user, "10.npy")
    if os.path.exists(path):
        print(os.remove(path))
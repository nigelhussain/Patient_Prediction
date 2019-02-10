import os

dir_list = []
root = '/Users/nigel.hussain/Desktop/Therapy'
for path, subdirs, files in os.walk(root):
	for name in files:
		path = os.path.join(root, subdirs, name)
		dir_list.append(path)
	print(dir_list)

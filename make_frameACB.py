"""Make the original frames into frameAs, frameBs, frameCs(same as frameBs here).

frameAs are original frames[0:-1] while frameCs/frame Bs are orginal frames[1:]

Usage:
    python make_frameACB.py 
    python make_frameACB.py path_to_image_folders all/single
"""
import os
import sys


# Choose the dir you want
# dirs = sorted([i for i in os.listdir('.') if i[:5] in ['glass']
# # and int(i.split('_')[-1].split('.')[0]) > 0
# ]
# # , key=lambda x: int(x.split('_')[-1])
# )[:]

cwd = os.getcwd()

if len(sys.argv) > 1:
    os.chdir(sys.argv[1]) 
if sys.argv[2] == 'single':
    dirs = ['./']
else:
    dirs = sorted([i for i in os.listdir('.')])[:]

for d in dirs:

    if os.path.isfile(d):
        continue
    os.chdir(d)
    try:
        os.mkdir('frameA')
    except:
        print("%s has already been processed."%d)
        os.chdir('..')
        continue

    print(d)
    os.mkdir('frameC') 
    try:
        files = sorted([f for f in os.listdir('.') if f[-4:] == '.png'], key=lambda x: int(x.split('.')[0]))
    except:
        os.chdir('..')
        continue
    os.system('cp ./*png frameA && cp ./*png frameC') 
    os.remove(os.path.join('frameA', files[-1])) 
    os.remove(os.path.join('frameC', files[0])) 
    for f in sorted(os.listdir('frameC'), key=lambda x: int(x.split('.')[0])): 
        f_new = os.path.join('frameC', '%06d' % (int(f.split('.')[0])-1) + '.png') 
        f = os.path.join('frameC', f) 
        os.rename(f, f_new) 
    os.system('cp -r frameC frameB') 
    # os.system('rm ./*.png')
    os.chdir('..')

os.chdir(cwd)


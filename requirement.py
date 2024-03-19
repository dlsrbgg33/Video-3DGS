      
# Here lists up commands to set-up our Video-3DGS-Recon
import os
bash_script = ''

# 1. COLMAP Setup
bash_script += 'sudo apt-get -y install \
     ninja-build \
     build-essential \
     libboost-program-options-dev \
     libboost-filesystem-dev \
     libboost-graph-dev \
     libboost-system-dev \
     libeigen3-dev \
     libflann-dev \
     libfreeimage-dev \
     libmetis-dev \
     libgoogle-glog-dev \
     libgtest-dev \
     libsqlite3-dev \
     libglew-dev \
     qtbase5-dev \
     libqt5opengl5-dev \
     libcgal-dev \
     libceres-dev' + '\n'
     
bash_script += 'cd required_modules/colmap_dev' + '\n'
bash_script += 'sudo rm -rf build' + '\n' 
bash_script += 'mkdir build' + '\n'  
bash_script += 'cd build' + '\n' 
bash_script += 'cmake .. -GNinja' + '\n' 
bash_script += 'ninja' + '\n' 
bash_script += 'sudo ninja install' + '\n'

# 2. install 3dgs related packages
bash_script += 'python3 -m pip install submodules/depth-diff-gaussian-rasterization' + '\n'
bash_script += 'python3 -m pip install submodules/simple-knn' + '\n'
bash_script += 'python3 -m pip install opencv-python' + '\n'
bash_script += 'python3 -m pip install tqdm' + '\n'
bash_script += 'python3 -m pip install scipy' + '\n'
bash_script += 'python3 -m pip install plyfile==0.8.1' + '\n'
bash_script += 'python3 -m pip install scikit-image' + '\n'
bash_script += 'python3 -m pip install transformers' +'\n'

# 3. install multi-hash resolution (tiny-cuda)
bash_script += 'cd submodules/tiny-cuda-nn' + '\n'
bash_script += 'sudo apt-get install build-essential git' + '\n'
bash_script += 'sudo rm -rf build/' + '\n'
bash_script += 'cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo' + '\n'
bash_script += 'cmake --build build --config RelWithDebInfo -j' +'\n'
bash_script += 'cd bindings/torch' + '\n'
bash_script += 'sudo python3 setup.py install' +'\n'


# 4. adopt video_segmentor (DEVA)
bash_script += 'mkdir models/video_segmentor' + '\n'
bash_script += 'git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git' + '\n'

with open('/opt/tiger/entry_script.sh', 'w+') as f:
    f.write(bash_script)

os.system('bash /opt/tiger/entry_script.sh')



    
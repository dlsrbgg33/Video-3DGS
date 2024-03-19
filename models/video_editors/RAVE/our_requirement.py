import os


# we use - for space, ~ for $
#advanced_kmax=str(os.environ['ADVANCED_KMAX'])

bash_script = ''

bash_script += 'python3 -m pip install diffusers==0.18.2 -i https://bytedpypi.tiktoke.org/simple' + '\n'
bash_script += 'python3 -m pip install cython -i https://bytedpypi.tiktoke.org/simple' + '\n'
bash_script += 'python3 -m pip install basicsr -i https://bytedpypi.tiktoke.org/simple' + '\n'
bash_script += 'python3 -m pip install einops -i https://bytedpypi.tiktoke.org/simple' + '\n'
bash_script += 'python3 -m pip install timm -i https://bytedpypi.tiktoke.org/simple' + '\n'
bash_script += 'python3 -m pip install accelerate -i https://bytedpypi.tiktoke.org/simple' + '\n'



with open('/opt/tiger/entry_script.sh', 'w+') as f:
    f.write(bash_script)

os.system('bash /opt/tiger/entry_script.sh')

    
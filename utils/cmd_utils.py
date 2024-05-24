import os
import subprocess

def edit_cmd_run(editor, cuda_num, prev_vid, text, edited_path, update_num, ori_text=None, ensembled_strategy="single"):
    
    cmd = ""
    if editor == "tokenflow":
        cmd =(
            f'CUDA_VISIBLE_DEVICES={cuda_num} python3 models/video_editors/tokenflow/src/tokenflow.py '
            f'-im {prev_vid} '
            f'-t "{text}" --batch_size 4 --batch_pivot --cpu '
            f'--out_dir "{edited_path}" --update_num {update_num} '
            f'--ensembled_strategy {ensembled_strategy} '
        )
        print(cmd)
        os.system(cmd)
    elif editor == "RAVE":
        # 
        cmds =[]
        cmd = (
            f'eval "$(conda shell.bash hook)" && '
            f'conda deactivate && '
            f'conda activate rave && '
            f'MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES={cuda_num} python3.8 models/video_editors/RAVE/scripts/run_experiment.py '
            f'{prev_vid} "{edited_path}" "{text}" '
            f'{update_num} {ensembled_strategy} && ' 
            f'conda deactivate && '
            f'conda activate 3dgs '
        )
        cmds.append(cmd)
        processes = [
            subprocess.Popen(cmd, shell=True, executable="/bin/bash") for cmd in cmds
        ]
        for p in processes:
            p.wait()
    else:
        NotImplementedError("Currently, we support using above editors")


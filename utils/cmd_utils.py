import os

def edit_cmd_run(editor, cuda_num, prev_vid, text, edited_path, progressive, output_num, ori_text=None):
    
    cmd = ""
    if editor == "tokenflow":
        cmd =(
            f'CUDA_VISIBLE_DEVICES={cuda_num} python3 models/video_editors/tokenflow/src/tokenflow.py '
            f'-im {prev_vid} '
            f'-t "{text}" --batch_size 4 --batch_pivot --cpu '
            f'--out_dir "{edited_path}" --progress {progressive} '
            f'--output_num {output_num}'
        )
    elif editor == "RAVE":
        cmd =(
            f'CUDA_VISIBLE_DEVICES={cuda_num} python3.8 models/video_editors/RAVE/scripts/run_experiment.py '
            f'{prev_vid} "{edited_path}" "{text}" '
            f'{progressive} {output_num}'
        )
    else:
        NotImplementedError("Currently, we support using above editors")

    print(cmd)
    os.system(cmd)

FILE_PATH="datasets/edit/loveu/LOVEU-TGVE-2023_Dataset_davis.csv"

# Read each line from the file
CUDA=2
while IFS=/ read -r video_name gt_caption style_change object_change background_change multiple_change source
do
    # Skip header line
    if [ "$video_name" != "Video name" ]; then
        for GNUM in -1
        do
            for ITERS in 3000
            do
                SOURCE_DATA="datasets/edit/loveu/DAVIS_480p/480p_frames/${video_name}"
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_edit_timecheck_3k_all.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="single_update" --initial_editor 4 \
                    --prompt "$style_change"  --cate="sty" --progressive_num 0\
                    --cuda_num=$CUDA
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_edit_timecheck_3k_all.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="single_update" --initial_editor 2 \
                    --prompt "$style_change"  --cate="sty" --progressive_num 0\
                    --cuda_num=$CUDA
                # CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_v2.py -s $SOURCE_DATA \
                #     --iteration=$ITERS --use_dual \
                #     --group_size=$GNUM --deform_type "multi" \
                #     --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                #     --editing_method="progressive" --initial_editor 1 \
                #     --prompt "$background_change"  --cate="back" --progressive_num 2\
                #     --cuda_num=$CUDA
                # CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_v2.py -s $SOURCE_DATA \
                #     --iteration=$ITERS --use_dual \
                #     --group_size=$GNUM --deform_type "multi" \
                #     --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                #     --editing_method="progressive_acc" --initial_editor 1 \
                #     --prompt "$background_change"  --cate="back" --progressive_num 2\
                #     --cuda_num=$CUDA
                # CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_v2.py -s $SOURCE_DATA \
                #     --iteration=$ITERS --use_dual \
                #     --group_size=$GNUM --deform_type "multi" \
                #     --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                #     --editing_method="single_update" --initial_editor 1 \
                #     --prompt "$multiple_change"  --cate="multi" --progressive_num 0\
                #     --cuda_num=$CUDA
                # CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_v2.py -s $SOURCE_DATA \
                #     --iteration=$ITERS --use_dual \
                #     --group_size=$GNUM --deform_type "multi" \
                #     --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                #     --editing_method="progressive" --initial_editor 1 \
                #     --prompt "$multiple_change"  --cate="multi" --progressive_num 2\
                #     --cuda_num=$CUDA
                # CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid_full_v2.py -s $SOURCE_DATA \
                #     --iteration=$ITERS --use_dual \
                #     --group_size=$GNUM --deform_type "multi" \
                #     --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                #     --editing_method="progressive_acc" --initial_editor 1 \
                #     --prompt "$multiple_change"  --cate="multi" --progressive_num 2\
                #     --cuda_num=$CUDA
            done
        done
    fi
done < "$FILE_PATH"


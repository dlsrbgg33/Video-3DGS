
FILE_PATH="datasets/edit/loveu/LOVEU-TGVE-2023_Dataset_youtube.csv"

# Read each line from the file
CUDA=0
while IFS=/ read -r video_name gt_caption style_change object_change background_change multiple_change source
do
    # Skip header line
    if [ "$video_name" != "Video name" ]; then
        for GNUM in 1
        do
            for ITERS in 3000
            do
                SOURCE_DATA="datasets/edit/loveu/youtube_480p/480p_frames/${video_name}"
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="norecursive_single" --initial_editor 2 \
                    --prompt "$style_change"  --cate="sty" --recursive_num 0\
                    --cuda_num=$CUDA
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="norecursive_single" --initial_editor 2 \
                    --prompt "$object_change"  --cate="obj" --recursive_num 0\
                    --cuda_num=$CUDA
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="norecursive_single" --initial_editor 2 \
                    --prompt "$background_change"  --cate="back" --recursive_num 0\
                    --cuda_num=$CUDA
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual \
                    --group_size=$GNUM --deform_type "multi" \
                    --random_pts_num 60000  --radius 1 3 --biloss_weight 2.0 \
                    --editing_method="norecursive_single" --initial_editor 2 \
                    --prompt "$multiple_change"  --cate="multi" --recursive_num 0\
                    --cuda_num=$CUDA
            done
        done
    fi
done < "$FILE_PATH"





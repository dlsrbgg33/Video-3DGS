CUDA=0

# for NAME in bear car-shadow dog elephant gold-fish hike horsejump-low lab-coat longboard motorbike swing blackswan camel car-turn flamingo hockey kid-football lucia scooter-gray boat car-roundabout cows drift-turn horsejump-high kite-surf mallard-water mbike-trick rhino
for NAME in blackswan
do
    SOURCE_DATA="datasets/recon/DAVIS/JPEGImages/480p/${NAME}"
    for GNUM in 1
    do
        for EXTSC in 4
        do
            for ITERS in 3000
            do
                CUDA_VISIBLE_DEVICES=$CUDA python3 main_3dgsvid.py -s $SOURCE_DATA \
                    --iteration=$ITERS --use_dual --group_size=$GNUM --only_eval
            done
        done
    done
done

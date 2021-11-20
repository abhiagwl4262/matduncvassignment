# matduncvassignment

##To run 
CUDA_VISIBLE_DEVICES=6 python3 -m pdb train.py --datadir 2_class_data/ --epochs 50 --classes 2 --batch_size 1 --logdir runs --scheduler step_lr --optimizer adam --wd

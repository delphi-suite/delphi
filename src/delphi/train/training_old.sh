export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=/ceph/jbrinkma/cache/transformers
export HF_DATASETS_CACHE=/ceph/jbrinkma/cache/datasets

python3 training_old.py --vocab_source=custom  --vocab_size=4096 --max_seq_len=512 --dim=48 --n_layers=8 --n_heads=8 --n_kv_heads=4

SUBJECT="194"
python train.py \
-s /home/vitor/Documents/doc/VHAP/export/nersemble_v2/194_EXP-1-head_v16_DS4_whiteBg_staticOffset_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--eval --bind_to_mesh --white_background --port 60000
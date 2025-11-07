TASK=mlvu_mc MAX_NUM_FRAMES=1024 EXTRA_PARAM_SELECT_AND_DROP=256 EXTRA_PARAM_DUPLICATE=2 bash my_scripts_XComp/run.sh >my_scripts_XComp/mlvu_256_from_1024_dp2.log































# TASK=mlvu_mc MAX_NUM_FRAMES=512 EXTRA_NEED_SCORE=false EXTRA_PARAM_SELECT_AND_DROP=0 bash my_scripts/new7/run.sh >my_scripts/new7/mlvu_512.log
# TASK=mlvu_mc MAX_NUM_FRAMES=1024 EXTRA_NEED_SCORE=false EXTRA_PARAM_SELECT_AND_DROP=0 bash my_scripts/new7/run.sh >my_scripts/new7/mlvu_1024.log
# TASK=mlvu_mc MAX_NUM_FRAMES=1024 EXTRA_PARAM_SELECT_AND_DROP=512 EXTRA_PARAM_DUPLICATE=2 bash my_scripts/new7/run.sh >my_scripts/new7/mlvu_512_from_1024_dp2.log
# TASK=mlvu_mc MAX_NUM_FRAMES=2048 EXTRA_PARAM_SELECT_AND_DROP=512 EXTRA_PARAM_DUPLICATE=2 bash my_scripts/new7/run.sh >my_scripts/new7/mlvu_512_from_2048_dp2.log


# TASK=mlvu_mc MAX_NUM_FRAMES=1024 EXTRA_PARAM_SELECT_AND_DROP=512 EXTRA_PARAM_DUPLICATE=2 bash my_scripts/new7/run.sh >my_scripts/new7/fixed_mlvu_512_from_1024_dp2.log
# TASK=mlvu_mc MAX_NUM_FRAMES=2048 EXTRA_PARAM_SELECT_AND_DROP=1024 EXTRA_PARAM_DUPLICATE=2 bash my_scripts/new7/run.sh >my_scripts/new7/fixed_mlvu_1024_from_2048_dp2.log
# TASK=mlvu_mc MAX_NUM_FRAMES=4096 EXTRA_PARAM_SELECT_AND_DROP=1024 EXTRA_PARAM_DUPLICATE=4 EXTRA_PARAM_SCORE_TYPE=default bash my_scripts/new7/run.sh >my_scripts/new7/fixed_mlvu_1024_from_4096_default_dp4.log


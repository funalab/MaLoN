#!/bin/zsh
cd ../../lib/models/

MODEL=sv  # 'mv' or 'sv'
RESULT_DIR=../results/single_view_lstm
CFG=../../../confs/single_view.cfg
LOG_FILE=log_invivo_sv_eval.txt
touch ${LOG_FILE}
# ====================================================

# IRMA switch-on dataset ===
echo "=== IRMA switch-on dataset ==="
# inference
INPUT=../datasets/irma_datasets/switch_on/Switch-on_concatenated.tsv
DSNAME=irma_switch_on
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/irma_datasets/switch_on/IRMA_network.txt
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}
# ==========================


# IRMA switch-off dataset ===
echo "=== IRMA switch-off dataset ==="
# inference
INPUT=../datasets/irma_datasets/switch_off/Switch-off_concatenated.tsv
DSNAME=irma_switch_off
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/irma_datasets/switch_off/IRMA_Simplified_network.txt
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}
# ==========================


# E.coli SOS pathway dataset ===
echo "=== Ecoli SOS pathway dataset ==="
# inference
INPUT=../datasets/ecoli_sospathway_datasets/ecoli_sospathway_concatenated.tsv
DSNAME=Ecoli_SOSPathway
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/ecoli_sospathway_datasets/Network_Ecoli_SOSPathway.txt
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}
# ===============================

mv ${LOG_FILE} ${RESULT_DIR}


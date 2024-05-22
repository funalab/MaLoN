#!/bin/zsh
cd ../../lib/models/

MODEL=mv_dnn
LOG_FILE=log_invivo_dnn_eval.txt
touch ${LOG_FILE}
DYN_REG_PLOT=../../tools/plot_dynamic_regulation_map.py 
# ====================================================

# Mv-DNN10 =============================================================
CFG=../../../confs/multi_view_dnn10.cfg
RESULT_DIR=../results/multi_view_dnn10

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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
# ===============================
# ======================================================================


# Mv-DNN36 =============================================================
CFG=../../../confs/multi_view_dnn36.cfg
RESULT_DIR=../results/multi_view_dnn36

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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
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

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
# ===============================
# ======================================================================

mv ${LOG_FILE} ${RESULT_DIR}


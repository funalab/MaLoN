#!/bin/zsh
source ~/python_environments/pytorch/bin/activate
cd ../../lib/models/

MODEL=$1  # 'mv' or 'sv'
RESULT_DIR=../results
CFG=../../../confs/multi_view.cfg
LOG_FILE=log_invivo_eval.txt
touch ${LOG_FILE}
DYN_REG_PLOT=../../tools/plot_dynamic_regulation_map.py
# ====================================================

# DDGni_E datase ===
echo "=== DDGni E dataset ==="
# inference
INPUT=../datasets/ddgni_datasets/DDGni_E.txt
DSNAME=DDGni_E
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/ddgni_datasets/DDGni_E_network.txt
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
# ==================

# DDGni_Yeast dataset ===
echo "=== DDGni Yeast dataset ==="
# inference
INPUT=../datasets/ddgni_datasets/DDGni_Yeast.txt
DSNAME=DDGni_Yeast
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/ddgni_datasets/DDGni_Yeast_network.txt
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}
# =======================

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


# Spellman cdc15 dataset ===
echo "=== Spellman cdc15 dataset ==="
# inference
INPUT=../datasets/spellman_datasets/Spellman_cdc15.txt
DSNAME=Spellman_cdc15
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/spellman_datasets/Spellman_Yeast9_network.tsv
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
# ==========================

# Spellman cdc28 dataset ===
echo "=== Spellman cdc28 dataset ==="
# inference
INPUT=../datasets/spellman_datasets/Spellman_cdc28.txt
DSNAME=Spellman_cdc28
OUTPUT_DIR=${RESULT_DIR}/${DSNAME}/
python main_${MODEL}.py ${INPUT} --config ${CFG} --name ${DSNAME}
cp ${CFG} ${OUTPUT_DIR}

# evaluation
TRUE_EDGELIST_FILE=../datasets/spellman_datasets/Spellman_Yeast9_network.tsv
PRED_EDGELIST_FILE=${OUTPUT_DIR}/${DSNAME}_edgelist.tsv
python evaluation.py ${PRED_EDGELIST_FILE} ${TRUE_EDGELIST_FILE} ${DSNAME} ${OUTPUT_DIR} >> ${LOG_FILE}

# visualization
DYN_REG_MAP_FILE=${OUTPUT_DIR}/${DSNAME}_dynamic_regulation_map.tsv
OUTPUT_DYN_REG_MAP_DIR=${OUTPUT_DIR}/dynamic_regulation_map
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type source --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
python ${DYN_REG_PLOT} ${DYN_REG_MAP_FILE} --type target --out ${OUTPUT_DYN_REG_MAP_DIR} --name ${DSNAME}
# ==========================

mv ${LOG_FILE} ${RESULT_DIR}


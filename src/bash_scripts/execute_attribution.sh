#!/bin/bash
set -e -o pipefail

# ./execute_attribution.sh local -1 &> ../../logs/execute_attribution_output.txt &

MODE=$1 # mod can be either local, srun or sbatch
SID=$2 # sid is used to select a setting when using local or srun (-1 equals all)
TASK=$3 # can be show to show the selected settings

echo "=================================================="
echo "Start $MODE"
echo "=================================================="

# ADJUST environment and working directory
if [[ "$MODE" == "srun" || "$MODE" == "sbatch" ]]
then
    # path to python env
    export PATH=/<path>/<to>/<env>/<env_name>/bin/:$PATH
    # path to script folder
    cd /<path>/<to>/timereise/src/scripts/
else
    # you may set the local env here
    cd ../scripts/
fi

#working dir
echo "Current working dir: $PWD"

# matches the slurm id and the setting number
if  [[ "$MODE" == "sbatch" ]]
then
    echo "Task ID: $SLURM_ARRAY_TASK_ID"
    SID=$SLURM_ARRAY_TASK_ID
else
    if [[ $SID -gt -1 ]]
    then
        echo "Selected setting: $SID"
    else
        echo "Selected all settings"
        SID=-1
    fi
fi

# logic for setting show and execution of python file
SETUP=0 # setup counter to select correct settings

# ADJUST define your parameters
DATASETS=("anomaly_new" "AsphaltPavementType" "AsphaltRegularity" \
"CharacterTrajectories" "Crop" "ECG5000" "ElectricDevices" "FaceDetection" \
"FordA" "HandOutlines" "MedicalImages" "MelbournePedestrian" \
"NonInvasiveFetalECGThorax1" "PhalangesOutlinesCorrect" "Strawberry" \
"UWaveGestureLibraryAll" "Wafer")

# ADJUST define your setup logic
for DID in ${!DATASETS[@]}
do
    # prints the settings if selected
    if [[ $SID -eq -1 || $SETUP -eq $SID ]]
    then
        # ADJUST print statement based on loops
        echo "=================================================="
        echo "Setup: $SETUP"
        echo "  Dataset: ${DATASETS[DID]}"
        echo "=================================================="

        if [[ $TASK != "show" ]]
        then
            # ADJUST script you want to execute
            python -u classification.py --verbose \
                    --dataset_name ${DATASETS[DID]} --exp_path 'baseline' \
                    --standardize --validation_split 0.3 \
                    --load_model --batch_size 32 \
                    --use_subset --subset_factor 100 \
                    --save_report --save_dicts --save_plots --not_show_plots \
                    --process_attributions --compute_attributions \
                    --attr_name "default" --save_memory
        fi
    fi
    SETUP=$(( SETUP+=1 ))
done

echo "=================================================="
echo "Finished $MODE execution"
echo "=================================================="

#!/bin/bash

#Expected Cmdline: ./vw_baselines.sh <vw_file_prefix> <no-click_encoding>
#Defaults for optional args:
#   <vw_file_prefix>        vw
#   <no-click_encoding>     0.999


VW_PREFIX=vw
if [[ $# -gt 0 ]]; then
    VW_PREFIX=$1
fi

NEG_LOSS=0.999
if [[ $# -gt 1 ]]; then
    NEG_LOSS=$2
fi

for method in dm ips dr
do
    mkdir -p Logs/${method}
    for P in 0 0.5 1
    do
        for L in 0.1 1 10
        do
            for Lambda in 0 0.00000001 0.000001 0.0001
            do
                echo "E1",${method},${P},${L},${Lambda}

                vw -d ${VW_PREFIX}_train.gz -c --compressed --save_resume -P 500000 --holdout_off --sort_features --noconstant --hash all -b 24 -l ${L} --power_t ${P} --l1 ${Lambda} -f Logs/${method}/${method}_${P}_${L}_${Lambda}.1 --random_weights 1 --id ${method}_${P}_${L}_${Lambda}.1 --cb_adf --cb_type ${method} &> Logs/${method}/${method}_${P}_${L}_${Lambda}.1.train.log

                vw -d ${VW_PREFIX}_validate.gz -c --compressed -P 500000 --holdout_off -i Logs/${method}/${method}_${P}_${L}_${Lambda}.1 -t --rank_all -p Logs/${method}/${method}_${P}_${L}_${Lambda}.1.val.txt &> Logs/${method}/${method}_${P}_${L}_${Lambda}.1.val.log

                python scorer.py Logs/${method}/${method}_${P}_${L}_${Lambda}.1.val.txt ${VW_PREFIX}_validate.gz ${NEG_LOSS} &> Logs/${method}_${P}_${L}_${Lambda}.1.val.scores

                vw -d ${VW_PREFIX}_test.gz -c --compressed -P 500000 --holdout_off -i Logs/${method}/${method}_${P}_${L}_${Lambda}.1 -t --rank_all -p Logs/${method}/${method}_${P}_${L}_${Lambda}.1.test.txt &> Logs/${method}/${method}_${P}_${L}_${Lambda}.1.test.log

                python scorer.py Logs/${method}/${method}_${P}_${L}_${Lambda}.1.test.txt ${VW_PREFIX}_test.gz ${NEG_LOSS} &> Logs/${method}_${P}_${L}_${Lambda}.1.test.scores

                rm Logs/${method}/${method}_${P}_${L}_${Lambda}.1.test.txt
                rm Logs/${method}/${method}_${P}_${L}_${Lambda}.1.val.txt
            done
        done
    done
done

for epoch in {2..40..1}
do
    for method in dm ips dr
    do
        for P in 0 0.5 1
        do
            for L in 0.1 1 10
            do
                for Lambda in 0 0.00000001 0.000001 0.0001
                do
                    echo ${epoch},${method},${P},${L},${Lambda}

                    vw --random_seed 387 -d ${VW_PREFIX}_train.gz -c --compressed --save_resume -P 500000 --holdout_off --sort_features --noconstant -l ${L} --power_t ${P} --l1 ${Lambda} -f Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch} -i Logs/${method}/${method}_${P}_${L}_${Lambda}.$((epoch-1)) --id ${method}_${P}_${L}_${Lambda}.${epoch} &> Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.train.log

                    vw --random_seed 387 -d ${VW_PREFIX}_validate.gz -c --compressed -P 500000 --holdout_off -i Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch} -t --rank_all -p Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.val.txt &> Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.val.log

                python scorer.py Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.val.txt ${VW_PREFIX}_validate.gz ${NEG_LOSS} &> Logs/${method}_${P}_${L}_${Lambda}.${epoch}.val.scores

                vw --random_seed 387 -d ${VW_PREFIX}_test.gz -c --compressed -P 500000 --holdout_off -i Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch} -t --rank_all -p Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.test.txt &> Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.test.log

                python scorer.py Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.test.txt ${VW_PREFIX}_test.gz ${NEG_LOSS} &> Logs/${method}_${P}_${L}_${Lambda}.${epoch}.test.scores

                #Cleanup -- to avoid massive disk footprint
                rm Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.test.txt
                rm Logs/${method}/${method}_${P}_${L}_${Lambda}.${epoch}.val.txt 
                rm Logs/${method}/${method}_${P}_${L}_${Lambda}.$((epoch-1)) 
                done
            done
        done
    done
done

#Cleanup -- to avoid massive disk footprint
for method in dm ips dr
do
    for P in 0 0.5 1
    do
        for L in 0.1 1 10
        do
            for Lambda in 0 0.00000001 0.000001 0.0001
            do
            rm Logs/${method}/${method}_${P}_${L}_${Lambda}.40
            done
        done
    done
done
'

#!/bin/zsh

# -----------------------------------------------------------------------------------------------
# Concatenation of all experiments for IRMA datasets under switch-off
head -1 switch_off/Switch-off_1.txt > switch_off/Switch-off_concatenated.tsv
foreach i (`seq 1 4`)
tail -n +2 switch_off/Switch-off_${i}.txt >> switch_off/Switch-off_concatenated.tsv
end
echo "Generated: switch_off/Switch-off_concatenated.tsv"

# Concatenation of all experiments for IRMA datasets under switch-on
head -1 switch_on/Switch-on_1.txt > switch_on/Switch-on_concatenated.tsv
foreach i (`seq 1 5`)
tail -n +2 switch_on/Switch-on_${i}.txt >> switch_on/Switch-on_concatenated.tsv
end
echo "Generated: switch_on/Switch-on_concatenated.tsv"

# -----------------------------------------------------------------------------------------------
# Concatenation of all experiments for IRMA datasets under switch-on and switch-off (mixed)
# switch-on_1 switch-off_1 switch-on_2 switch-off_2 ... switch-on_4 switch-off_4 switch-on_5
head -1 switch_on/Switch-on_1.txt > Switch-on-off_concatenated_mixed.tsv
foreach i (`seq 1 4`)
tail -n +2 switch_on/Switch-on_${i}.txt >> Switch-on-off_concatenated_mixed.tsv
tail -n +2 switch_off/Switch-off_${i}.txt >> Switch-on-off_concatenated_mixed.tsv
end
tail -n +2 switch_on/Switch-on_5.txt >> Switch-on-off_concatenated_mixed.tsv
echo "Generated: Switch-on-off_concatenated_mixed.tsv"
# -----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Concatenation of all experiments for IRMA datasets under switch-on and switch-off (seqeunced)
# switch-on_1 switch-on_2 ... switch-on_5 switch-off_1 switch-off_2 ... switch-off_4
head -1 switch_on/Switch-on_1.txt > Switch-on-off_concatenated_sequenced.tsv
foreach i (`seq 1 5`)
tail -n +2 switch_on/Switch-on_${i}.txt >> Switch-on-off_concatenated_sequenced.tsv
end

foreach i (`seq 1 4`)
tail -n +2 switch_off/Switch-off_${i}.txt >> Switch-on-off_concatenated_sequenced.tsv
end

echo "Generated: Switch-on-off_concatenated_sequenced.tsv"
# -----------------------------------------------------------------------------------------------

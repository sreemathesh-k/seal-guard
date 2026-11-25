#!/bin/bash
# Seal-Guard 2-minute demo. Press Enter to move to the next part.
cd ~/seal_guard && source .venv/bin/activate
R=~/executorch-rpi/executorch/cmake-out/examples/models/llama/llama_main
M=~/llama1b/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte
T=~/llama1b/tokenizer.model
p(){ echo; read -p ">>> $1  [Enter]" _; clear; }
clear
echo "SEAL-GUARD: offline blister-strip inspection + Llama 3.2 1B advice via ExecuTorch"
echo "Running on this laptop CPU (x86). No cloud, Wi-Fi off."
p "1. Inspection result for a defect strip"
explorer.exe "$(wslpath -w runs/e2e/defect_strip_overlay.png)" 2>/dev/null
python -c "import json;f=json.load(open('runs/e2e/defect_strip_facts.json'));print('verdict',f['verdict']);[print(' pocket',q['pocket'],q['type'],q['severity']) for q in f['pockets'] if q['verdict']=='FAIL']"
p "2. Llama 3.2 1B (ExecuTorch) writes the operator advice, fact-checked"
python advise_strip.py --facts runs/e2e/defect_strip_facts.json --log logs/sealguard.csv --llm executorch --runner $R --model $M --tokenizer $T
p "3. Tamper-evident log: verify, then edit a copy"
python advise_strip.py --verify logs/sealguard.csv
cp logs/sealguard.csv /tmp/tamper.csv && sed -i '0,/PASS/s//FAIL/' /tmp/tamper.csv
python advise_strip.py --verify /tmp/tamper.csv
p "4. Measured results, including misses"
grep -E "^---|PASS RATE" results/llama_passrate.txt | cut -c1-100
tail -7 results/real_classic.txt
echo "Not yet run on a Raspberry Pi 5. Real-photo empty-pocket recall is low (4/28): Phase 2 adds a light box."

@echo off
cd /d "%~dp0"
echo Running execution (5 questions) with --stub so Molmo is skipped. Results in results\2026-02-08_19-00-36\program_execution\
py -3.11 execute_only.py --dataset clevr --annotations-json data/clevr_subset/annotations.json --image-pth data/clevr_subset/images/ --results-pth results/2026-02-08_19-00-36 --api-json results/2026-02-08_19-00-36/api_generator/api.json --programs-json results/2026-02-08_19-00-36/program_generator/programs.json --num-questions 5 --stub
pause

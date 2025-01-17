# export OPENAI_API_KEY=""
export OPENAI_API_BASE="https://api.openai.com/v1/"
# python ../process.py --resident Agent.Resident.GPT --resident_openai_model_name gpt-3.5-turbo \
#     --evaluator Agent.Evaluator.GPT --evaluator_openai_model_name gpt-3.5-turbo \
#     --doctor Agent.Doctor.GPT --doctor_openai_model_name gpt-3.5-turbo \
#     --patient Agent.Patient.GPT --patient_openai_model_name gpt-3.5-turbo \
#     --ff_print True \

python ../score_evaluate.py --resident Agent.Resident.GPT --resident_openai_model_name gpt-3.5-turbo \
    --evaluator Agent.Evaluator.GPT --evaluator_openai_model_name gpt-3.5-turbo \
    --doctor Agent.Doctor.GPT --doctor_openai_model_name gpt-3.5-turbo \
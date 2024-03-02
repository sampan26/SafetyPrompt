model_names=(
    "microsoft/Orca-2-7b"
    "meta-llama/Llama-2-7b-chat-hf"
    "lmsys/vicuna-7b-v1.5"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "openchat/openchat-3.5"
    "codellama/CodeLlama-7b-Instruct-hf"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "openchat/openchat-3.5-1210"
)
full_model_names=()
for model_name in ${model_names[@]}; do
    full_model_names+=("${model_name}")
done

for system_prompt_type in "all"; do

python compare_pca_soft_harmfulness.py \
    --pretrained_model_paths ${model_names[@]} \
    --config sampling \
    --system_prompt_type ${system_prompt_type} \
    --output_path comparisons/pca_soft

done
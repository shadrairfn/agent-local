# Buat varian model dengan num_ctx besar. Butuh internet SEKALI untuk `ollama pull`;
# di PC lomba (offline) pull model dulu sebelum berangkat, atau salin folder ~/.ollama/models.
$models = @("qwen3:4b-instruct-2507-q4_K_M", "qwen3:8b")
foreach ($m in $models) { ollama pull $m }
ollama create qwen3-4b-ctx16k -f "$PSScriptRoot/Modelfile.qwen3-4b-ctx16k"
ollama create qwen3-4b-ctx24k -f "$PSScriptRoot/Modelfile.qwen3-4b-ctx24k"
ollama create qwen3-8b-ctx20k -f "$PSScriptRoot/Modelfile.qwen3-8b-ctx20k"
ollama list

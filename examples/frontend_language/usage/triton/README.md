# sglang_triton

Build the docker image:
```
docker build -t sglang-triton .
```

Then do:
```
docker run -ti --gpus=all --network=host --name sglang-triton -v ./models:/mnt/models sglang-triton
```

inside the docker container:
```
cd sglang
python3 -m sglang.launch_server --model-path mistralai/Mistral-7B-Instruct-v0.2 --port 30000 --mem-fraction-static 0.9
```

with another shell, inside the docker container:
```
docker exec -ti sglang-triton /bin/bash
cd /mnt
tritonserver --model-repository=/mnt/models
```


Send request to the server:
```
curl -X POST http://localhost:8000/v2/models/character_generation/generate \
-H "Content-Type: application/json" \
-d '{
  "INPUT_TEXT": ["harry"]
}'

```

## Running as a non-root user

The image installs SGLang under `/opt/sglang`, which is owned by root and not writable by group or others. To run the container as a non-root user, mount a writable cache directory and point the relevant environment variables at it:

```bash
mkdir -p ext_cache
docker run -ti --gpus=all --network=host --user $(id -u) \
  -v "$PWD/ext_cache:/ext_cache" \
  -e TORCH_EXTENSIONS_DIR=/ext_cache \
  -e SGLANG_CACHE_DIR=/ext_cache \
  -v ./models:/mnt/models sglang-triton
```

Without this, JIT-compiled kernels and torch extension builds may fail because the default cache locations are not writable.

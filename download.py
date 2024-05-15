from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rrsong/utc-base",
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    local_dir="/root/autodl-tmp/models/utc_base_pytorch",
)
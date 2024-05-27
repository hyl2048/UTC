from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rrsong/utc-base",
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    local_dir="/root/models/utc_base_pytorch",
)
# from paddlenlp.transformers import UTC, ErnieTokenizer

# tokenier = ErnieTokenizer.from_pretrained("utc-base")
# model = UTC.from_pretrained("utc-base")
# print(tokenier, model)

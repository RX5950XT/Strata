# writes /content/strata-ds.json, the server's run config; the engine flags after the script name
import json, sys
args = ["--pack", "/content/m/maya-ds4", "--max-context", "204800", "--eos", "1", *sys.argv[1:]]
cfg = {"exe": "/content/eng/strata-dsv41", "args": args, "cwd": "/content/Strata", "tokenizer": "/content/m/maya-ds4/tokenizer",
       "model_name": "deepseek-v4.1-flash-uncensored-iq2_xxs", "arch": "deepseek41", "log": "/content/logs/strata-ds.log",
       "lib_dirs": ["/content/eng", "/usr/lib64-nvidia"], "port": 8080, "engine_ready_s": 3600}
json.dump(cfg, open("/content/strata-ds.json", "w"), indent=1)
print(json.dumps(cfg))

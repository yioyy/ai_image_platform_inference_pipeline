## 1. Env config files
- [x] 1.1 Add `radax/config/radax-test.env`
- [x] 1.2 Add `radax/config/radax-official.env`
- [x] 1.3 Add `radax/config/radax.env` (active config)
- [x] 1.4 Add `radax/config/load_env.py` loader (KEY=VALUE)

## 2. Aneurysm pipeline (python)
- [x] 2.1 Read env file automatically in `__main__`
- [x] 2.2 Replace hardcoded `api_url` with env-driven URL
- [x] 2.3 Replace hardcoded path variables in `__main__` with env-driven paths

## 3. Aneurysm runner (bash)
- [x] 3.1 Update `radax/pipeline_aneurysm.sh` to source env file
- [x] 3.2 Update `radax/pipeline_aneurysm.sh` to use env for conda env / python / script path

## 4. Spec delta
- [x] 4.1 Add spec delta describing required env keys and behavior

